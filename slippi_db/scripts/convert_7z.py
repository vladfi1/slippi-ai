"""Convert 7z files to zip for faster processing."""

import itertools
import subprocess
import os
import py7zr
import subprocess
import tempfile
import tqdm
import typing as tp
import zipfile
from pathlib import Path

from absl import app, flags, logging

from slippi_db import utils

T = tp.TypeVar('T')

def extract_file_list(
    path: str,
    files: tp.Iterable[str],
    output_dir: str,
) -> None:
  """Extract files from a 7z archive."""
  path = os.path.abspath(path)

  with tempfile.NamedTemporaryFile() as input_list:
    for file in files:
      input_list.write(f'{file}\n'.encode('utf-8'))
    input_list.seek(0)

    # TODO: is there a way to do this multithreaded?
    subprocess.check_call(
        ['7z', 'x', path, f'@{input_list.name}'],
        cwd=output_dir,
    )

def convert(
    input_path: str,
    output_path: str,
    max_chunk_size_gb: float = 16,  # uncompressed
    in_memory: bool = True,
    work_dir: tp.Optional[str] = None,
) -> None:
  cwd = os.getcwd()
  input_path = os.path.abspath(input_path)
  output_path = os.path.abspath(output_path)

  chunks = utils.traverse_7z_fast(input_path, chunk_size_gb=max_chunk_size_gb)

  with tempfile.TemporaryDirectory(dir=work_dir) as zipdir:
    zip_paths = []
    for i, chunk in enumerate(tqdm.tqdm(chunks, smoothing=0)):

      with tempfile.TemporaryDirectory(dir=utils.get_tmp_dir(in_memory=in_memory)) as tmpdir:
        extract_file_list(input_path, chunk.files, tmpdir)

        # zip all from tmpdir
        zip_path = os.path.join(zipdir, f'{i}.zip')
        zip_paths.append(zip_path)
        subprocess.check_call(['7z', '-tzip', 'a', zip_path, '*'], cwd=tmpdir)

    # combine all zip files
    if len(zip_paths) == 1:
      os.rename(zip_paths[0], output_path)
    else:
      subprocess.check_call(['zipmerge', output_path, *zip_paths])

  archive = py7zr.SevenZipFile(input_path, 'r')
  zip_archive = zipfile.ZipFile(output_path)
  for sf in archive.files:
    if sf.is_directory:
      continue
    info = zip_archive.getinfo(sf.filename)
    assert info.file_size == sf.uncompressed

  os.chdir(cwd)  # for line_profiler

INPUT = flags.DEFINE_string('input', None, 'Input file or directory.', required=True)
OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'Output directory. If not specified, converts in-place.')
CHUNK_SIZE = flags.DEFINE_float('chunk_size', 1, 'Max chunk size in GB.')
IN_MEMORY = flags.DEFINE_bool('in_memory', True, 'Use in-memory temporary files for conversion.')
WORK_DIR = flags.DEFINE_string('work_dir', None, 'Optional working directory for temporary zip files.')
REMOVE_ORIGINAL = flags.DEFINE_bool('remove_original', False, 'Remove original 7z files after successful conversion.')

def process_single_7z(input_path: Path, output_dir: Path):
  """Process a single 7z file."""
  output_name = input_path.stem + '.zip'
  output_path = output_dir / output_name
  print(f'Converting {input_path} to {output_path}')
  try:
    convert(
        str(input_path),
        str(output_path),
        max_chunk_size_gb=CHUNK_SIZE.value,
        in_memory=IN_MEMORY.value,
        work_dir=WORK_DIR.value,
    )
    if REMOVE_ORIGINAL.value:
      input_path.unlink()
      logging.info(f'Removed original file: {input_path}')
    return True
  except Exception as e:
    logging.error(f'Failed to convert {input_path}: {e}')
    return False


def main(_):
  input_path = Path(INPUT.value)

  if input_path.is_file():
    # Process single file
    if not input_path.suffix == '.7z':
      raise ValueError(f'Input file must be a .7z file: {input_path}')

    # Determine output directory
    if OUTPUT_DIR.value:
      output_dir = Path(OUTPUT_DIR.value)
      output_dir.mkdir(parents=True, exist_ok=True)
    else:
      # In-place conversion - use parent directory of input file
      output_dir = input_path.parent

    process_single_7z(input_path, output_dir)
  elif input_path.is_dir():
    # Process directory recursively
    output_dir = None
    if OUTPUT_DIR.value:
      output_dir = Path(OUTPUT_DIR.value)
      if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f'Output path must be a directory: {output_dir}')

    # Find all 7z files recursively
    seven_z_files = list(input_path.rglob('*.7z'))
    logging.info(f'Found {len(seven_z_files)} 7z files to process')

    successful_conversions = 0
    for seven_z_file in tqdm.tqdm(seven_z_files, desc='Converting files'):
      if output_dir:
        # Calculate relative path from input directory
        rel_path = seven_z_file.relative_to(input_path)

        # Create output subdirectory maintaining structure
        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
      else:
        # In-place conversion - use parent directory of each 7z file
        output_subdir = seven_z_file.parent

      if process_single_7z(seven_z_file, output_subdir):
        successful_conversions += 1

    logging.info(f'Successfully converted {successful_conversions}/{len(seven_z_files)} files')
  else:
    raise ValueError(f'Input path does not exist: {input_path}')

if __name__ == '__main__':
  app.run(main)
