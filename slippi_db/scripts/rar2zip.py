"""Convert rar files to zip for faster processing.

Requires `unrar` and `7z` to be installed.
"""

import os
import shutil
import subprocess
import tempfile
import typing as tp
from pathlib import Path

import tqdm
from absl import app, flags, logging

from slippi_db import utils

INPUT = flags.DEFINE_string('input', None, 'Input file or directory.', required=True)
OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'Output directory. If not specified, converts in-place.')
IN_MEMORY = flags.DEFINE_bool('in_memory', True, 'Use in-memory temporary files for conversion.')
REMOVE_ORIGINAL = flags.DEFINE_bool('remove_original', False, 'Remove original rar files after successful conversion.')
CHECK_SPACE = flags.DEFINE_bool('check_space', True, 'Skip archives that do not fit in the temporary and output filesystems.')

# Deflate (what `7z a -tzip` uses) compresses worse than rar's default -m3, so
# scale up the packed size to estimate the zip we produce.
ZIP_SIZE_FACTOR = 1.5

def check_dependencies():
  for tool in ['unrar', '7z']:
    if shutil.which(tool) is None:
      raise RuntimeError(f'{tool} is not installed. Please install it.')

class ArchiveInfo(tp.NamedTuple):
  num_entries: int  # files and directories
  num_files: int
  unpacked_size: int
  packed_size: int

  @property
  def estimated_zip_size(self) -> int:
    return int(self.packed_size * ZIP_SIZE_FACTOR)

  @property
  def estimated_tmp_size(self) -> int:
    """Peak temp usage: the extracted files plus the zip written beside them."""
    return self.unpacked_size + self.estimated_zip_size

def read_archive_info(path: str) -> ArchiveInfo:
  """Parses `unrar lt`, which lists sizes along with the entry names."""
  output = subprocess.run(
      ['unrar', 'lt', path],
      check=True, capture_output=True, text=True).stdout

  num_entries = num_files = 0
  unpacked_size = packed_size = 0
  for line in output.splitlines():
    key, _, value = line.strip().partition(': ')
    if key == 'Name':
      num_entries += 1
    elif key == 'Type' and value == 'File':
      num_files += 1
    elif key == 'Size':
      unpacked_size += int(value)
    elif key == 'Packed size':
      packed_size += int(value)

  return ArchiveInfo(
      num_entries=num_entries, num_files=num_files,
      unpacked_size=unpacked_size, packed_size=packed_size)

def format_size(num_bytes: int) -> str:
  return f'{num_bytes / 1024 ** 3:.2f} GiB'

def check_free_space(info: ArchiveInfo, tmp_dir: str, output_dir: str) -> None:
  """Raises if the estimated extraction won't fit."""
  needs = {tmp_dir: info.estimated_tmp_size, output_dir: info.estimated_zip_size}

  # A rename within one filesystem is free, but the zip does coexist with the
  # extracted files, so a shared filesystem needs the sum.
  if os.stat(tmp_dir).st_dev == os.stat(output_dir).st_dev:
    needs = {tmp_dir: info.estimated_tmp_size}

  for path, needed in needs.items():
    free = shutil.disk_usage(path).free
    if free < needed:
      raise RuntimeError(
          f'Not enough space in {path}: need ~{format_size(needed)}, '
          f'have {format_size(free)}.')

def run_with_progress(
    cmd: list[str],
    total: int,
    desc: str,
    is_progress_line: tp.Callable[[str], bool],
    cwd: tp.Optional[str] = None,
) -> None:
  """Run cmd, showing a tqdm bar advanced by matching output lines."""
  proc = subprocess.Popen(
      cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
      text=True, errors='replace')
  with tqdm.tqdm(total=total, desc=desc, unit='file', leave=False) as bar:
    output = []
    for line in proc.stdout:
      output.append(line)
      if is_progress_line(line):
        bar.update(1)
  if proc.wait() != 0:
    print(''.join(output))
    raise subprocess.CalledProcessError(proc.returncode, cmd)

def convert(
    input_path: str,
    output_path: str,
    in_memory: bool = True,
) -> None:
  input_path = os.path.abspath(input_path)
  output_path = os.path.abspath(output_path)

  info = read_archive_info(input_path)
  tmp_dir = utils.get_tmp_dir(in_memory=in_memory) or tempfile.gettempdir()
  print(
      f'{info.num_files} files, {format_size(info.unpacked_size)} extracted;'
      f' needs ~{format_size(info.estimated_tmp_size)} in {tmp_dir}')
  if CHECK_SPACE.value:
    check_free_space(info, tmp_dir, os.path.dirname(output_path))

  with tempfile.TemporaryDirectory(dir=tmp_dir) as tmpdir:
    extract_dir = os.path.join(tmpdir, 'extracted')
    os.makedirs(extract_dir)
    # `unrar x` prints an Extracting/Creating line per file and directory.
    run_with_progress(
        ['unrar', 'x', '-o+', '-idcp', input_path, extract_dir + '/'],
        total=info.num_entries, desc='extract',
        # 'Extracting from <archive>' is a per-volume header, not an entry.
        is_progress_line=lambda l: (
            l.startswith(('Extracting ', 'Creating '))
            and not l.startswith('Extracting from ')),
    )

    # Write the zip to the temp dir first so a failed conversion doesn't
    # leave a partial zip at the output path.
    tmp_zip = os.path.join(tmpdir, 'out.zip')
    run_with_progress(
        ['7z', 'a', '-tzip', '-mmt=on', '-bb1', '-bd', tmp_zip, '*'],
        total=info.num_files, desc='zip',
        is_progress_line=lambda l: l.startswith('+ '),
        cwd=extract_dir,
    )
    shutil.move(tmp_zip, output_path)

def process_single_rar(input_path: Path, output_dir: Path) -> bool:
  output_path = output_dir / (input_path.stem + '.zip')
  print(f'Converting {input_path} to {output_path}')
  try:
    convert(str(input_path), str(output_path), in_memory=IN_MEMORY.value)
    if REMOVE_ORIGINAL.value:
      input_path.unlink()
      logging.info(f'Removed original file: {input_path}')
    return True
  except Exception as e:
    logging.error(f'Failed to convert {input_path}: {e}')
    return False

def main(_):
  check_dependencies()
  input_path = Path(INPUT.value)

  if input_path.is_file():
    if input_path.suffix != '.rar':
      raise ValueError(f'Input file must be a .rar file: {input_path}')

    if OUTPUT_DIR.value:
      output_dir = Path(OUTPUT_DIR.value)
      output_dir.mkdir(parents=True, exist_ok=True)
    else:
      output_dir = input_path.parent

    process_single_rar(input_path, output_dir)
  elif input_path.is_dir():
    output_dir = None
    if OUTPUT_DIR.value:
      output_dir = Path(OUTPUT_DIR.value)
      if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f'Output path must be a directory: {output_dir}')

    rar_files = list(input_path.rglob('*.rar'))
    logging.info(f'Found {len(rar_files)} rar files to process')

    successful_conversions = 0
    for rar_file in tqdm.tqdm(rar_files, desc='Converting files'):
      if output_dir:
        rel_path = rar_file.relative_to(input_path)
        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
      else:
        output_subdir = rar_file.parent

      if process_single_rar(rar_file, output_subdir):
        successful_conversions += 1

    logging.info(f'Successfully converted {successful_conversions}/{len(rar_files)} files')
  else:
    raise ValueError(f'Input path does not exist: {input_path}')

if __name__ == '__main__':
  app.run(main)
