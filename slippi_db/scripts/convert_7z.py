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

from absl import app, flags

from slippi_db import utils

T = tp.TypeVar('T')

def chunked(iterable: tp.Iterable[T], n: int) -> tp.Iterator[tp.Iterator[T]]:
  """Maximally lazy chunking."""
  it = iter(iterable)
  while True:
    chunk = itertools.islice(it, n)

    try:
      first = next(chunk)
    except StopIteration:
      break

    yield itertools.chain([first], chunk)

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
    in_memory: bool = False,
) -> None:
  cwd = os.getcwd()
  input_path = os.path.abspath(input_path)
  output_path = os.path.abspath(output_path)
  archive = py7zr.SevenZipFile(input_path, 'r')

  # calculate optimal chunks
  folders = archive.header.main_streams.unpackinfo.folders

  max_chunk_size = max_chunk_size_gb * 1024**3
  chunks: list[list[str]] = []
  chunk_size = 0
  chunk: list[str] = []

  for folder in folders:
    for file in folder.files:
      if chunk_size + file.uncompressed > max_chunk_size:
        chunks.append(chunk)
        chunk = []
        chunk_size = 0

      chunk_size += file.uncompressed
      chunk.append(file.filename)

  # commit last chunk
  if chunk:
    chunks.append(chunk)

  print('Chunks:', len(chunks))

  # relpaths = [p for p in archive.getnames() if p.endswith('.slp')]
  # relpaths = reversed(relpaths)
  # chunks = chunked(tqdm.tqdm(relpaths), chunk_size)

  with tempfile.TemporaryDirectory() as zipdir:
    zip_paths = []
    for i, chunk in enumerate(tqdm.tqdm(chunks, smoothing=0)):

      # with tempfile.TemporaryDirectory() as tmpdir:
      with tempfile.TemporaryDirectory(dir=utils.get_tmp_dir(in_memory=in_memory)) as tmpdir:
        extract_file_list(input_path, chunk, tmpdir)

        # zip all from tmpdir
        zip_path = os.path.join(zipdir, f'{i}.zip')
        zip_paths.append(zip_path)
        subprocess.check_call(['7z', '-tzip', 'a', zip_path, '*'], cwd=tmpdir)

    # combine all zip files
    subprocess.check_call(['zipmerge', output_path, *zip_paths])

  zip_archive = zipfile.ZipFile(output_path)
  for sf in archive.files:
    if sf.is_directory:
      continue
    info = zip_archive.getinfo(sf.filename)
    assert info.file_size == sf.uncompressed

  os.chdir(cwd)  # for line_profiler

INPUT = flags.DEFINE_string('input', None, 'Input path.', required=True)
OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'Output directory.')
CHUNK_SIZE = flags.DEFINE_float('chunk_size', 1, 'Max chunk size in GB.')

# TODO: recurse directories and remove old 7z files

def main(_):
  output_dir = OUTPUT_DIR.value or os.path.dirname(INPUT.value)
  os.makedirs(output_dir, exist_ok=True)
  output_name = os.path.basename(INPUT.value).removesuffix('.7z') + '.zip'
  output_path = os.path.join(output_dir, output_name)
  print('Converting', INPUT.value, 'to', output_path)
  convert(INPUT.value, output_path, max_chunk_size_gb=CHUNK_SIZE.value)

if __name__ == '__main__':
  app.run(main)
