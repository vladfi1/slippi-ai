import abc
import concurrent.futures
from contextlib import contextmanager
import functools
import gzip
import hashlib
import io
import os
from typing import Generator

import subprocess
import sys
import tempfile
import time
import typing as tp
import zipfile

import numpy as np
import py7zr


T = tp.TypeVar('T')

class Timer:

  def __init__(self, name: str, verbose=True):
    self.name = name
    self.verbose = verbose

  def __enter__(self):
    self.start = time.perf_counter()

  def __exit__(self, *_):
    self.duration = time.perf_counter() - self.start
    if self.verbose:
      print(f'{self.name}: {self.duration:.1f}')

def monitor(
    futures: tp.Dict[concurrent.futures.Future[T], str],
    log_interval: int = 10,
) -> tp.Iterator[T]:
  total_items = len(futures)
  remaining_items = set(futures)
  last_log = time.perf_counter()
  finished_items = 0
  items_since_last_log = 0
  for finished in concurrent.futures.as_completed(futures):
    current_time = time.perf_counter()

    finished_items += 1
    items_since_last_log += 1
    remaining_items.remove(finished)

    if current_time - last_log > log_interval:
      run_time = current_time - last_log
      items_per_second = items_since_last_log / run_time
      num_items_remaining = total_items - finished_items
      estimated_time_remaining = num_items_remaining / items_per_second
      progress_percent = finished_items / total_items

      print(
          f'{finished_items}/{total_items} = {100 * progress_percent:.1f}%'
          f' rate={items_per_second:.1f}'
          f' eta={estimated_time_remaining:.0f}'
      )
      # display one of the remaining items
      if num_items_remaining:
        remaining_item = next(iter(remaining_items))
        print('remaining: ' + futures[remaining_item])

      last_log = current_time
      items_since_last_log = 0

    yield finished.result()

def md5(b: bytes) -> str:
  return hashlib.md5(b).hexdigest()

_MACOS_SHM_DISK = 'ramdisk'
_MACOS_SHM_SIZE = 1024 # MB

@functools.cache
def get_tmp_dir(in_memory: bool) -> tp.Optional[str]:
  if not in_memory:
    return None

  if os.path.exists('/dev/shm'):
    return '/dev/shm'

  # from https://stackoverflow.com/questions/2033362/does-os-x-have-an-equivalent-to-dev-shm
  if sys.platform == 'darwin':
    shm_dir = os.path.join('/Volumes', _MACOS_SHM_DISK)
    if os.path.exists(shm_dir):
      return shm_dir

    p = subprocess.run(
      ['hdiutil', 'attach', '-nomount', f'ram://{2 * 1024 * _MACOS_SHM_SIZE}'],
      capture_output=True, check=True)
    disk_name = p.stdout.decode('utf-8').strip()
    subprocess.check_call(['diskutil', 'eraseVolume', 'APFS', _MACOS_SHM_DISK, disk_name])
    print(f'Created ramdisk {disk_name} at {shm_dir}')

    # def cleanup():
    #   subprocess.check_call(['umount', shm_dir])
    #   subprocess.check_call(['hdiutil', 'detach', disk_name])
    # atexit.register(cleanup)

    return shm_dir

  raise RuntimeError('No in-memory tmp dir available')

def extract_zip(src: str, dst_dir: str):
  with Timer("unzip"):
    subprocess.check_call(
        ['unzip', src],
        cwd=dst_dir,
        stdout=subprocess.DEVNULL)

def extract_7z(src: str, dst_dir: str):
  with Timer("7z x"):
    subprocess.check_call(
        ['7z', 'x', src, '-o' + dst_dir],
        stdout=subprocess.DEVNULL)

class FileReadException(Exception):
  """Failed to read a file."""

class LocalFile(abc.ABC):
  """Identifies a file on the local system."""

  @abc.abstractmethod
  def read(self) -> bytes:
    """Read the file bytes."""

  @contextmanager
  @abc.abstractmethod
  @contextmanager
  def extract(self, tmpdir: str) -> tp.Generator[str, None, None]:
    """Extract the file to a temporary directory and return it."""

  def md5(self) -> str:
    return md5(self.read())

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """The file's name."""

class SimplePath(LocalFile):

  def __init__(self, root: str, path: str):
    self.root = root
    self.path = path

  @property
  def name(self) -> str:
    return self.path

  @contextmanager
  def extract(self, tmpdir: str) -> Generator[str, None, None]:
    del tmpdir
    yield os.path.join(self.root, self.path)

  def read(self):
    with open(os.path.join(self.root, self.path), 'rb') as f:
      return f.read()

GZ_SUFFIX = '.gz'

class GZipFile(LocalFile):
  """A gzipped file."""

  def __init__(self, root: str, path: str):
    self.root = root
    self.path = path
    if not path.endswith(GZ_SUFFIX):
      raise ValueError(f'{root}/{path} is not a gz file?')

  @property
  def name(self) -> str:
    return self.path.removesuffix(GZ_SUFFIX)

  def read(self) -> bytes:
    with gzip.open(os.path.join(self.root, self.path)) as f:
      return f.read()

  @contextmanager
  def extract(self, tmpdir: str) -> Generator[str, None, None]:
    path = os.path.join(tmpdir, self.name)
    try:
      with open(path, 'wb') as f:
        f.write(self.read())
      yield path
    finally:
      os.remove(path)

class SevenZipFile(LocalFile):
  """File inside a 7z archive."""

  def __init__(self, root: str, path: str):
    self.root = root
    self.path = path

  @property
  def name(self) -> str:
    return self.path

  def read(self) -> bytes:
    result = subprocess.run(
        ['7z', 'e', '-so', self.root, self.path],
        capture_output=True, check=True)
    return result.stdout

  @contextmanager
  def extract(self, tmpdir: str) -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory(dir=tmpdir) as tmp_dir:
      subprocess.check_call(
        ['7z', 'x', '-o' + tmp_dir, self.root, self.path],
        stdout=subprocess.DEVNULL)
      # with py7zr.SevenZipFile(self.root) as archive:
      #   archive.extract(path=tmpdir, targets=[self.path])
      yield os.path.join(tmp_dir, self.path)

class ZipFile(LocalFile):
  """File inside a zip archive."""

  def __init__(self, root: str, path: str, raw: tp.Optional[bytes] = None):
    self.root = root
    self.path = path
    self.raw = raw

    suffix_found = False
    for suffix in VALID_SUFFIXES:
      if path.endswith(suffix):
        suffix_found = True
        self.suffix = suffix
        break

    if not suffix_found:
      raise ValueError(f'{root}/{path} is not a valid Slippi file?')

    self.base_name = self.path.removesuffix(self.suffix)

    self.is_gzipped = path.endswith(GZ_SUFFIX)
    self.is_slpz = path.endswith(SLPZ_SUFFIX)

  @property
  def name(self) -> str:
    return self.base_name + _SLP_SUFFIX

  def read_raw(self) -> bytes:
    if self.raw is not None:
      return self.raw

    try:
      result = subprocess.run(
          ['unzip', '-p', self.root, self.path],
          check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
      raise FileReadException(e.stderr.decode()) from e
    return result.stdout

  def from_raw(self, data: bytes) -> bytes:
    if self.is_gzipped:
      data = gzip.decompress(data)

    if self.is_slpz:
      try:
        result = subprocess.run(
            ['slpz', '-d', '-o', '-', '-'],
            capture_output=True, input=data, check=True)
      except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr.decode()) from e
      data = result.stdout

    return data

  def read(self) -> bytes:
    return self.from_raw(self.read_raw())

  @contextmanager
  def extract(self, tmpdir: str) -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir:
      path = os.path.join(tmpdir, 'game.slp')
      with open(path, 'wb') as f:
        f.write(self.read())
      yield path

def traverse_slp_files(root: str) -> list[LocalFile]:
  files = []
  for abspath, _, filenames in os.walk(root):
    for name in filenames:
      if name.endswith('.slp'):
        reldir = os.path.relpath(abspath, root)
        relpath = os.path.join(reldir, name)
        files.append(SimplePath(root, relpath))
      elif name.endswith('.slp.gz'):
        reldir = os.path.relpath(abspath, root)
        relpath = os.path.join(reldir, name)
        files.append(GZipFile(root, relpath))

  return files

def traverse_slp_files_7z(root: str) -> list[SevenZipFile]:
  files = []
  relpaths = py7zr.SevenZipFile(root).getnames()
  for path in relpaths:
    if path.endswith('.slp'):
      files.append(SevenZipFile(root, path))
  return files

class SevenZipChunk:

  def __init__(self, path: str, files: list[str]) -> None:
    self.path = path
    self.files = files

  @contextmanager
  def extract(self, in_memory: bool = True) -> Generator[list[LocalFile], None, None]:
    """Extract the chunk to a temporary directory and return the files."""
    with tempfile.TemporaryDirectory(dir=get_tmp_dir(in_memory=in_memory)) as tmpdir:
      py7zr.SevenZipFile(self.path).extract(targets=self.files, path=tmpdir)
      yield [SimplePath(tmpdir, f) for f in self.files]

def traverse_7z_fast(
    path: str,
    chunk_size_gb: float = 0.5,
) -> list[SevenZipChunk]:
# ) -> tuple[tp.Iterator[list[LocalFile]], int]:
  """Efficiently iterate through a 7z archive."""
  archive = py7zr.SevenZipFile(path, 'r')

  # calculate optimal chunks
  assert archive.header.main_streams is not None
  assert archive.header.main_streams.unpackinfo is not None
  folders = archive.header.main_streams.unpackinfo.folders

  max_chunk_size = chunk_size_gb * 1024**3

  folder_sizes = []
  for folder in folders:
    folder_sizes.append(sum(file.uncompressed for file in folder.files))
  folder_sizes = np.array(folder_sizes)

  print(
    'folder sizes: num={num}, mean={mean:.1f}, std={std:.1f}'.format(
      num=len(folder_sizes), mean=folder_sizes.mean(), std=folder_sizes.std()))

  # First pack small folders into chunks
  small_folders = [
    i for i, size in enumerate(folder_sizes)
    if size <= max_chunk_size]

  chunks = []
  chunk = []
  chunk_size = 0

  for i in small_folders:
    folder = folders[i]
    size = folder_sizes[i]

    if chunk_size + size > max_chunk_size:
      chunks.append(chunk)
      chunk = []
      chunk_size = 0

    chunk_size += size
    chunk.extend([file.filename for file in folder.files])

  # commit last chunk
  if chunk:
    chunks.append(chunk)

  # Then split large folders into chunks
  large_folders = [
    i for i, size in enumerate(folder_sizes)
    if size > max_chunk_size]

  for i in large_folders:
    folder = folders[i]
    chunk = []
    chunk_size = 0
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

  print(f'Split {len(folders)} folders into {len(chunks)} chunks')
  print(f'Folders per chunk = {len(folders) / len(chunks):.1f}')

  return [SevenZipChunk(path, chunk) for chunk in chunks]

_SLP_SUFFIX = '.slp'
SLPZ_SUFFIX = '.slpz'
VALID_SUFFIXES = [
    _SLP_SUFFIX,
    _SLP_SUFFIX + GZ_SUFFIX,
    SLPZ_SUFFIX,
]

def is_slp_file(path: str) -> bool:
  """Check if the file is a valid Slippi replay file."""
  return any(path.endswith(s) for s in VALID_SUFFIXES)

def traverse_slp_files_zip(root: str) -> list[ZipFile]:
  files = []
  relpaths = zipfile.ZipFile(root).namelist()
  for path in relpaths:
    if is_slp_file(path):
      files.append(ZipFile(root, path))
  return files

def extract_zip_files(source_zip: str, file_names: list[str], dest_zip: str) -> None:
  """Extracts specified files without recompressing."""

  if os.path.exists(dest_zip):
    raise FileExistsError(f'Destination zip file {dest_zip} already exists')

  with subprocess.Popen(
      ['zip',  '-U', source_zip, '-@', '--out', dest_zip],
      stdin=subprocess.PIPE) as zip_proc:
    assert zip_proc.stdin is not None
    for file_name in file_names:
      zip_proc.stdin.write(file_name.encode('utf-8'))
      zip_proc.stdin.write(b'\n')
    zip_proc.stdin.close()
    zip_proc.wait()

def copy_zip_files(
    source_zip: str,
    file_names: list[str],
    dest_zip: str,
    tmpdir: tp.Optional[str] = None,
) -> None:
  """Copies specified files from source zip archive to destination zip archive.

  Extracts specified files from the source archive and adds them to the destination
  archive. If the destination archive doesn't exist, it will be created.

  Args:
    source_zip: Path to the source zip archive.
    file_names: List of file names within the source archive to copy.
    dest_zip: Path to the destination zip archive.
  """

  if os.path.exists(dest_zip):
    with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir:
      tmp_zip = os.path.join(tmpdir, 'tmp.zip')
      extract_zip_files(source_zip, file_names, tmp_zip)
      # TODO: add -k for libzip >= 1.10.0 to keep uncompressed files uncompressed
      subprocess.check_call(['zipmerge', dest_zip, tmp_zip])

  else:
    extract_zip_files(source_zip, file_names, dest_zip)

def copy_multi_zip_files(
    sources_and_files: list[tuple[str, list[str]]],
    dest_zip: str,
    tmpdir: tp.Optional[str] = None,
) -> None:
  """Copies specified files from multiple source zip archives to destination zip archive.

  Extracts specified files from the source archives and adds them to the destination
  archive. If the destination archive doesn't exist, it will be created.

  Args:
    sources_and_files: List of tuples, where each tuple contains a source zip archive path
      and a list of file names within the source archive to copy.
    dest_zip: Path to the destination zip archive.
  """

  with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir:
    tmp_zips = []
    for i, (source_zip, file_names) in enumerate(sources_and_files):
      tmp_zip = os.path.join(tmpdir, f'tmp_{i}.zip')
      extract_zip_files(source_zip, file_names, tmp_zip)
      tmp_zips.append(tmp_zip)

    # TODO: add -k for libzip >= 1.10.0 to keep uncompressed files uncompressed
    subprocess.check_call(['zipmerge', dest_zip, *tmp_zips])

def rename_within_zip(zip_path: str, to_rename: list[tuple[str, str]]) -> None:
  """Renames files within a zip archive.

  Args:
    zip_path: Path to the zip archive.
    to_rename: List of (old, new) file names.
  """

  # Note: zipnote is very picky about the input format.
  with subprocess.Popen(['zipnote', zip_path], stdout=subprocess.PIPE) as proc:
    assert proc.stdout is not None
    lines = proc.stdout.readlines()
    proc.wait()

  rename_mapping = {}
  for src, dst in to_rename:
    rename_mapping[f'@ {src}\n'.encode('utf-8')] = f'@={dst}\n'.encode('utf-8')

  with subprocess.Popen(['zipnote', '-w', zip_path], stdin=subprocess.PIPE) as proc:
    assert proc.stdin is not None
    for line in lines:
      proc.stdin.write(line)
      if line in rename_mapping:
        proc.stdin.write(rename_mapping.pop(line))
    proc.stdin.close()
    proc.wait()

def delete_from_zip(zip_path: str, file_names: list[str]) -> None:
  """Deletes specified files from a zip archive.

  Args:
    zip_path: Path to the zip archive.
    file_names: List of file names within the archive to delete.
  """
  if not os.path.exists(zip_path):
    raise FileNotFoundError(f'Zip file {zip_path} does not exist')

  if not file_names:
    return

  with subprocess.Popen(
      ['zip', '-d', '-q', zip_path, '-@'],
      stdin=subprocess.PIPE) as zip_proc:
    assert zip_proc.stdin is not None
    for file_name in file_names:
      zip_proc.stdin.write(file_name.encode('utf-8'))
      zip_proc.stdin.write(b'\n')
    zip_proc.stdin.close()
    zip_proc.wait()
    if zip_proc.returncode != 0:
      raise subprocess.CalledProcessError(zip_proc.returncode, ['zip', '-d', zip_path, '-@'])

def stream_files_zip(
    archive_path: str,
    file_limit: tp.Optional[int] = None,
) -> tp.Iterator[tuple[str, bytes]]:

  with zipfile.ZipFile(archive_path) as zf:
    infolist = zf.infolist()

  infolist = [info for info in infolist if not info.is_dir()]

  if file_limit is not None:
    infolist = infolist[:file_limit]

  # Start unzip process to stream all files
  proc = subprocess.Popen(
      ['unzip', '-p', archive_path],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      # bufsize=0  # Unbuffered for immediate streaming
  )

  try:
    assert isinstance(proc.stdout, io.BufferedReader)

    # Read each file's content based on expected size
    for info in infolist:
      expected_size = info.file_size

      if expected_size == 0:
        # Handle empty files
        yield (info.filename, b'')
        continue

      # Read exactly expected_size bytes using a pre-allocated buffer
      buffer = bytearray(expected_size)
      bytes_read = 0
      while bytes_read < expected_size:
        chunk_size = proc.stdout.readinto(memoryview(buffer)[bytes_read:])
        if chunk_size is None or chunk_size == 0:
          break
        bytes_read += chunk_size

      content = bytes(buffer[:bytes_read])

      if len(content) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes for {info.filename}, got {len(content)}")

      yield (info.filename, content)

    if file_limit is None:
      proc.wait()

  finally:
    proc.terminate()

def stream_slp_files_zip(
    archive_path: str,
    file_limit: tp.Optional[int] = None,
) -> tp.Iterator[ZipFile]:
  """Streams Slippi files from a zip archive."""

  for file_name, raw_data in stream_files_zip(archive_path, file_limit=file_limit):
    yield ZipFile(archive_path, file_name, raw=raw_data)
