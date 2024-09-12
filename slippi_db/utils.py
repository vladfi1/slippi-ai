import abc
import concurrent.futures
from contextlib import contextmanager
import functools
import gzip
import hashlib
import numpy as np
import os
from typing import Generator
import py7zr
import subprocess
import sys
import tarfile
import tempfile
import time
import typing as tp
import zipfile

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
    futures: tp.Dict[concurrent.futures.Future, str],
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

    def cleanup():
      subprocess.check_call(['umount', shm_dir])
      subprocess.check_call(['hdiutil', 'detach', disk_name])
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

class LocalFile(abc.ABC):
  """Identifies a file on the local system."""

  @abc.abstractmethod
  def read(self) -> bytes:
    """Read the file bytes."""

  @abc.abstractmethod
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

_GZ_SUFFIX = '.gz'

class GZipFile(LocalFile):
  """A gzipped file."""

  def __init__(self, root: str, path: str):
    self.root = root
    self.path = path
    if not path.endswith(_GZ_SUFFIX):
      raise ValueError(f'{root}/{path} is not a gz file?')

  @property
  def name(self) -> str:
    return self.path.removesuffix(_GZ_SUFFIX)

  def read(self) -> bytes:
    with gzip.open(os.path.join(self.root, self.path)) as f:
      return f.read()

  @contextmanager
  def extract(self, tmpdir: str) -> Generator[str, None, None]:
    try:
      path = os.path.join(tmpdir, self.name)
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
        stdout=subprocess.PIPE)
    return result.stdout

  @contextmanager
  def extract(self, tmpdir: str) -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir:
      subprocess.check_call(
        ['7z', 'x', '-o' + tmpdir, self.root, self.path],
        stdout=subprocess.DEVNULL)
      # with py7zr.SevenZipFile(self.root) as archive:
      #   archive.extract(path=tmpdir, targets=[self.path])
      yield os.path.join(tmpdir, self.path)

class ZipFile(LocalFile):
  """File inside a zip archive."""

  def __init__(self, root: str, path: str):
    self.root = root
    self.path = path
    self.is_gzipped = path.endswith(_GZ_SUFFIX)

  @property
  def name(self) -> str:
    return self.path.removesuffix(_GZ_SUFFIX)

  def read(self) -> bytes:
    result = subprocess.run(
        ['unzip', '-p', self.root, self.path],
        check=True, stdout=subprocess.PIPE)
    data = result.stdout
    if self.is_gzipped:
      data = gzip.decompress(data)
    return data

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

  def __enter__(self):
    self.tmpdir = tempfile.TemporaryDirectory(dir=get_tmp_dir(in_memory=True))
    self.extract()
    return self

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
VALID_SUFFIXES = [_SLP_SUFFIX, _SLP_SUFFIX + _GZ_SUFFIX]

def traverse_slp_files_zip(root: str) -> list[LocalFile]:
  files = []
  relpaths = zipfile.PyZipFile(root).namelist()
  for path in relpaths:
    if any(path.endswith(s) for s in VALID_SUFFIXES):
      files.append(ZipFile(root, path))
  return files
