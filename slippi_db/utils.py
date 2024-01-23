import abc
import atexit
import concurrent.futures
from contextlib import contextmanager
import functools
import hashlib
import os
from typing import Generator
import py7zr
import subprocess
import sys
import tempfile
import time
import typing as tp

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
_MACOS_SHM_SIZE = 256 # MB

@functools.cache
def tmp_dir(in_memory: bool) -> tp.Optional[str]:
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
  def extract(self) -> tp.Generator[str, None, None]:
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
  def extract(self) -> Generator[str, None, None]:
    yield os.path.join(self.root, self.path)

  def read(self):
    with open(os.path.join(self.root, self.path), 'rb') as f:
      return f.read()

class SevenZipFile(LocalFile):
  """File inside a 7z archive."""

  def __init__(self, root: str, path: str):
    self.root = root
    self.path = path

  @property
  def name(self) -> str:
    return self.path

  def read(self):
    result = subprocess.run(
        ['7z', 'e', '-so', self.root, self.path],
        stdout=subprocess.PIPE)
    return result.stdout

  @contextmanager
  def extract(self) -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory(dir=tmp_dir(in_memory=True)) as tmpdir:
      subprocess.check_call(
        ['7z', 'x', '-o' + tmpdir, self.root, self.path],
        stdout=subprocess.DEVNULL)
      # with py7zr.SevenZipFile(self.root) as archive:
      #   archive.extract(path=tmpdir, targets=[self.path])
      yield os.path.join(tmpdir, self.path)

def traverse_slp_files(root: str) -> list[SimplePath]:
  files = []
  for abspath, _, filenames in os.walk(root):
    for name in filenames:
      if name.endswith('.slp'):
        reldir = os.path.relpath(abspath, root)
        relpath = os.path.join(reldir, name)
        files.append(SimplePath(root, relpath))
  return files

def traverse_slp_files_7z(root: str) -> list[SevenZipFile]:
  files = []
  relpaths = py7zr.SevenZipFile(root).getnames()
  for path in relpaths:
    if path.endswith('.slp'):
      files.append(SevenZipFile(root, path))
  return files
