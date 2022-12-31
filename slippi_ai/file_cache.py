import os
import pathlib
import shutil
import subprocess
import typing as tp

from slippi_ai import s3_lib

# in the slp-replays bucket
DATASET_TEMPLATE = "https://slp-replays.s3.amazonaws.com/{version}/datasets/pq/{file}"

DATASET_VERSIONS = (
    'test', 'prod',
)

GAMES_TAR = 'games.tar'
GAMES_DIR = 'games'
META_PQ = 'meta.pq'

# in the slippi-data bucket
ISO = 'SSBM.iso'
DOLPHIN = 'dolphin'


def download(url: str, path: tp.Union[str, pathlib.Path]):
  if isinstance(path, pathlib.Path):
    path = str(path)

  subprocess.check_call(
      ['curl', url, '-o', path])

  print(f'Downloaded {url} to {path}.')

def download_s3(s3_key: str, path: tp.Union[str, pathlib.Path]):
  if isinstance(path, pathlib.Path):
    path = str(path)

  store = s3_lib.get_store()
  store.get_file(s3_key, path)

  print(f'Downloaded s3://{s3_lib.BUCKET_NAME}/{s3_key} to {path}.')

class FileCache:
  """Caches files locally."""

  def __init__(
    self,
    root: str,  # automatic tmp dir?
    wipe: bool = False,
  ) -> None:
    self._root = pathlib.Path(root)

    if wipe and self._root.exists():
      shutil.rmtree(self._root)

    if self._root.exists():
      if not self._root.is_dir():
        raise FileExistsError(f'{self._root} is exists but is not a directory')
    else:
      os.makedirs(self._root)

    self.games_dir = self._root / GAMES_DIR
    self.meta_path = self._root / META_PQ

  def pull_games(self, url: str) -> bool:
    games_dir = self.games_dir
    if games_dir.exists():
      print(f'Games dir "{games_dir}" already exists.')
      return False

    os.makedirs(games_dir)

    games_tar = self._root / GAMES_TAR
    download(url, games_tar)

    subprocess.check_call(
        ['tar', 'xf', games_tar, '-C', games_dir])
    games_tar.unlink()

  def pull_file(self, url: str, path: str) -> pathlib.Path:
    """Pulls a remote file to a local (root-relative) path."""
    path = self._root / path

    if path.exists():
      print(f'"{path}" already exists.')
    else:
      download(url, path)
  
    return path

  def pull_dataset(self, version: str):
    self.pull_games(DATASET_TEMPLATE.format(version=version, file=GAMES_TAR))
    self.pull_file(
        DATASET_TEMPLATE.format(version=version, file=META_PQ),
        META_PQ)

  def pull_s3(self, s3_key: str, local_path: str):
    path = self._root / local_path

    if path.exists():
      print(f'"{path}" already exists.')
    else:
      download_s3(s3_key, path)

    return path

  def pull_iso(self):
    return self.pull_s3(ISO, ISO)

  def pull_dolphin(self):
    path = self.pull_s3(DOLPHIN, DOLPHIN)
    subprocess.check_call(['chmod', '+x', path])
    return path
