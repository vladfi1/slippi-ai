import os
import pathlib
import shutil
import subprocess
import typing as tp

DATASET_TEMPLATE = "https://example.com/{version}/datasets/pq/{file}"

DATASET_VERSIONS = (
    'test', 'prod',
)

GAMES_TAR = 'games.tar'
GAMES_DIR = 'games'
META_PQ = 'meta.pq'

def download(url: str, path: tp.Union[str, pathlib.Path]):
  if isinstance(path, pathlib.Path):
    path = str(path)

  subprocess.check_call(
      ['curl', url, '-o', path])

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
      return False

    download(url, path)
    return path

  def pull_dataset(self, version: str):
    self.pull_games(DATASET_TEMPLATE.format(version=version, file=GAMES_TAR))
    self.pull_file(
        DATASET_TEMPLATE.format(version=version, file=META_PQ),
        META_PQ)
