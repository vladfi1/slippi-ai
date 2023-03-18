import pathlib
import subprocess
import typing as tp

_SELF_PATH = pathlib.Path(__file__)
_SLIPPI_DB = _SELF_PATH.parent
EXE_PATH = _SLIPPI_DB / 'fix-onedrive-zip.pl'

def fix_zip(path: tp.Union[str, tp.Iterable[str]]):
  paths = [path] if isinstance(path, str) else path
  subprocess.check_call(
      ['perl', EXE_PATH, *paths],
      stdout=subprocess.DEVNULL)
