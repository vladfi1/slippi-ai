"""Settings remembered between runs, and checks on the Dolphin and ISO paths."""

import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import sys
import typing as tp

import melee

from slippi_ai import dolphin as dolphin_lib

# MD5 of the NTSC 1.02 ISO that Slippi expects.
MELEE_102_MD5 = '0e63d4223b01d9aba596259dc155a174'


@dataclasses.dataclass
class Settings:
  dolphin_path: str = ''
  iso_path: str = ''
  models_dir: str = ''
  model_path: str = ''  # The chosen model, in models_dir.
  character: str = ''  # phillip's melee.Character name.
  # Filters the models by who they were trained against; empty for any.
  opponent_character: str = ''
  opponent: str = 'human'  # 'human' or 'cpu'
  human_port: int = 1
  cpu_level: int = 9
  cpu_character: str = 'FOX'
  # The onnxruntime provider to run phillip on; empty for automatic.
  provider: str = ''
  # Copy Slippi Dolphin's user folder (DolphinConfig.copy_home_directory), so
  # Dolphin keeps the user's graphics, audio and controller settings.
  # Otherwise it uses defaults, and a human's port is a GameCube adapter.
  copy_dolphin_settings: bool = True


def settings_path() -> pathlib.Path:
  if sys.platform == 'win32':
    base = os.environ.get('APPDATA') or pathlib.Path.home() / 'AppData' / 'Roaming'
  else:
    base = os.environ.get('XDG_CONFIG_HOME') or pathlib.Path.home() / '.config'
  return pathlib.Path(base) / 'slippi-ai' / 'gui.json'


def load() -> Settings:
  path = settings_path()
  try:
    with open(path) as f:
      values = json.load(f)
  except FileNotFoundError:
    return Settings()
  except (OSError, ValueError) as e:
    logging.warning(f'Ignoring unreadable settings at {path}: {e}')
    return Settings()

  # Renamed when it started applying to more than controllers.
  if 'use_slippi_controller_settings' in values:
    values.setdefault(
        'copy_dolphin_settings', values.pop('use_slippi_controller_settings'))

  fields = {f.name for f in dataclasses.fields(Settings)}
  settings = Settings(**{k: v for k, v in values.items() if k in fields})
  # Older settings had a model file but no folder.
  if not settings.models_dir and settings.model_path:
    settings.models_dir = os.path.dirname(settings.model_path)
  return settings


def save(settings: Settings):
  path = settings_path()
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, 'w') as f:
    json.dump(dataclasses.asdict(settings), f, indent=2)


def detect_slippi() -> tuple[str, str]:
  """Returns Slippi Launcher's Dolphin folder and ISO path, or empty strings."""
  try:
    info = melee.console.default_dolphin_info()
  except Exception as e:  # Slippi Launcher isn't installed, or has moved.
    logging.info(f'Could not find Slippi Launcher: {e}')
    return '', ''
  return info.install_dir, info.iso_path or ''


def check_dolphin(path: str) -> tp.Optional[str]:
  """Returns an error message, or None if `path` is a usable Slippi Dolphin."""
  if not path:
    return 'Choose the Slippi Dolphin folder.'
  if not os.path.isdir(path):
    return 'Folder not found.'
  try:
    dolphin_lib.get_dolphin_version(path)
  except Exception as e:
    return f'Not a Slippi Dolphin folder: {e}'
  return None


def iso_md5(path: str, progress: tp.Optional[tp.Callable[[float], None]] = None) -> str:
  md5 = hashlib.md5()
  size = os.path.getsize(path)
  done = 0
  with open(path, 'rb') as f:
    while chunk := f.read(1 << 22):
      md5.update(chunk)
      done += len(chunk)
      if progress is not None:
        progress(done / size)
  return md5.hexdigest()
