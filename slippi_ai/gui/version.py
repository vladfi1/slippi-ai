"""The phillip app's version, for the GUI, the bundle and the installer.

Independent of the slippi-ai package version in setup.cfg. Bump it for each
app release, which is tagged launcher-v<VERSION> (see slippi_ai/gui/updates.py
and .github/workflows/bundle.yml).
"""

import json
import pathlib

VERSION = '0.3.0'


def build_info() -> str:
  """The app version, and what the bundle was built from if it's frozen."""
  text = f'phillip {VERSION}'
  path = pathlib.Path(__file__).with_name('build_info.json')
  try:
    info = json.loads(path.read_text())
  except (OSError, ValueError):
    return text
  return f'{text} (slippi-ai {info["slippi_ai"]}, commit {info["commit"]})'
