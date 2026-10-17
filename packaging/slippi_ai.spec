# PyInstaller spec for a one-dir build of the GUI and the play CLI, on Windows
# or macOS.
#
#   pip install .[gui,winml] pyinstaller
#   pyinstaller packaging/slippi_ai.spec
#
# With the winml extra (the release build), models run on the CPU anywhere
# and on Windows ML's providers, such as TensorRT-RTX, where available; see
# slippi_ai/winml.py. Building with .[gui,onnx] instead gives a CPU-only
# bundle.
#
# Produces dist/phillip/phillip.exe (the GUI), eval_two.exe and
# benchmark_eval_two.exe, sharing one dist/phillip/_internal.
#
# On macOS, build with .[gui,onnx]; this also produces dist/phillip.app, with
# the GUI as the app and eval_two and benchmark_eval_two in
# phillip.app/Contents/MacOS.

import configparser
import json
import os
import subprocess
import sys

from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules)

ROOT = SPECPATH + '/..'


def build_info_file() -> str:
  """What the bundle was built from, logged by the GUI for bug reports."""
  config = configparser.ConfigParser()
  config.read(f'{ROOT}/setup.cfg')
  try:
    commit = subprocess.run(
        ['git', 'describe', '--always', '--dirty', '--exclude=*'],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    commit = 'unknown'
  path = os.path.join(workpath, 'build_info.json')
  with open(path, 'w') as f:
    json.dump(dict(slippi_ai=config['metadata']['version'], commit=commit), f)
  return path



def app_version() -> str:
  """slippi_ai/gui/version.py's VERSION, without importing slippi_ai."""
  namespace = {}
  with open(f'{ROOT}/slippi_ai/gui/version.py') as f:
    exec(f.read(), namespace)
  return namespace['VERSION']


datas = (
    # Frame data CSVs and the Gecko codes ini, read relative to the package.
    collect_data_files('melee')
    # Read next to slippi_ai/gui/version.py.
    + [(build_info_file(), 'slippi_ai/gui')]
)
binaries = collect_dynamic_libs('onnxruntime')
hiddenimports = []
try:
  import winui3  # pylint: disable=unused-import
except ImportError:
  pass
else:
  # Imported inside slippi_ai.winml.initialize, and the Windows App SDK
  # bootstrap DLL next to them.
  hiddenimports += collect_submodules('winui3')
  binaries += collect_dynamic_libs('winui3')

# Imported lazily on the play path, and too large to bundle by accident.
excludes = ['jax', 'jaxlib', 'tensorflow', 'tensorflow_probability', 'wandb',
            'pandas', 'IPython', 'matplotlib', 'tkinter']

# (name, script, console, hidden imports)
programs = [
    # The launcher runs slippi_ai.gui by name, which PyInstaller can't follow.
    ('phillip', 'packaging/gui.py', False, ['slippi_ai.gui.__main__']),
    ('eval_two', 'scripts/eval_two.py', True, []),
    ('benchmark_eval_two', 'scripts/benchmark_eval_two.py', True, []),
]

analyses = []
exes = []
for name, script, console, program_imports in programs:
  a = Analysis(
      [f'{ROOT}/{script}'],
      pathex=[ROOT],
      hiddenimports=hiddenimports + program_imports,
      binaries=binaries,
      datas=datas,
      excludes=excludes,
  )
  analyses.append(a)
  exes.append(EXE(
      PYZ(a.pure), a.scripts, [],
      exclude_binaries=True,
      name=name,
      console=console,
  ))

coll = COLLECT(
    *exes,
    *(a.binaries for a in analyses),
    *(a.datas for a in analyses),
    name='phillip',
)

if sys.platform == 'darwin':
  app = BUNDLE(
      coll,
      name='phillip.app',
      bundle_identifier='com.github.vladfi1.phillip',
      version=app_version(),
      info_plist={
          # BUNDLE would pick the last executable; the app runs the GUI.
          'CFBundleExecutable': 'phillip',
          'NSHighResolutionCapable': True,
          # Dark mode follows the system, as Qt supports it.
          'NSRequiresAquaSystemAppearance': False,
      },
  )
