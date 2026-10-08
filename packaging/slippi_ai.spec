# PyInstaller spec for a one-dir Windows build of the GUI and the play CLI.
#
#   pip install .[gui,winml] pyinstaller
#   pyinstaller packaging/slippi_ai.spec
#
# With the winml extra (the release build), models run on the CPU anywhere
# and on Windows ML's providers, such as TensorRT-RTX, where available; see
# slippi_ai/winml.py. Building with .[gui,onnx] instead gives a CPU-only
# bundle.
#
# Produces dist/slippi-ai/slippi-ai.exe (the GUI), eval_two.exe and
# benchmark_eval_two.exe, sharing one dist/slippi-ai/_internal.

from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules)

ROOT = SPECPATH + '/..'

datas = (
    # Frame data CSVs and the Gecko codes ini, read relative to the package.
    collect_data_files('melee')
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
    ('slippi-ai', 'packaging/gui.py', False, ['slippi_ai.gui.__main__']),
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
    name='slippi-ai',
)
