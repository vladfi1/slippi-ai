# PyInstaller spec for a one-dir Windows build of the play CLI.
#
#   pip install .[onnx] pyinstaller
#   pyinstaller packaging/slippi_ai.spec
#
# Produces dist/slippi-ai/eval_two.exe and benchmark_eval_two.exe, sharing one
# dist/slippi-ai/_internal.

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = SPECPATH + '/..'

datas = (
    # Frame data CSVs and the Gecko codes ini, read relative to the package.
    collect_data_files('melee')
)
binaries = collect_dynamic_libs('onnxruntime')

# Imported lazily on the play path, and too large to bundle by accident.
excludes = ['jax', 'jaxlib', 'tensorflow', 'tensorflow_probability', 'wandb',
            'pandas', 'IPython', 'matplotlib', 'tkinter']


def analysis(script):
  return Analysis(
      [f'{ROOT}/scripts/{script}.py'],
      pathex=[ROOT],
      binaries=binaries,
      datas=datas,
      excludes=excludes,
  )


scripts = ['eval_two', 'benchmark_eval_two']
analyses = [analysis(s) for s in scripts]
exes = []
for script, a in zip(scripts, analyses):
  pyz = PYZ(a.pure)
  exes.append(EXE(
      pyz, a.scripts, [],
      exclude_binaries=True,
      name=script,
      console=True,
  ))

coll = COLLECT(
    *exes,
    *(a.binaries for a in analyses),
    *(a.datas for a in analyses),
    name='slippi-ai',
)
