"""Launches the GUI in the frozen build.

slippi_ai/gui/__main__.py can't be the script: PyInstaller would put its
folder on the import path, where slippi_ai/types.py hides the stdlib types.
"""

import runpy

if __name__ == '__main__':
  runpy.run_module('slippi_ai.gui', run_name='__main__', alter_sys=True)
