"""Runs the GUI: python -m slippi_ai.gui"""

import multiprocessing
import os
import sys

if __name__ == '__main__':
  # Windowed builds have no console, so writes to stdout/stderr would fail.
  if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
  if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

  # Sessions run in child processes; needed in frozen builds.
  multiprocessing.freeze_support()
  from slippi_ai.gui import app
  sys.exit(app.main())
