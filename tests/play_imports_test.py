"""Tests that the code needed to play against an agent has a minimal footprint.

Run this from an install without any extras (`pip install .`) to check that
the play path doesn't depend on training-only packages.
"""

import sys
from slippi_ai import eval_lib, dolphin, saving

TRAINING_ONLY_MODULES = [
    'jax',
    'tensorflow',
    'wandb',
    'pandas',
    'peppi_py',
    'py7zr',
    'fsspec',
]

if __name__ == '__main__':
  loaded = [m for m in TRAINING_ONLY_MODULES if m in sys.modules]
  assert not loaded, f'Play path imported training-only modules: {loaded}'
