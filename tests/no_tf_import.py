"""Tests that tensorflow is not imported when importing neutral or jax code."""

import sys
from slippi_ai import envs, saving, eval_lib, data, utils, evaluators
from slippi_ai.jax import *

if __name__ == '__main__':
  assert 'tensorflow' not in sys.modules
