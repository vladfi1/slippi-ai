"""Tests that jax is not imported when importing neutral or tensorflow code."""

import sys
from slippi_ai import envs, saving, eval_lib, data, utils, evaluators
from slippi_ai.tf import *

if __name__ == '__main__':
  assert 'jax' not in sys.modules
