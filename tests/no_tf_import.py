"""Tests that tensorflow is not imported in the main module."""

import sys
from slippi_ai import envs  # pylint: disable=unused-import

if __name__ == '__main__':
  assert 'tensorflow' not in sys.modules
