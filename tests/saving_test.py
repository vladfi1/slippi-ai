import unittest

from slippi_ai import saving, paths

class SavingTest(unittest.TestCase):

  def test_load_demo_checkpoint(self):
    saving.load_policy_from_disk(paths.DEMO_CHECKPOINT)
