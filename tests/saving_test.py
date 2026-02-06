import unittest

from slippi_ai import paths
from slippi_ai.tf import saving

class SavingTest(unittest.TestCase):

  def test_load_demo_checkpoint(self):
    saving.load_policy_from_disk(str(paths.DEMO_CHECKPOINT))
