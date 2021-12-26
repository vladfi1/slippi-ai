import unittest

import numpy as np

from slippi_ai import data

class CompressRepeatedActionsTest(unittest.TestCase):

  def test_indices_and_counts(self):
    actions = np.random.randint(3, size=100)
    repeats = data.detect_repeated_actions(actions)
    indices, counts = data.indices_and_counts(repeats)

    reconstruction = []
    for i, c in zip(indices, counts):
      reconstruction.extend([actions[i]] * (c + 1))

    self.assertSequenceEqual(reconstruction, actions.tolist())

if __name__ == '__main__':
  unittest.main(failfast=True)
