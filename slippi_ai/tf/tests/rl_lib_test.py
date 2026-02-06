import unittest

import numpy as np

from slippi_ai.tf import rl_lib

class RlLibTest(unittest.TestCase):

  def test_discounted_returns(self):
    rewards = np.array([1, 2, 3], np.float32)
    discounts = np.array([1, 0.5, 0.25], np.float32)
    bootstrap = np.float32(4)

    returns = rl_lib.discounted_returns(rewards, discounts, bootstrap)
    expected = np.array([5, 4, 4], np.float32)

    np.testing.assert_allclose(returns.numpy(), expected)

if __name__ == '__main__':
  unittest.main(failfast=True)
