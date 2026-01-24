import unittest

import numpy as np

from slippi_ai.jax import rl_lib


class RlLibTest(unittest.TestCase):

  def test_discounted_returns(self):
    rewards = np.array([1, 2, 3], np.float32)
    discounts = np.array([1, 0.5, 0.25], np.float32)
    bootstrap = np.array(4, np.float32)

    returns = rl_lib.discounted_returns(rewards, discounts, bootstrap)
    expected = np.array([5, 4, 4], np.float32)

    np.testing.assert_allclose(returns, expected)

  def test_discounted_returns_batched(self):
    # Test with batch dimension [T, B]
    rewards = np.array([[1, 2], [2, 3], [3, 4]], np.float32)
    discounts = np.array([[1, 1], [0.5, 0.5], [0.25, 0.25]], np.float32)
    bootstrap = np.array([4, 5], np.float32)

    returns = rl_lib.discounted_returns(rewards, discounts, bootstrap)
    # returns[2] = [3, 4] + [0.25, 0.25] * [4, 5] = [4, 5.25]
    # returns[1] = [2, 3] + [0.5, 0.5] * [4, 5.25] = [4, 5.625]
    # returns[0] = [1, 2] + [1, 1] * [4, 5.625] = [5, 7.625]
    expected = np.array([[5, 7.625], [4, 5.625], [4, 5.25]], np.float32)

    np.testing.assert_allclose(returns, expected)


if __name__ == '__main__':
  unittest.main(failfast=True)
