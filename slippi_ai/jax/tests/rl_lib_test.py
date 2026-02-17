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


  def test_generalized_returns_lambda_one(self):
    # When lambda=1, generalized_returns should equal discounted_returns.
    rewards = np.array([1, 2, 3], np.float32)
    discounts = np.array([1, 0.5, 0.25], np.float32)
    bootstrap = np.array(4, np.float32)
    values = np.array([10, 20, 30], np.float32)
    lambdas = np.ones(3, np.float32)

    returns = rl_lib.generalized_returns(
        rewards, discounts, values, bootstrap, lambdas)
    expected = rl_lib.discounted_returns(rewards, discounts, bootstrap)

    np.testing.assert_allclose(returns, expected)

  def test_generalized_returns_lambda_zero(self):
    # When lambda=0, generalized_returns should return the values unchanged.
    rewards = np.array([1, 2, 3], np.float32)
    discounts = np.array([1, 0.5, 0.25], np.float32)
    bootstrap = np.array(4, np.float32)
    values = np.array([10, 20, 30], np.float32)
    lambdas = np.zeros(3, np.float32)

    returns = rl_lib.generalized_returns(
        rewards, discounts, values, bootstrap, lambdas)

    np.testing.assert_allclose(returns, values)

  def test_generalized_returns_mixed_lambdas(self):
    rewards = np.array([3, 2, 2], np.float32)
    discounts = np.array([1, 1, 1], np.float32)
    values = np.array([4, 4, 6], np.float32)
    bootstrap = np.array(4, np.float32)
    lambdas = np.array([0.75, 0.25, 0.5], np.float32)

    returns = rl_lib.generalized_returns(
        rewards, discounts, values, bootstrap, lambdas)

    # Manual computation (reverse scan):
    # t=2: value=2+1*4=6; smoothed=0.5*6+0.5*6=6
    # t=1: value=2+1*6=8; smoothed=0.25*8+0.75*4=5
    # t=0: value=3+1*5=8; smoothed=0.75*8+0.25*4=7
    expected = np.array([7, 5, 6], np.float32)

    np.testing.assert_allclose(returns, expected)


if __name__ == '__main__':
  unittest.main(failfast=True)
