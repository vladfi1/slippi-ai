import unittest

import numpy as np

from slippi_ai.nash import solve_zero_sum_nash_pulp


class NashTest(unittest.TestCase):

  def test_solve_zero_sum_nash_pulp_rps(self):
    payoff_matrix = np.array([
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ])

    p1, p2, nash_value = solve_zero_sum_nash_pulp(payoff_matrix)

    expected = np.full([3], 1 / 3)
    np.testing.assert_allclose(p1, expected, atol=1e-6)
    np.testing.assert_allclose(p2, expected, atol=1e-6)
    np.testing.assert_allclose(nash_value, 0.0, atol=1e-6)


if __name__ == '__main__':
  unittest.main()
