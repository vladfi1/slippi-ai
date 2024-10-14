import time

from absl import app, flags
import numpy as np
import pulp

from slippi_ai.nash import nash

RUNTIME = flags.DEFINE_float('runtime', 10, 'Running time, in seconds.')
SIZE = flags.DEFINE_integer('size', 10, 'Size of the problem.')

solvers = {
    'pulp': nash.solve_zero_sum_nash_pulp,
    'gambit': nash.solve_zero_sum_nash_gambit,
}

SOLVER = flags.DEFINE_enum('solver', 'pulp', solvers.keys(), 'Solver to use.')

def test_solve_zero_sum_nash(n: int, solver: str):
  payoff_matrix = np.random.randn(n, n).astype(np.float64)
  return solvers[solver](payoff_matrix)

def main(_):
  start_time = time.perf_counter()

  num_solved = 0
  while time.perf_counter() - start_time < RUNTIME.value:
    test_solve_zero_sum_nash(SIZE.value, SOLVER.value)
    num_solved += 1

  run_time = time.perf_counter() - start_time
  problems_per_second = num_solved / run_time
  print(
      f'Solved {num_solved} problems in {run_time:.2f} seconds, '
      f'{problems_per_second:.2f} problems per second')

if __name__ == '__main__':
  app.run(main)
