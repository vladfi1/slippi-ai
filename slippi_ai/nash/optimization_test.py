import time

import numpy as np
import tensorflow as tf
import tqdm

from slippi_ai import utils
from slippi_ai.nash import optimization, nash


class QuadraticOptimizationProblem(optimization.ConstrainedOptimizationProblem[tf.Tensor]):

  def __init__(self, num_dims: int, initial_x: np.ndarray):
    self.num_dims = num_dims
    self.initial_x = initial_x
    assert len(initial_x.shape) == 1

  def initial_variables(self) -> tf.Tensor:
    return tf.stack([self.initial_x] * self.num_dims, axis=-1)

  def batch_size(self) -> int:
    return self.initial_x.shape[0]

  def objective(self, variables: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.square(variables), axis=-1)

  def constraint_violations(self, variables: tf.Tensor) -> tf.Tensor:
    zero = tf.constant(0, dtype=variables.dtype)
    return tf.fill([self.batch_size(), 0], zero)

  def equality_violations(self, variables: tf.Tensor) -> tf.Tensor:
    zero = tf.constant(0, dtype=variables.dtype)
    return tf.fill([self.batch_size(), 0], zero)

def test_solve_quadratic_optimization(num_dims=3, batch_size=1):
  xs = np.arange(batch_size, dtype=np.float32)
  problem = QuadraticOptimizationProblem(num_dims, xs)
  variables = optimization.solve_optimization_interior_point_barrier(
      problem, error=1e-3)
  assert tf.reduce_all(tf.abs(variables) < 1e-3).numpy()

CornerVariables = tf.Tensor

class CornerOptimizationProblem(optimization.ConstrainedOptimizationProblem[CornerVariables]):

  def __init__(self, num_dims: int, sizes: np.ndarray, dtype=tf.float32):
    self.num_dims = num_dims
    self.sizes = tf.convert_to_tensor(sizes, dtype=dtype)
    self.dtype = dtype

  def batch_size(self) -> int:
    return self.sizes.shape[0]

  def initial_variables(self) -> CornerVariables:
    return tf.zeros([self.batch_size(), self.num_dims], dtype=self.dtype)

  def objective(self, variables: CornerVariables) -> tf.Tensor:
    return -tf.reduce_sum(variables, axis=-1)

  def constraint_violations(self, variables: CornerVariables) -> tf.Tensor:
    return variables - tf.expand_dims(self.sizes, -1)

  def equality_violations(self, variables: CornerVariables) -> tf.Tensor:
    zero = tf.constant(0, dtype=variables.dtype)
    return tf.fill([self.batch_size(), 0], zero)

def test_solve_corner_optimization(
    num_dims: int = 1,
    max_size: int = 1,
    solver: optimization.Solver[CornerVariables] = optimization.solve_optimization_interior_point_barrier,
    **kwargs,
):
  sizes = 1 + np.arange(max_size)
  problem = CornerOptimizationProblem(num_dims, sizes=sizes)
  variables = solver(problem, **kwargs)

  actual = variables.numpy()
  expected = np.stack([sizes] * num_dims, axis=-1)

  np.testing.assert_allclose(actual, expected, atol=1e-3)

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
  nonzero = p > 1e-6
  safe_p = np.where(nonzero, p, 1)
  safe_q = np.where(nonzero, q, 1)
  log_ratio = np.log(safe_p / safe_q)
  return np.sum(p * log_ratio, axis=-1)

@tf.function
def solve_nash(payoff_matrices: np.ndarray, **kwargs):
  print('retracing with shape', payoff_matrices.shape)
  problem = nash.ZeroSumNashProblem(payoff_matrices)
  variables, stats = optimization.solve_feasibility(problem, optimum=0, **kwargs)
  return variables.normalize(), stats

def test_nash(
    payoff_matrices: np.ndarray,
    atol=1e-1,
    verify=True,
    **kwargs,
) -> dict:
  start_time = time.perf_counter()
  variables, stats = solve_nash(payoff_matrices, **kwargs)
  solve_time = time.perf_counter() - start_time

  if verify:
    tf_p1 = variables.p1.numpy()
    tf_p2 = variables.p2.numpy()
    tf_nash_value = variables.p1_nash_value.numpy()

    for i, payoff_matrix in enumerate(payoff_matrices):
      p1, p2, nash_value = nash.solve_zero_sum_nash_pulp(payoff_matrix)
      np.testing.assert_allclose(p1 @ payoff_matrix @ p2, nash_value, atol=1e-4)

      kl1 = kl_divergence(p1, tf_p1[i])
      assert kl1 < atol, kl1

      kl2 = kl_divergence(p2, tf_p2[i])
      assert kl2 < atol, kl2

      np.testing.assert_allclose(tf_nash_value[i], nash_value, atol=atol)

  stats = {k: v.numpy() for k, v in stats.items()}
  stats['time'] = solve_time
  return stats

def test_rps(**kwargs):
  payoff_matrix = np.array([[
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0],
  ]], dtype=np.float32)
  test_nash(payoff_matrix, **kwargs)


def test_random_nash(
    solver=optimization.solve_optimization_interior_point_barrier,
    size: tuple[int, int] = (3, 3),
    dtype: np.dtype = np.float32,
    batch_size: int = 1,
    **kwargs,
):
  payoff_matrix = np.random.randn(batch_size, *size).astype(dtype)
  # test_nash(
  #     payoff_matrix,
  #     num_iterations=200,
  #     initial_constraint_weight=1e-1,
  #     constraint_weight_decay=0.95,
  #     damping=2,
  # )
  return test_nash(
      payoff_matrix,
      optimization_solver=solver,
      **kwargs,
  )

def random_nash_tests(
    num_tests: int = 10,
    batch_size: int = 10,
    **kwargs,
):
  all_stats = []
  solve_times = []
  for i in tqdm.trange(num_tests):
    stats = test_random_nash(
        solver=optimization.solve_optimization_interior_point_barrier,
        batch_size=batch_size,
        **kwargs,
    )
    all_stats.append(stats)
    if i > 0:  # first iteration is warmup
      solve_times.append(stats['time'])

  if solve_times:
    total_solved = batch_size * len(solve_times)
    total_time = sum(solve_times)
    mean_time = total_time / total_solved
    problems_per_second = total_solved / total_time
    print(f'Mean solve time: {mean_time} s, {problems_per_second} problems/s')

  stats = utils.batch_nest(all_stats)

  for key in ['num_steps', 'centering_steps', 'slack']:
    values = stats[key]
    mean, std = np.mean(values), np.std(values)
    min_value = np.min(values)
    max_value = np.max(values)
    print(f'{key}: {mean:.1e} ± {std:.1e}, [{min_value:.1e}, {max_value:.1e}]')

if __name__ == '__main__':
  test_solve_quadratic_optimization(batch_size=3)
  test_solve_corner_optimization(
      solver=optimization.solve_optimization_interior_point_barrier,
      error=1e-3,
      max_size=3,
      num_dims=2,
  )
  test_rps(
      optimization_solver=optimization.solve_optimization_interior_point_barrier,
      error=1e-3,
  )

  random_nash_tests(
      num_tests=10,
      batch_size=10,
      size=(10, 11),
      dtype=np.float64,
      error=1e-4,
      atol=1e-1,
  )
