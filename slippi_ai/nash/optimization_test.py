import numpy as np
import tensorflow as tf

from slippi_ai.nash import optimization, nash


class QuadraticOptimizationProblem(optimization.ConstrainedOptimizationProblem[tf.Tensor]):

  def __init__(self, num_dims: int, initial_x: float):
    self.num_dims = num_dims
    self.initial_x = initial_x

  def initial_variables(self) -> tf.Tensor:
    x = tf.constant(self.initial_x, dtype=tf.float32)
    return tf.fill([self.num_dims], x)

  def objective(self, variables: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.square(variables))

  def constraint_violations(self, variables: tf.Tensor) -> tf.Tensor:
    del variables
    return tf.constant([], dtype=tf.float32)

  def equality_violations(self, variables: tf.Tensor) -> tf.Tensor:
    del variables
    return tf.constant([], dtype=tf.float32)

CornerVariables = tf.Tensor

class CornerOptimizationProblem(optimization.ConstrainedOptimizationProblem[CornerVariables]):

  def __init__(self, num_dims: int, size: float, dtype=tf.float32):
    self.num_dims = num_dims
    self.size = size
    self.dtype = dtype

  def initial_variables(self) -> CornerVariables:
    return tf.zeros([self.num_dims], dtype=self.dtype)

  def objective(self, variables: CornerVariables) -> tf.Tensor:
    return -tf.reduce_sum(variables)

  def constraint_violations(self, variables: CornerVariables) -> tf.Tensor:
    return variables - self.size

  def equality_violations(self, variables: CornerVariables) -> tf.Tensor:
    return tf.constant([], dtype=self.dtype)

def test_solve_quadratic_optimization(num_dims=3):
  problem = QuadraticOptimizationProblem(num_dims, 1.0)
  variables = optimization.solve_optimization_interior_point_barrier(
      problem, error=1e-3)
  assert tf.reduce_all(tf.abs(variables) < 1e-3).numpy()

def test_solve_corner_optimization(
    n: int = 1,
    size: float = 1,
    solver: optimization.Solver[CornerVariables] = optimization.solve_optimization_interior_point_barrier,
    **kwargs,
):
  problem = CornerOptimizationProblem(n, size=size)
  variables = solver(problem, **kwargs)

  actual = variables.numpy()
  expected = np.ones([n]) * problem.size

  np.testing.assert_allclose(actual, expected, atol=1e-3)

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
  nonzero = p > 1e-6
  safe_p = np.where(nonzero, p, 1)
  safe_q = np.where(nonzero, q, 1)
  log_ratio = np.log(safe_p / safe_q)
  return np.sum(p * log_ratio, axis=-1)

@tf.function
def solve_nash(payoff_matrix: np.ndarray, **kwargs) -> nash.NashVariables:
  problem = nash.ZeroSumNashProblem(payoff_matrix)
  variables = optimization.solve_feasibility(problem, **kwargs)
  return variables.normalize()

def test_nash(payoff_matrix: np.ndarray, atol=1e-1, **kwargs):
  # problem = nash.ZeroSumNashProblemWithLogits(payoff_matrix)
  variables = solve_nash(payoff_matrix, **kwargs)
  # problem = nash.ZeroSumNashProblem(payoff_matrix)
  # variables = optimization.solve_feasibility(problem, **kwargs)
  # variables = variables.normalize()

  # tf_p1 = tf.nn.softmax(variables.p1_logits).numpy()
  # tf_p2 = tf.nn.softmax(variables.p2_logits).numpy()
  tf_p1 = variables.p1.numpy()
  tf_p2 = variables.p2.numpy()
  tf_nash_value = variables.p1_nash_value.numpy()

  p1, p2, nash_value = nash.solve_zero_sum_nash_pulp(payoff_matrix)
  np.testing.assert_allclose(p1 @ payoff_matrix @ p2, nash_value, atol=1e-4)

  kl1 = kl_divergence(p1, tf_p1)
  assert kl1 < atol, kl1

  kl2 = kl_divergence(p2, tf_p2)
  assert kl2 < atol, kl2

  np.testing.assert_allclose(tf_nash_value, nash_value, atol=atol)

  return p1, p2, nash_value

def test_rps(**kwargs):
  payoff_matrix = np.array([
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0],
  ], dtype=np.float32)
  p1, p2, nash_value = test_nash(payoff_matrix, **kwargs)

  atol = 1e-4
  np.testing.assert_allclose(p1, np.ones([3]) / 3, atol=atol)
  np.testing.assert_allclose(p2, np.ones([3]) / 3, atol=atol)
  np.testing.assert_allclose(nash_value, 0, atol=atol)

def test_random_nash(
    solver=optimization.solve_optimization_interior_point_barrier,
    size: tuple[int, int] = (3, 3),
    dtype: np.dtype = np.float32,
    n: int = 10,
    **kwargs,
):
  for _ in range(n):
    payoff_matrix = np.random.randn(*size).astype(dtype)
    # test_nash(
    #     payoff_matrix,
    #     num_iterations=200,
    #     initial_constraint_weight=1e-1,
    #     constraint_weight_decay=0.95,
    #     damping=2,
    # )
    test_nash(
        payoff_matrix,
        optimization_solver=solver,
        **kwargs,
    )

if __name__ == '__main__':
  # test_solve_quadratic_optimization()
  # test_solve_corner_optimization(
  #     solver=optimization.solve_optimization_interior_point_barrier,
  #     error=1e-3,
  # )
  # test_rps(
  #     optimization_solver=optimization.solve_optimization_interior_point_barrier,
  #     error=1e-3,
  # )

  test_random_nash(
      solver=optimization.solve_optimization_interior_point_barrier,
      size=(100, 100),
      dtype=np.float64,
      error=1e-5,
      atol=1e-2,
  )
