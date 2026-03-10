import time
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from slippi_ai import utils
from slippi_ai.jax import jax_utils
from slippi_ai.jax.nash import optimization, nash

EmptyT = tuple[()]
Empty = ()

class QuadraticOptimizationProblem(optimization.ConstrainedOptimizationProblem[EmptyT, jax.Array]):
  """Quadratic bowl centered at the origin with no constraints."""

  def __init__(self, initial_x: np.ndarray):
    self.initial_x = np.asarray(initial_x)
    assert len(initial_x.shape) == 1

  def initial_variables(self, parameters: EmptyT) -> jax.Array:
    return self.initial_x

  def objective(self, parameters: EmptyT, variables: jax.Array) -> jax.Array:
    return jnp.sum(jnp.square(variables), axis=-1)

  def constraint_violations(self, parameters: EmptyT, variables: jax.Array) -> jax.Array:
    return jnp.zeros([0], dtype=variables.dtype)

  def equality_violations(self, parameters: EmptyT, variables: jax.Array) -> jax.Array:
    return jnp.zeros([0], dtype=variables.dtype)


def test_solve_quadratic_optimization(num_dims: int = 3):
  xs = np.arange(num_dims, dtype=np.float32)
  problem = QuadraticOptimizationProblem(xs)
  variables, _ = optimization.solve_optimization_interior_point_primal_dual(
      problem, Empty, error=1e-3, expected_dtype=jnp.float32)
  assert np.all(np.abs(np.asarray(variables)) < 1e-3)

CornerParams = jax.Array
CornerVariables = jax.Array

class CornerOptimizationProblem(optimization.ConstrainedOptimizationProblem[CornerParams, CornerVariables]):
  """Optimal solution is at the upper corner of the feasible region."""

  def initial_variables(self, parameters: CornerParams) -> CornerVariables:
    return jnp.zeros([parameters.shape[0]], dtype=parameters.dtype)

  def objective(self, parameters: CornerParams, variables: CornerVariables) -> jax.Array:
    return -jnp.sum(variables, axis=-1)

  def constraint_violations(self, parameters: CornerParams, variables: CornerVariables) -> jax.Array:
    return variables - parameters

  def equality_violations(self, parameters: CornerParams, variables: CornerVariables) -> jax.Array:
    return jnp.zeros([0], dtype=variables.dtype)

P = tp.ParamSpec('P')

def test_solve_corner_optimization(
    max_size: int = 1,
    solver: optimization.Solver[CornerParams, CornerVariables, P] = optimization.solve_optimization_interior_point_primal_dual,
    *solver_args: P.args,
    **solver_kwargs: P.kwargs,
):
  problem = CornerOptimizationProblem()
  sizes = 1 + jnp.arange(max_size).astype(jnp.float32)
  solver_kwargs.setdefault('expected_dtype', jnp.float32)
  variables, _ = solver(problem, sizes, *solver_args, **solver_kwargs)

  actual = np.asarray(variables)
  expected = np.asarray(sizes)

  atol = solver_kwargs.get('error', 1e-2)
  np.testing.assert_allclose(actual, expected, atol=atol)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
  nonzero = p > 1e-6
  safe_p = np.where(nonzero, p, 1)
  safe_q = np.where(nonzero, q, 1)
  log_ratio = np.log(safe_p / safe_q)
  return np.sum(p * log_ratio, axis=-1)


def verify_nash(
    payoff_matrix: np.ndarray,
    solution: nash.NashVariables,
    atol: float = 1e-1,
):
  jax_p1 = np.asarray(solution.p1)
  jax_p2 = np.asarray(solution.p2)
  jax_nash_value = np.asarray(solution.p1_nash_value)

  p1, p2, nash_value = nash.solve_zero_sum_nash_pulp(payoff_matrix)
  np.testing.assert_allclose(p1 @ payoff_matrix @ p2, nash_value, atol=1e-4)

  kl1 = kl_divergence(p1, jax_p1)
  assert kl1 < atol, kl1

  kl2 = kl_divergence(p2, jax_p2)
  assert kl2 < atol, kl2

  np.testing.assert_allclose(jax_nash_value, nash_value, atol=atol)

def run_nash_test(
    payoff_matrix: np.ndarray,
    solver: nash.NashSolver,
    atol: float = 1e-1,
    verify: bool = True,
    **kwargs,
) -> dict:
  start_time = time.perf_counter()
  variables, stats = solver(payoff_matrix, **kwargs)
  # Reading the stats will block until the solution is ready,
  # so we can measure the solve time here.
  stats = {k: np.asarray(v) for k, v in stats.items()}
  solve_time = time.perf_counter() - start_time

  if verify:
    batched = payoff_matrix.ndim == 3
    if not batched:
      payoff_matrix = payoff_matrix[None]
      variables = jax.tree.map(lambda x: x[None], variables)

    for i in range(payoff_matrix.shape[0]):
      verify_nash(payoff_matrix[i], jax.tree.map(lambda x: x[i], variables), atol=atol)

  stats['time'] = solve_time
  return stats


def test_rps(dtype=np.float64, **kwargs):
  payoff_matrix = np.array([
      [0, -1, 1],
      [1, 0, -1],
      [-1, 1, 0],
  ], dtype=dtype)
  with jax.enable_x64():
    return run_nash_test(payoff_matrix, **kwargs)


def test_random_nash(
    size: tuple[int, int] = (3, 3),
    dtype: np.dtype = np.float64,
    batch_size: int = 0,
    solver: tp.Optional[nash.NashSolver] = None,
    multi_device: bool = False,
    **kwargs,
):
  if batch_size > 0:
    dims = (batch_size, *size)
  else:
    dims = size
  payoff_matrix = np.random.standard_normal(dims).astype(dtype)

  with jax.enable_x64():
    if multi_device:
      mesh = jax_utils.get_mesh()
      sharding = jax_utils.data_sharding(mesh)
      payoff_matrix = jax.device_put(payoff_matrix, sharding)

    return run_nash_test(payoff_matrix, solver=solver, **kwargs)


def random_nash_tests(
    num_tests: int = 10,
    batch_size: int = 0,
    solver: tp.Optional[nash.NashSolver] = None,
    stat_keys: tuple[str, ...] = ('num_steps', 'total_violation'),
    **kwargs,
):
  all_stats = []
  solve_times = []
  for i in tqdm.trange(num_tests):
    stats = test_random_nash(
        batch_size=batch_size,
        solver=solver,
        **kwargs,
    )
    all_stats.append(stats)
    if i > 0:
      solve_times.append(stats['time'])

  if solve_times:
    bs = 1 if batch_size == 0 else batch_size
    total_solved = len(solve_times) * bs
    total_time = sum(solve_times)
    mean_time = total_time / total_solved
    problems_per_second = total_solved / total_time
    print(f'Mean solve time: {mean_time} s, {problems_per_second} problems/s')

  stats = utils.batch_nest(all_stats)

  for key in stat_keys:
    if key not in stats:
      continue
    values = stats[key]
    mean, std = np.mean(values), np.std(values)
    min_value = np.min(values)
    max_value = np.max(values)
    print(f'{key}: {mean:.1e} ± {std:.1e}, [{min_value:.1e}, {max_value:.1e}]')

def run_nash_tests(
    **solver_kwargs,
):
  solver_kwargs = dict(
    solver_kwargs,
    dtype=np.float64,
  )

  print('RPS')
  test_rps(**solver_kwargs)

  nash_kwargs = dict(
      solver_kwargs,
      num_tests=10,
      size=(10, 11),
      # jit=False,
      # debug=True,
  )

  print('Unbatched')
  random_nash_tests(**nash_kwargs)

  print('Batched')
  random_nash_tests(
      batch_size=10,
      **nash_kwargs,
  )

if __name__ == '__main__':
  test_solve_quadratic_optimization()

  for is_linear in [True, False]:
    print(f'Testing with is_linear={is_linear}')

    test_solve_corner_optimization(
        error=1e-3,
        max_size=3,
        is_linear=is_linear,
    )

    run_nash_tests(
        solver=nash.solve_zero_sum_nash_ippd,
        error=1e-5,
        max_steps=200,
        is_linear=is_linear,
    )

  # qpax solver tests
  print('\n--- qpax solver ---')

  print('Corner optimization (qpax)')
  test_solve_corner_optimization(
      max_size=3,
      solver=optimization.solve_optimization_qpax,
      error=1e-3,
      debug=True,
  )

  qpax_kwargs = dict(
      error=1e-5,
      max_steps=30,
      atol=1e-1,
      # debug=True,
      # jit=False,
  )

  print('qpax default')
  run_nash_tests(
      solver=nash.solve_zero_sum_nash_qpax,
      **qpax_kwargs,
  )

  qpax_fast_kwargs = dict(
      error=1e-5,
      atol=1e-3,
      # jit=False,
  )

  print('qpax fast')
  run_nash_tests(
      solver=nash.solve_zero_sum_nash_qpax_fast,
      **qpax_fast_kwargs,
  )
