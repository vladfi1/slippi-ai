import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from slippi_ai.jax import jax_utils
from slippi_ai.jax.nash import optimization
from slippi_ai.nash.nash import solve_zero_sum_nash_pulp  # re-export

PayoffMatrix = jax.Array

class NashVariables(tp.NamedTuple):
  p1: jax.Array  # [N1]
  p2: jax.Array  # [N2]
  p1_nash_value: jax.Array  # []


class ZeroSumNashProblem(optimization.FeasibilityProblem[PayoffMatrix, NashVariables]):

  def initial_variables(self, parameters: PayoffMatrix) -> NashVariables:
    d1, d2 = parameters.shape
    dtype = parameters.dtype

    return NashVariables(
        p1=jnp.ones([d1], dtype=dtype) / d1,
        p2=jnp.ones([d2], dtype=dtype) / d2,
        p1_nash_value=jnp.zeros([], dtype=dtype),
    )

  def constraint_violations(self, parameters: PayoffMatrix, variables: NashVariables) -> jax.Array:
    # Probabilities must be non-negative
    constraints = [
        -variables.p1,
        -variables.p2,
    ]

    # No strategy for p1 does better than the nash value
    p1_payoffs = jnp.matvec(parameters, variables.p2)
    p1_nash_value = jnp.expand_dims(variables.p1_nash_value, -1)
    p1_optimality = p1_payoffs - p1_nash_value

    # No strategy for p2 does better than the nash value
    # p2_payoffs = -jnp.vecmat(variables.p1, parameters)
    p2_payoffs = -jnp.matvec(parameters.T, variables.p1)
    p2_nash_value = -jnp.expand_dims(variables.p1_nash_value, -1)
    p2_optimality = p2_payoffs - p2_nash_value

    constraints.extend([p1_optimality, p2_optimality])
    return jnp.concatenate(constraints, axis=-1)

  def equality_violations(self, parameters: PayoffMatrix, variables: NashVariables) -> jax.Array:
    del parameters
    return jnp.stack([
        jnp.sum(variables.p1, axis=-1) - 1.0,
        jnp.sum(variables.p2, axis=-1) - 1.0,
    ], axis=-1)

ZeroSumNash = ZeroSumNashProblem()

_solve_zero_sum_nash_ippd = jax_utils.partial(
    optimization.solve_feasibility_ippd,
    ZeroSumNash)
_jitted_solve_zero_sum_nash_ippd = optimization.jitted_ippd_feasibility_solver(ZeroSumNash)
_batched_solve_zero_sum_nash_ippd = optimization.vmap1_ippd_feasibility_solver(ZeroSumNash)

def solve_zero_sum_nash_ippd(
    payoff_matrix: np.ndarray | jax.Array,
    *,
    is_linear: bool = True,
    optimum: float | None = 0,
    jit: bool = True,
    **kwargs,
) -> tuple[NashVariables, dict]:
  if payoff_matrix.ndim == 3:
    solver = _batched_solve_zero_sum_nash_ippd
  elif jit:
    solver = _jitted_solve_zero_sum_nash_ippd
  else:
    solver = _solve_zero_sum_nash_ippd

  return solver(
    jnp.asarray(payoff_matrix),
    is_linear=is_linear, optimum=optimum,
    expected_dtype=payoff_matrix.dtype,
    **kwargs)


_solve_zero_sum_nash_qpax = jax_utils.partial(
    optimization.solve_feasibility_qpax,
    ZeroSumNash)
_jitted_solve_zero_sum_nash_qpax = optimization.jitted_qpax_feasibility_solver(ZeroSumNash)
_batched_solve_zero_sum_nash_qpax = optimization.vmap1_qpax_feasibility_solver(ZeroSumNash)


def solve_zero_sum_nash_qpax(
    payoff_matrix: np.ndarray | jax.Array,
    *,
    jit: bool = True,
    debug: bool = False,
    **kwargs,
) -> tuple[NashVariables, dict]:
  """Solve a zero-sum Nash equilibrium using qpax's LP solver."""
  if payoff_matrix.ndim == 3:
    assert not debug
    solver = _batched_solve_zero_sum_nash_qpax
  elif jit and not debug:
    solver = _jitted_solve_zero_sum_nash_qpax
  else:
    solver = _solve_zero_sum_nash_qpax

  return solver(
      jnp.asarray(payoff_matrix),
      expected_dtype=payoff_matrix.dtype,
      debug=debug,
      **kwargs)

def _solve_nash_simplex_impl(
    payoff_matrix: jax.Array,
    *,
    max_steps: int = 200,
    expected_dtype: tp.Optional[jnp.dtype] = None,
) -> tuple[NashVariables, dict]:
  """Solve zero-sum Nash equilibrium using the simplex method.

  Solves player 2's LP: maximize 1^T z s.t. A z + s = 1_m, z >= 0, s >= 0
  where A is shifted to have all positive entries. Extracts p1 from dual variables.

  The LP has a natural initial BFS (z=0, s=1), so no Phase I is needed.
  """
  m, n = payoff_matrix.shape
  dtype = payoff_matrix.dtype

  if expected_dtype is not None:
    assert dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {dtype}"

  eps = jnp.asarray(1e-9, dtype=dtype)
  inf = jnp.asarray(jnp.inf, dtype=dtype)

  # Shift to make all entries > 0 so the LP value is positive
  shift = jnp.min(payoff_matrix) - jnp.ones([], dtype=dtype)
  A = payoff_matrix - shift  # All entries >= 1

  # Tableau: (m+1, n+m+1)
  # Rows 0..m-1: [A | I_m | 1_m]  (constraints)
  # Row m:       [1_n | 0_m | 0]  (objective, positive reduced costs for max)
  tab = jnp.concatenate([
      jnp.concatenate([A, jnp.eye(m, dtype=dtype), jnp.ones([m, 1], dtype=dtype)], axis=1),
      jnp.concatenate([jnp.ones([1, n], dtype=dtype), jnp.zeros([1, m + 1], dtype=dtype)], axis=1),
  ], axis=0)

  # Initial basis: slack variables s_i in row i (columns n..n+m-1)
  basis = jnp.arange(n, n + m, dtype=jnp.int32)
  done = jnp.zeros([], dtype=jnp.bool_)

  def step(carry: tuple[jax.Array, jax.Array, jax.Array], _):
    tab, basis, done = carry

    obj = tab[m, :n + m]
    entering = jnp.argmax(obj).astype(jnp.int32)
    optimal = obj[entering] <= eps

    col = tab[:m, entering]
    rhs = tab[:m, -1]
    ratios = jnp.where(col > eps, rhs / col, inf)
    leaving = jnp.argmin(ratios).astype(jnp.int32)
    pval = tab[leaving, entering]

    prow = tab[leaving, :]
    entering_col_vals = tab[:, entering]
    new_tab = tab - (entering_col_vals / pval)[:, None] * prow[None, :]
    is_pivot_row = jnp.arange(m + 1) == leaving
    new_tab = jnp.where(is_pivot_row[:, None], (prow / pval)[None, :], new_tab)

    new_basis = basis.at[leaving].set(entering)

    update = ~done & ~optimal & (pval > eps)
    return (
        jnp.where(update, new_tab, tab),
        jnp.where(update, new_basis, basis),
        done | optimal,
    ), None

  (tab, basis, _), _ = jax.lax.scan(step, (tab, basis, done), None, length=max_steps)

  # Extract p2: z[j] = tab[row, -1] for the row where basis[row] == j
  eq = basis[:, None] == jnp.arange(n)[None, :]  # (m, n)
  z = jnp.sum(jnp.where(eq, tab[:m, -1:], jnp.zeros_like(tab[:m, -1:])), axis=0)
  sum_z = jnp.sum(z)
  p2 = z / sum_z

  # Extract p1 from dual: y[i] = -tab[m, n+i] (reduced cost of slack var i)
  # In this convention (positive = enter), tab[m, n+i] = -y_i <= 0 at optimality.
  y = -tab[m, n:n + m]
  p1 = y / jnp.sum(y)

  # Game value: v_shifted = 1/sum(z*); undo the shift.
  # Note: tab[m, -1] = -sum(z*) in this convention, so use sum_z directly.
  v = sum_z ** -1 + shift

  return NashVariables(p1=p1, p2=p2, p1_nash_value=v), {}


_simplex_static_argnames = ('max_steps', 'expected_dtype')

_jitted_solve_nash_simplex = jax_utils.jit(
    _solve_nash_simplex_impl,
    static_argnames=_simplex_static_argnames)

_batched_solve_nash_simplex = jax_utils.vmap1(
    _solve_nash_simplex_impl,
    static_argnames=_simplex_static_argnames)


def solve_zero_sum_nash_simplex(
    payoff_matrix: np.ndarray | jax.Array,
    *,
    jit: bool = True,
    max_steps: int = 200,
    **kwargs,
) -> tuple[NashVariables, dict]:
  """Solve zero-sum Nash equilibrium using the simplex method (vmappable)."""
  if payoff_matrix.ndim == 3:
    solver = _batched_solve_nash_simplex
  elif jit:
    solver = _jitted_solve_nash_simplex
  else:
    solver = _solve_nash_simplex_impl

  return solver(
      jnp.asarray(payoff_matrix),
      max_steps=max_steps,
      expected_dtype=payoff_matrix.dtype,
      **kwargs)


class NashSolver(tp.Protocol):
  def __call__(
      self,
      payoff_matrix: np.ndarray | jax.Array,
      *,
      jit: bool,
      max_steps: int,
      error: float,
      **kwargs,
  ) -> tuple[NashVariables, dict]: ...


class P1NashVariables(tp.NamedTuple):
  p1: jax.Array  # [N1]
  p1_value: jax.Array  # []

class P1ZeroSumNashProblem(optimization.ConstrainedOptimizationProblem[PayoffMatrix, P1NashVariables]):
  """Player 1 is the row player and is maximizing."""

  def initial_variables(self, parameters: PayoffMatrix) -> P1NashVariables:
    d1, _ = parameters.shape
    dtype = parameters.dtype

    return P1NashVariables(
        p1=jnp.ones([d1], dtype=dtype) / d1,
        p1_value=jnp.zeros([], dtype=dtype),
    )

  def objective(self, parameters: jax.Array, variables: P1NashVariables) -> jax.Array:
    del parameters
    return -variables.p1_value

  def constraint_violations(self, parameters: PayoffMatrix, variables: P1NashVariables) -> jax.Array:
    # Probabilities must be non-negative
    constraints = [
        -variables.p1,
    ]

    # p1's value is the min over p2's responses to p1
    payoffs = jnp.vecmat(variables.p1, parameters)
    p1_value = jnp.expand_dims(variables.p1_value, -1)
    constraints.append(p1_value - payoffs)

    return jnp.concatenate(constraints, axis=-1)

  def equality_violations(self, parameters: PayoffMatrix, variables: P1NashVariables) -> jax.Array:
    del parameters
    return jnp.sum(variables.p1, axis=-1, keepdims=True) - 1.0


def _solve_zero_sum_nash_qpax_fast(
    payoff_matrix: np.ndarray | jax.Array,
    **kwargs,
) -> tuple[NashVariables, dict]:
  N, M = payoff_matrix.shape
  problem = P1ZeroSumNashProblem()

  opt_x, extras, stats = optimization.solve_optimization_qpax_with_extras(
      problem,
      payoff_matrix,
      **kwargs,
  )

  p1 = opt_x.p1
  p1_value = opt_x.p1_value

  # Recover p2 from the dual variables
  assert extras.ineq_dual.shape == (N + M,)
  p2 = extras.ineq_dual[N:]

  return NashVariables(p1=p1, p2=p2, p1_nash_value=p1_value), stats

_jitted_solve_zero_sum_nash_qpax_fast = jax_utils.jit(
    _solve_zero_sum_nash_qpax_fast,
    static_argnames=optimization.qpax_static_argnames)

_batched_solve_zero_sum_nash_qpax_fast = jax_utils.vmap1(
    _solve_zero_sum_nash_qpax_fast,
    static_argnames=optimization.qpax_static_argnames)

def solve_zero_sum_nash_qpax_fast(
    payoff_matrix: np.ndarray | jax.Array,
    *,
    jit: bool = True,
    **kwargs,
) -> tuple[NashVariables, dict]:
  """Solve a zero-sum Nash equilibrium using qpax's LP solver."""
  if payoff_matrix.ndim == 3:
    solver = _batched_solve_zero_sum_nash_qpax_fast
  elif jit:
    solver = _jitted_solve_zero_sum_nash_qpax_fast
  else:
    solver = _solve_zero_sum_nash_qpax_fast

  return solver(
      payoff_matrix,
      expected_dtype=payoff_matrix.dtype,
      **kwargs)
