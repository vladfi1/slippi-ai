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

class Player2NashLP(optimization.ConstrainedOptimizationProblem[PayoffMatrix, jax.Array]):
  """Player 2's LP for finding the Nash equilibrium via the simplex method.

  Reduces the zero-sum Nash problem to an LP in three steps:

  1. Eliminate the normalization constraint via unnormalized variables.
     Player 2's Nash LP is: min v s.t. A p2 <= v*1_m, p2 >= 0, 1^T p2 = 1.
     Let shift = min(A) - 1 so A_shifted = A - shift has all entries >= 1.
     Because both players use probability vectors, the shifted game value is
     v_shifted = v - shift > 0. Substituting z = p2 / v_shifted gives:
       A_shifted z <= 1_m    (divide the Nash constraint by v_shifted)
       1^T z = 1/v_shifted   (automatically satisfied at the LP optimum)
     Player 2 minimizes v_shifted = 1/(1^T z), equivalently maximizes 1^T z.
     The normalization equality drops out, leaving a pure inequality LP with
     z=0 as a feasible initial point (slacks = 1_m >= 0, no Phase I needed).

  2. Recover p1 from LP duality.
     The dual of max 1^T z s.t. A_shifted z <= 1_m, z >= 0 is:
       min 1_m^T y s.t. A_shifted^T y >= 1_n, y >= 0.
     Substituting w = p1 / v_shifted into player 1's Nash LP yields exactly
     this dual, so the optimal dual variables y = w are the unnormalized p1.
     In the simplex tableau, ineq_dual[i] = y[i] (shadow price of constraint i).
     Hence p1 = ineq_dual / sum(ineq_dual).

  3. Recover the game value.
     At the optimum, 1^T z* = 1/v_shifted, so v = 1/sum(z) + shift.

  Variables: z [N2] (unnormalized p2; z >= 0 is maintained structurally by the
  simplex solver and must NOT be listed as explicit constraints).
  """

  def initial_variables(self, parameters: PayoffMatrix) -> jax.Array:
    _, d2 = parameters.shape
    return jnp.zeros([d2], dtype=parameters.dtype)

  def objective(self, parameters: PayoffMatrix, variables: jax.Array) -> jax.Array:
    del parameters
    return -jnp.sum(variables)  # minimize -sum(z) = maximize sum(z)

  def constraint_violations(self, parameters: PayoffMatrix, variables: jax.Array) -> jax.Array:
    z = variables
    # Shift payoff matrix so all entries >= 1; then A*z <= 1_m defines the LP.
    shift = jnp.min(parameters) - jnp.ones([], dtype=parameters.dtype)
    A = parameters - shift
    return jnp.matvec(A, z) - jnp.ones([parameters.shape[0]], dtype=parameters.dtype)

  def equality_violations(self, parameters: PayoffMatrix, variables: jax.Array) -> jax.Array:
    del parameters
    return jnp.zeros([0], dtype=variables.dtype)


_player2_nash_lp = Player2NashLP()


def _solve_nash_simplex_impl(
    payoff_matrix: jax.Array,
    *,
    max_steps: int = 200,
    expected_dtype: tp.Optional[jnp.dtype] = None,
) -> tuple[NashVariables, dict]:
  """Solve zero-sum Nash equilibrium using the simplex method.

  Solves player 2's LP via Player2NashLP and recovers p1 from the dual variables.
  The LP has a natural initial BFS (z=0), so no Phase I is needed.
  """
  dtype = payoff_matrix.dtype
  z, ineq_dual, stats = optimization.solve_optimization_simplex_with_extras(
      _player2_nash_lp, payoff_matrix,
      max_steps=max_steps, expected_dtype=expected_dtype)

  # z is the unnormalized p2 strategy; ineq_dual[i] is the shadow price for
  # constraint (A_shifted @ z)[i] <= 1, which equals the dual weight for p1's pure strategy i.
  sum_z = jnp.sum(z)
  p2 = z / sum_z
  p1 = ineq_dual / jnp.sum(ineq_dual)

  # Value of the shifted game is 1/sum(z); undo the shift.
  shift = jnp.min(payoff_matrix) - jnp.ones([], dtype=dtype)
  v = sum_z ** -1 + shift

  return NashVariables(p1=p1, p2=p2, p1_nash_value=v), stats


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
