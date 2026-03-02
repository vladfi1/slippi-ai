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
    p2_payoffs = -jnp.matvec(parameters.transpose(), variables.p1)
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

_solve_zero_sum_nash_jax = jax_utils.partial(
    optimization.solve_feasibility_interior_point_primal_dual,
    ZeroSumNash)
_jitted_solve_zero_sum_nash_jax = optimization.jitted_ippd_feasibility_solver(ZeroSumNash)

def solve_zero_sum_nash_jax(
    payoff_matrix: np.ndarray,
    *,
    is_linear: bool = True,
    optimum: float | None = 0,
    jit: bool = True,
    **kwargs,
) -> tuple[NashVariables, dict]:
  if jit:
    solver = _jitted_solve_zero_sum_nash_jax
  else:
    solver = _solve_zero_sum_nash_jax

  return solver(payoff_matrix, is_linear=is_linear, optimum=optimum, **kwargs)
