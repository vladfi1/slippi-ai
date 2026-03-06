import abc
import functools
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from slippi_ai.jax import jax_utils

Parameters = tp.TypeVar('Parameters')
Variables = tp.TypeVar('Variables')

class FeasibilityProblem(abc.ABC, tp.Generic[Parameters, Variables]):
  """A problem of finding variables which satisfy constraints."""

  @abc.abstractmethod
  def constraint_violations(
    self,
    parameters: Parameters,
    variables: Variables,
  ) -> jax.Array:
    """Returns a vector of violations of inequality constraints.

    Positive means a violation, zero or negative means satisfied.
    """

  @abc.abstractmethod
  def equality_violations(self, parameters: Parameters, variables: Variables) -> jax.Array:
    """Returns a vector of violations of equality constraints.

    Zero means satisfied, nonzero means violated.
    """

  @abc.abstractmethod
  def initial_variables(self, parameters: Parameters) -> Variables:
    """Generate initial variables.

    These must satisfy the equality constraints, but in general will violate
    the inequality constraints.
    """

class ConstrainedOptimizationProblem(FeasibilityProblem[Parameters, Variables]):
  """An optimization problem with an objective function to minimize."""

  @abc.abstractmethod
  def initial_variables(self, parameters: Parameters) -> Variables:
    """Generate initial variables which satisfy all constraints."""

  @abc.abstractmethod
  def objective(self, parameters: Parameters, variables: Variables) -> jax.Array:
    """Compute the objective function to minimize."""


class SlackVariables(tp.NamedTuple, tp.Generic[Variables]):
  """Variables for a slack feasibility problem."""

  variables: Variables
  slack: jax.Array


class SlackFeasibilityProblem(ConstrainedOptimizationProblem[Parameters, SlackVariables[Variables]]):

  def __init__(
      self,
      problem: FeasibilityProblem[Parameters, Variables],
      initial_slack: float | jax.Array = 0.1,
  ):
    self.problem = problem
    self.initial_slack = initial_slack

  def initial_variables(self, parameters: Parameters) -> SlackVariables[Variables]:
    variables = self.problem.initial_variables(parameters)
    violations = self.problem.constraint_violations(parameters, variables)
    assert len(violations.shape) == 1
    self.violated = violations >= 0
    slack = jnp.max(violations) + self.initial_slack
    return SlackVariables(variables=variables, slack=slack)

  def constraint_violations(self, parameters: Parameters, variables: SlackVariables[Variables]) -> jax.Array:
    assert self.violated is not None
    slack = jnp.expand_dims(variables.slack, axis=-1)
    slack = jnp.where(self.violated, slack, 0)
    return self.problem.constraint_violations(parameters, variables.variables) - slack

  def equality_violations(self, parameters: Parameters, variables: SlackVariables[Variables]) -> jax.Array:
    return self.problem.equality_violations(parameters, variables.variables)

  def objective(self, parameters: Parameters, variables: SlackVariables[Variables]) -> jax.Array:
    return variables.slack


def jvp_bwd(
    f: tp.Callable[[jax.Array], jax.Array],
    x: jax.Array,
    dx: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  y, vjp = jax.vjp(f, x)
  dd, = vjp(dx)
  return y, dd


def jvp_fwd(
    f: tp.Callable[[jax.Array], jax.Array],
    x: jax.Array,
    dx: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  y, dy = jax.jvp(f, (x,), (dx,))
  return y, dy


def mat_inv(A: jax.Array, b: jax.Array) -> jax.Array:
  return jnp.squeeze(jnp.linalg.solve(A, jnp.expand_dims(b, -1)), -1)

def exponential_decay_search(
    condition: tp.Callable[[jax.Array], jax.Array],
    initial_value: jax.Array,
    decay: jax.Array | float = 0.5,
    max_iters: int = 100,
    debug: bool = False,
) -> jax.Array:

  initial_loop_vars = jnp.int32(0), initial_value

  def cond_fn(loop_vars: tuple[jax.Array, jax.Array]):
    i, value = loop_vars
    valid = condition(value)
    return jnp.logical_and(i < max_iters, ~valid)

  def body_fn(loop_vars):
    i, value = loop_vars
    value = value * decay
    return i + 1, value

  if debug:
    while_loop_fn = jax_utils.nonjit_while_loop
  else:
    while_loop_fn = jax.lax.while_loop

  _, value = while_loop_fn(cond_fn, body_fn, initial_loop_vars)
  # TODO: check that the result is actually valid
  return value

def line_search(
    objective: tp.Callable[[jax.Array], jax.Array],
    condition: tp.Callable[[jax.Array], jax.Array],
    variables: jax.Array,
    direction: jax.Array,
    directional_derivative: tp.Optional[jax.Array] = None,
    initial_step_size: float | jax.Array = 1.0,
    alpha: jax.Array | float = 0.1,
    beta: jax.Array | float = 0.5,
    max_iters: int = 200,
    debug: bool = False,
) -> jax.Array:
  alpha = jnp.asarray(alpha, dtype=variables.dtype)
  beta = jnp.asarray(beta, dtype=variables.dtype)

  if directional_derivative is None:
    objective_value, directional_derivative = jvp_fwd(
        objective, variables, direction)
  else:
    objective_value = objective(variables)

  assert direction.shape == variables.shape

  def take_step(size: jax.Array) -> jax.Array:
    return variables + jnp.expand_dims(size, -1) * direction

  feasible_step_size = exponential_decay_search(
      condition=lambda size: condition(take_step(size)),
      initial_value=jnp.asarray(initial_step_size, dtype=variables.dtype),
      decay=beta,
      max_iters=max_iters,
      debug=debug,
  )

  # Apparently this is known as the Armijo-Goldstein condition.
  def is_good_enough(size: jax.Array) -> jax.Array:
    new_objective = objective(take_step(size))
    decrease = new_objective - objective_value
    expected_decrease = size * directional_derivative
    return decrease <= alpha * expected_decrease

  step_size = exponential_decay_search(
      condition=is_good_enough,
      initial_value=feasible_step_size,
      decay=beta,
      max_iters=max_iters,
      debug=debug,
  )

  return take_step(step_size)


def _setup_flatten(
    variables: Variables,
) -> tuple[
    tp.Callable[[Variables], jax.Array],
    tp.Callable[[jax.Array], Variables],
    int,
]:
  flat_vars, treedef = jax.tree.flatten(variables)
  flat_shapes = [v.shape for v in flat_vars]
  flat_sizes = [int(np.prod(shape)) for shape in flat_shapes]
  flat_size = sum(flat_sizes)

  def flatten(vars_: Variables) -> jax.Array:
    leaves = jax.tree.leaves(vars_)
    if not leaves:
      return jnp.zeros([0], dtype=jnp.float32)
    return jnp.concatenate([
        jnp.reshape(v, [-1])
        for v in leaves], axis=-1)

  def unflatten(flat_var: jax.Array) -> Variables:
    if not flat_sizes:
      return jax.tree.unflatten(treedef, [])
    split_points = np.cumsum(flat_sizes)[:-1].tolist()
    split = jnp.split(flat_var, split_points, axis=-1)
    reshaped = [
        jnp.reshape(flat, shape)
        for flat, shape in zip(split, flat_shapes)]
    return jax.tree.unflatten(treedef, reshaped)

  return flatten, unflatten, flat_size

# https://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/primal-dual.pdf
def solve_optimization_interior_point_primal_dual(
    problem: ConstrainedOptimizationProblem[Parameters, Variables],
    parameters: Parameters,
    error: float | jax.Array = 1e-2,
    initial_constraint_weight: float | jax.Array = 1.0,
    constraint_weight_decay: float | jax.Array = 0.9,
    optimum: tp.Optional[float | jax.Array] = None,
    max_steps: int = 200,
    *,
    is_linear: bool = False,
    cholesky: bool = False,
    debug: bool = False,
    expected_dtype: tp.Optional[jnp.dtype] = jnp.float64,
) -> tuple[Variables, dict]:
  """Solve a convex optimization problem using a primal-dual interior point method.

  It is recommended that parameters are passed as float64. Even float32 can lead
  to instability in the optimization. To use float64 you will need to set the
  JAX_ENABLE_X64 environment variable to true. If you want to use float32, you
  should at least set JAX_DEFAULT_MATMUL_PRECISION=float32 -- otherwise jax will
  use "tensorflow32" which is faster but less precise. Even if instability is
  rare, one unstable problem will slow down the entire batch.

  Empircally, Nash problems take about 100 steps to converge with error=1e-4.

  Args:
    problem: The optimization problem to solve.
    parameters: Parameters defining the problem.
    error: The desired accuracy of the solution.
    initial_constraint_weight: Initial constraint weight.
    constraint_weight_decay: Decay factor for the constraint weight in each iteration.
    optimum: If provided, the known optimal value of the objective function. Used for logging and debugging.
    is_linear: If true, the objective and constraints are linear, which allows some optimizations.
    cholesky: If true, use Cholesky decomposition to solve the linear system in the Newton step.
    expected_dtype: Check that the variables have this dtype.
  """
  variables = problem.initial_variables(parameters)

  flatten, unflatten, flat_size = _setup_flatten(variables)

  flat_var = flatten(variables)
  dtype = flat_var.dtype
  if expected_dtype is not None:
    assert dtype == expected_dtype, f"Expected dtype {expected_dtype}, but got {dtype}"

  initial_constraints = problem.constraint_violations(parameters, variables)
  num_constraints = initial_constraints.shape[-1]
  num_equalities = jax.eval_shape(problem.equality_violations, parameters, variables).shape[-1]

  N = flat_size
  M = num_constraints
  K = num_equalities

  def objective(x_flat: jax.Array) -> jax.Array:
    x_struct = unflatten(x_flat)
    return problem.objective(parameters, x_struct)

  def constraint_violations(x_flat: jax.Array) -> jax.Array:
    x_struct = unflatten(x_flat)
    return problem.constraint_violations(parameters, x_struct)

  def equality_violations(x_flat: jax.Array) -> jax.Array:
    x_struct = unflatten(x_flat)
    return problem.equality_violations(parameters, x_struct)

  def split(combined: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    return jnp.split(combined, [flat_size, flat_size + num_constraints], axis=-1)

  def combine(vars_: Variables, constraint_vars: jax.Array, equality_vars: jax.Array) -> jax.Array:
    return jnp.concatenate([flatten(vars_), constraint_vars, equality_vars], axis=-1)

  def uncombine(combined: jax.Array) -> tuple[Variables, jax.Array, jax.Array]:
    vars_flat, constraint_vars, equality_vars = split(combined)
    return unflatten(vars_flat), constraint_vars, equality_vars

  def is_valid(combined: jax.Array) -> jax.Array:
    vars_ = uncombine(combined)[0]
    # Inequality here is tight because we want to be on the inside of the feasible region.
    return jnp.all(problem.constraint_violations(parameters, vars_) < 0, axis=-1)

  # Convex equality constraints must be linear, so the Jacobian is constant.
  A = jax.jacrev(equality_violations)(flat_var)  # [K, N]
  assert A.shape == (K, N)
  A_t = jnp.matrix_transpose(A)  # [N, K]

  # If the constraints and objective are linear, we can precompute the gradients.
  if is_linear:
    grad_f_linear = jax.grad(objective)(flat_var)  # [N]
    grad_g_linear = jax.jacrev(constraint_violations)(flat_var)  # [M, N]
    grad_g_t_linear = jnp.matrix_transpose(grad_g_linear)  # [N, M]

    def linear_residuals(combined: jax.Array, epsilon: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
      x, u, v = split(combined)
      x_struct = unflatten(x)
      g = problem.constraint_violations(parameters, x_struct)
      r_dual = grad_f_linear + jnp.matvec(grad_g_t_linear, u) + jnp.matvec(A_t, v)
      r_cent = u * g + jnp.expand_dims(epsilon, -1)
      r_prim = problem.equality_violations(parameters, x_struct)
      return r_dual, r_cent, r_prim

    def residual(combined: jax.Array, epsilon: jax.Array) -> jax.Array:
      return jnp.concatenate(linear_residuals(combined, epsilon), axis=-1)

  if not is_linear or debug:
    def lagrangian_with_aux(x, u, v):
      x_struct = unflatten(x)
      f = problem.objective(parameters, x_struct)
      g = problem.constraint_violations(parameters, x_struct)
      eq = problem.equality_violations(parameters, x_struct)
      L = f + jnp.vecdot(u, g) + jnp.vecdot(v, eq)
      return L, (g, eq)

    def residuals(combined: jax.Array, epsilon: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
      x, u, v = split(combined)
      r_dual, (g, eq) = jax.grad(lagrangian_with_aux, argnums=0, has_aux=True)(x, u, v)
      r_cent = u * g + jnp.expand_dims(epsilon, -1)
      r_prim = eq

      if is_linear:
        linear_r_dual, linear_r_cent, linear_r_prim = linear_residuals(combined, epsilon)

        for name, x, y in zip(['dual', 'cent', 'prim'], [r_dual, r_cent, r_prim], [linear_r_dual, linear_r_cent, linear_r_prim]):
          diff = jnp.max(jnp.abs(x - y))
          assert diff < 1e-3, f"Linear and nonlinear residuals differ too much for {name}: {diff}"

      return r_dual, r_cent, r_prim

    def residual(combined: jax.Array, epsilon: jax.Array) -> jax.Array:
      return jnp.concatenate(residuals(combined, epsilon), axis=-1)

  def newton_step(combined: jax.Array, epsilon: jax.Array) -> tuple[jax.Array, jax.Array]:
    x, u, v = split(combined)

    x_struct = unflatten(x)
    g = problem.constraint_violations(parameters, x_struct)  # [M]
    eq = problem.equality_violations(parameters, x_struct)  # [K]

    if is_linear:
      grad_g = grad_g_linear
      grad_g_t = grad_g_t_linear
      r_dual = grad_f_linear + jnp.matvec(grad_g_t, u) + jnp.matvec(A_t, v)
      H = jnp.zeros([N, N], dtype)
    else:
      def lagrangian(x: jax.Array) -> jax.Array:
        x_struct = unflatten(x)
        f = problem.objective(parameters, x_struct)
        g = problem.constraint_violations(parameters, x_struct)
        eq = problem.equality_violations(parameters, x_struct)
        return f + jnp.vecdot(u, g) + jnp.vecdot(v, eq)

      r_dual_fn = jax.grad(lagrangian)

      r_dual = r_dual_fn(x) # [N]
      H = jax.jacfwd(r_dual_fn)(x)  # [N, N]

      grad_g = jax.jacrev(constraint_violations)(x)  # [M, N]
      grad_g_t = jnp.matrix_transpose(grad_g)  # [N, M]

    r_cent = u * g + jnp.expand_dims(epsilon, -1)  # [M]
    r_prim = eq  # [K]
    residual_value = jnp.concatenate([r_dual, r_cent, r_prim], axis=-1)

    J_xx = H - jnp.matmul(grad_g_t * (u / g), grad_g)  # [N, N]
    J_xv = A_t  # [N, K]
    J_x = jnp.concatenate([J_xx, J_xv], axis=-1)  # [N, N + K]

    target_x = -r_dual + jnp.matvec(grad_g_t, r_cent / g)  # [N]

    J_vx = A  # [K, N]
    J_vv = jnp.zeros([K, K], dtype)
    J_v = jnp.concatenate([J_vx, J_vv], axis=-1)  # [K, N + K]
    target_v = -r_prim  # [K]

    if cholesky:
      chol_b = jax.scipy.linalg.cho_factor(J_xx)
      b_inv_a = jax.scipy.linalg.cho_solve(chol_b, A_t)  # [N, K]
      a_b_inv_a = jnp.matmul(A, b_inv_a)  # [K, K]
      chol_a_b_inv_a = jax.scipy.linalg.cho_factor(a_b_inv_a)

      chol_inv = lambda M, z: jnp.squeeze(
          jax.scipy.linalg.cho_solve(M, jnp.expand_dims(z, -1)), -1)
      b_inv_target_x = chol_inv(chol_b, target_x)

      new_target_v = jnp.matvec(A, b_inv_target_x) - target_v
      delta_v = chol_inv(chol_a_b_inv_a, new_target_v)
      # delta_v = mat_inv(a_b_inv_a, new_target_v)
      delta_x = b_inv_target_x - jnp.matvec(b_inv_a, delta_v)  # [N]
    else:
      J = jnp.concatenate([J_x, J_v], axis=-2)  # [N + K, N + K]
      target = jnp.concatenate([target_x, target_v], axis=-1)  # [N + K]
      delta_xv = mat_inv(J, target)
      delta_x, delta_v = jnp.split(delta_xv, [N], axis=-1)  # [N], [K]

    delta_u = (-r_cent - u * jnp.matvec(grad_g, delta_x)) / g  # [M]
    delta = jnp.concatenate([delta_x, delta_u, delta_v], axis=-1)

    # grad(residual) * delta = -residual

    # Make sure constraint vars remain positive.
    if num_constraints > 0:
      max_step_sizes = jnp.where(delta_u < 0, -u / delta_u, 1.0)
      max_step_size = jnp.min(max_step_sizes, axis=-1)
      max_step_size = jnp.minimum(max_step_size, 1.0) * 0.99
    else:
      max_step_size = jnp.ones([], dtype=dtype)

    def residual_objective(comb: jax.Array) -> jax.Array:
      rv = residual(comb, epsilon)
      return 0.5 * jnp.sum(jnp.square(rv), axis=-1)

    # grad(residual_objective) = residual * grad(residual)
    # => grad(residual_objective) * delta
    # = residual * grad(residual) * delta
    # = residual * -residual = - |residual|^2

    new_combined = line_search(
        objective=residual_objective,
        condition=is_valid,
        variables=combined,
        direction=delta,
        directional_derivative=-jnp.sum(jnp.square(residual_value), axis=-1),
        initial_step_size=max_step_size,
        max_iters=max_steps,
        debug=debug,
    )

    return new_combined, residual_value

  error = jnp.asarray(error, dtype=dtype)
  if optimum is not None:
    optimum = jnp.asarray(optimum, dtype=dtype)

  constraint_weight_decay = jnp.asarray(constraint_weight_decay, dtype=dtype)
  u0 = jnp.asarray(initial_constraint_weight, dtype=dtype)
  constraint_vars = jnp.full([num_constraints], u0)
  equality_vars = jnp.zeros([num_equalities], dtype=dtype)
  combined = jnp.concatenate([flat_var, constraint_vars, equality_vars], axis=-1)

  eta = -jnp.vecdot(constraint_vars, initial_constraints)

  num_steps = jax_utils.as_vma(jnp.int32(1), combined)
  done = jax_utils.as_vma(jnp.bool(False), combined)

  initial_loop_vars = done, num_steps, combined, eta

  def body_fn(loop_vars):
    _, num_steps, combined, eta = loop_vars

    epsilon = constraint_weight_decay * eta / num_constraints
    combined, residual_value = newton_step(combined, epsilon)

    variables, constraint_vars, _ = uncombine(combined)
    constraints = problem.constraint_violations(parameters, variables)
    eta = -jnp.vecdot(constraints, constraint_vars)

    r_dual, _, r_prim = split(residual_value)
    r_dual_norm2 = jnp.sum(jnp.square(r_dual), axis=-1)
    r_prim_norm2 = jnp.sum(jnp.square(r_prim), axis=-1)

    if optimum is not None:
      done = jnp.logical_and(
          problem.objective(parameters, variables) <= optimum + error,
          r_prim_norm2 <= jnp.square(error),
      )
    else:
      feasible = r_dual_norm2 + r_prim_norm2 <= jnp.square(error)
      done = jnp.logical_and(eta <= error, feasible)

    return done, num_steps + 1, combined, eta

  def cond_fn(loop_vars):
    done, num_steps, _, _ = loop_vars
    return jnp.logical_and(~done, num_steps < max_steps)

  if debug:
    while_loop_fn = jax_utils.nonjit_while_loop
  else:
    while_loop_fn = jax.lax.while_loop

  done, num_steps, combined, eta = while_loop_fn(
    cond_fn, body_fn, initial_loop_vars)

  final_vars, _, _ = uncombine(combined)
  stats = dict(num_steps=num_steps)
  return final_vars, stats

Stats = dict
P = tp.ParamSpec('P')
Solver = tp.Callable[
    tp.Concatenate[ConstrainedOptimizationProblem[Parameters, Variables], Parameters, P],
    tuple[Variables, Stats]]

def as_feasibility_solver(
    optimization_solver: Solver[Parameters, SlackVariables[Variables], P],
):
  def feasibility_solver(
      problem: FeasibilityProblem[Parameters, Variables],
      parameters: Parameters,
      *solver_args: P.args,
      **solver_kwargs: P.kwargs,
  ) -> tuple[Variables, Stats]:
    slack_problem = SlackFeasibilityProblem(problem)
    variables, stats = optimization_solver(
      slack_problem, parameters, *solver_args, **solver_kwargs)
    stats['slack'] = variables.slack

    vs = variables.variables
    eq = problem.equality_violations(parameters, vs)
    ineq = problem.constraint_violations(parameters, vs)

    total_violation = jnp.sum(jnp.maximum(0, ineq)) + jnp.sum(jnp.abs(eq))

    stats.update(
        max_equality_violation=jnp.max(jnp.abs(eq)),
        max_inequality_violation=jnp.max(ineq),
        total_violation=total_violation,
    )

    return vs, stats

  return feasibility_solver

solve_feasibility_ippd = as_feasibility_solver(
  solve_optimization_interior_point_primal_dual)

_ippd_static_argnames = ['is_linear', 'cholesky', 'expected_dtype', 'debug']

def jitted_ippd_feasibility_solver(
    problem: FeasibilityProblem[Parameters, Variables],
):
  solver = jax_utils.partial(solve_feasibility_ippd, problem)
  return jax_utils.jit(solver, static_argnames=_ippd_static_argnames)

def vmap1_ippd_feasibility_solver(
    problem: FeasibilityProblem[Parameters, Variables],
):
  solver = jax_utils.partial(solve_feasibility_ippd, problem)
  return jax_utils.vmap1(solver, static_argnames=_ippd_static_argnames)
