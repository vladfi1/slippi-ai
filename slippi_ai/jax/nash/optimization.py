import abc
import functools
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from slippi_ai.jax import jax_utils

from linrax.optim import linprog

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

class QPaxExtras(tp.NamedTuple):
  ineq_slack: jax.Array
  ineq_dual: jax.Array
  eq_dual: jax.Array

def solve_optimization_qpax_with_extras(
    problem: ConstrainedOptimizationProblem[Parameters, Variables],
    parameters: Parameters,
    *,
    error: float,
    max_steps: int = 30,
    expected_dtype: tp.Optional[jnp.dtype] = None,
    debug: bool = False,
    **_,
) -> tuple[Variables, QPaxExtras, Stats]:
  """Solve a linear program (LP) using qpax's primal-dual interior point method.

  Assumes the objective and all constraints are linear in the variables.

  Args:
    problem: A ConstrainedOptimizationProblem with linear objective and constraints.
    parameters: Parameters defining the problem.
    error: KKT residual tolerance for convergence.
    max_steps: Maximum number of iterations.
    expected_dtype: Check that the variables have this dtype.
  """
  import qpax

  variables = problem.initial_variables(parameters)
  flatten, unflatten, flat_size = _setup_flatten(variables)
  flat_var = flatten(variables)

  dtype = flat_var.dtype
  if expected_dtype is not None:
    assert dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {dtype}"

  N = flat_size

  def obj_fn(x: jax.Array) -> jax.Array:
    return problem.objective(parameters, unflatten(x))

  def constr_fn(x: jax.Array) -> jax.Array:
    return problem.constraint_violations(parameters, unflatten(x))

  def eq_fn(x: jax.Array) -> jax.Array:
    return problem.equality_violations(parameters, unflatten(x))

  # LP: Q = 0, q = gradient of (linear) objective
  Q = jnp.zeros([N, N], dtype=dtype)
  q = jax.grad(obj_fn)(flat_var)

  # Inequality constraints: constr_fn(x) = G @ x + c <= 0, i.e. G @ x <= h = -c
  G = jax.jacrev(constr_fn)(flat_var)  # [M, N]
  h = G @ flat_var - constr_fn(flat_var)

  # Equality constraints: eq_fn(x) = A @ x + d = 0, i.e. A @ x = b = -d
  A = jax.jacrev(eq_fn)(flat_var)  # [K, N]
  b = A @ flat_var - eq_fn(flat_var)

  x_opt, s, z, y, converged, num_steps = qpax.pdip.solve_qp(
      Q, q, A, b, G, h, solver_tol=error, max_iter=max_steps, debug=debug)

  vs = unflatten(x_opt)
  eq = problem.equality_violations(parameters, vs)
  ineq = problem.constraint_violations(parameters, vs)

  total_violation = jnp.sum(jnp.maximum(0, ineq)) + jnp.sum(jnp.abs(eq))

  extras = QPaxExtras(
      ineq_slack=s,
      ineq_dual=z,
      eq_dual=y,
  )

  stats = dict(
      # max_equality_violation=jnp.max(jnp.abs(eq)),
      # max_inequality_violation=jnp.max(ineq),
      total_violation=total_violation,
      converged=converged,
      num_steps=num_steps,
  )

  return vs, extras, stats

def solve_optimization_qpax(
    problem: ConstrainedOptimizationProblem[Parameters, Variables],
    parameters: Parameters,
    *,
    error: float,
    max_steps: int = 30,
    expected_dtype: tp.Optional[jnp.dtype] = None,
    debug: bool = False,
    **_,
) -> tuple[Variables, Stats]:
  variables, _, stats = solve_optimization_qpax_with_extras(
      problem, parameters, error=error, max_steps=max_steps, expected_dtype=expected_dtype, debug=debug)
  return variables, stats


solve_feasibility_qpax = as_feasibility_solver(solve_optimization_qpax)

qpax_static_argnames = ['expected_dtype', 'debug']


def jitted_qpax_feasibility_solver(
    problem: FeasibilityProblem[Parameters, Variables],
):
  solver = jax_utils.partial(solve_feasibility_qpax, problem)
  return jax_utils.jit(solver, static_argnames=qpax_static_argnames)


def vmap1_qpax_feasibility_solver(
    problem: FeasibilityProblem[Parameters, Variables],
):
  solver = jax_utils.partial(solve_feasibility_qpax, problem)
  return jax_utils.vmap1(solver, static_argnames=qpax_static_argnames)


def ruiz_equilibration(
    G: jax.Array,
    num_iters: int,
) -> tuple[jax.Array, jax.Array]:
  """Compute Ruiz diagonal scaling factors for the matrix G.

  Ruiz equilibration finds positive diagonal scalings D_r = diag(d_r) and
  D_c = diag(d_c) such that every row and column of the scaled matrix
  D_r @ G @ D_c has infinity-norm close to 1. This makes the entries O(1),
  which (a) lets a single fixed tolerance like sqrt(machine_eps) be meaningful
  across the whole tableau and (b) improves the conditioning of the pivots.

  The iteration repeatedly divides each row and column by the square root of its
  current max absolute entry; the square root splits each rebalancing step evenly
  between the row and column factors, which is what makes the two-sided iteration
  converge. A fixed `num_iters` is run unconditionally (no convergence test) so
  the routine stays jit/vmap-friendly; ~5-10 iterations is typically plenty.

  Zero rows/columns (max entry 0) are left unscaled (factor 1) to avoid division
  by zero.

  Args:
    G: Matrix to equilibrate, shape [m, n].
    num_iters: Number of Ruiz iterations to run.

  Returns:
    (d_r, d_c) with shapes [m] and [n]: the diagonal row and column scale
    factors. The equilibrated matrix is d_r[:, None] * G * d_c[None, :].
  """
  m, n = G.shape
  dtype = G.dtype

  def body(_, carry):
    M, d_r, d_c = carry
    r = jnp.sqrt(jnp.max(jnp.abs(M), axis=1))  # [m]
    c = jnp.sqrt(jnp.max(jnp.abs(M), axis=0))  # [n]
    # Leave zero rows/columns unscaled.
    r = jnp.where(r > 0, r, jnp.ones_like(r))
    c = jnp.where(c > 0, c, jnp.ones_like(c))
    M = M / r[:, None] / c[None, :]
    return M, d_r / r, d_c / c

  init = (G, jnp.ones([m], dtype), jnp.ones([n], dtype))
  _, d_r, d_c = jax.lax.fori_loop(0, num_iters, body, init)
  return d_r, d_c

SimplexVariables = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def solve_lp_simplex(
    c: jax.Array,
    G: jax.Array,
    h: jax.Array,
    *,
    max_steps: int = 200,
    equilibrate_iters: int = 5,
    expected_dtype: tp.Optional[jnp.dtype] = None,
) -> tuple[jax.Array, jax.Array, Stats]:
  """Solve LP: max c^T x s.t. G x <= h, x >= 0 using the simplex method.

  This is a JAX-friendly implementation of the (primal) simplex method in
  "standard form". Every step is expressed as dense array operations on a single
  tableau so that the whole routine can be jitted, vmapped, and differentiated
  through `jax.lax.while_loop` with a fixed memory footprint.

  Shapes and batching
  -------------------
  This function solves a SINGLE LP; it is NOT internally batched. It expects:

      c: [n]        G: [m, n]        h: [m]
      ->  x_opt: [n]   ineq_dual: [m]

  where n is the number of decision variables and m the number of inequalities.
  `m, n = G.shape` assumes G is exactly 2-D, and all intermediates (the tableau,
  basis, etc.) carry no batch axis. To solve many LPs at once, vmap the whole
  function over a leading batch axis (see `vmap1_simplex_feasibility_solver` /
  `jax_utils.vmap1`, which map over the leading axis of the parameters). All
  shapes documented below and inline are for the un-batched problem.

  Standard form and slack variables
  ---------------------------------
  We introduce one nonnegative slack variable s_i per inequality to turn the
  inequalities into equalities:

      max  c^T x
      s.t. G x + s = h,   x >= 0,   s >= 0

  Stacking the decision and slack variables as z = [x; s] (length n + m), the
  feasible region is described by the equalities [G | I] z = h together with
  z >= 0. The simplex method walks along vertices ("basic feasible solutions",
  BFS) of this polytope, at each step swapping one variable into the basis and
  one out, until no objective-improving swap remains.

  Why h >= 0 is required
  ----------------------
  The algorithm needs a feasible starting vertex and uses the trivial one x = 0,
  s = h (i.e. the slacks form the initial basis). That point is feasible iff
  s = h >= 0. We therefore *require* h >= 0 and skip the usual "phase 1" that
  would otherwise be needed to find an initial feasible vertex. Callers who only
  have h >= 0 by construction (e.g. slacks of an already-feasible point) can use
  this directly; others must pre-process their problem.

  Tableau layout
  --------------
  We store an (m + 1) x (n + m + 1) tableau `tab`:

      columns:  [ 0 .. n-1 ]      decision variables x
                [ n .. n+m-1 ]    slack variables s
                [ n+m ]           right-hand side (RHS)

      rows 0 .. m-1:  the constraint rows [G | I | h]
      row  m:         the objective row   [c | 0 | 0]

  An invariant maintained throughout: each basic variable appears as a unit
  column (a single 1 in its constraint row, 0 elsewhere including the objective
  row). The RHS column of the constraint rows then holds the current values of
  the basic variables, and the RHS of the objective row holds the current
  objective value (negated, see below). `basis[r]` records which variable index
  is basic in row r; it starts as the slacks (columns n .. n+m-1).

  One simplex iteration (Dantzig's rule)
  --------------------------------------
  - Entering variable: pick the column with the largest objective-row coefficient
    `obj`. A positive coefficient means increasing that variable still improves
    the (maximization) objective. If the largest coefficient is <= eps, no
    improving direction remains and we are optimal.
  - Leaving variable: increasing the entering variable is blocked when some basic
    variable would go negative. The min-ratio test rhs/col over rows with a
    positive entry `col > eps` finds the first basic variable to hit zero; that
    row leaves the basis. (Rows with col <= eps impose no bound and get ratio
    +inf so they are never selected.)
  - Pivot: Gauss-Jordan eliminate the entering column so it becomes a unit column
    on the leaving row. This updates the whole tableau, including the objective
    row, in one rank-1 update.

  Branchless control flow for JAX
  -------------------------------
  Because everything runs inside `lax.while_loop` we cannot branch out early.
  Instead each iteration always computes a candidate pivot, then uses
  `can_pivot = ~optimal & (pval > eps)` to decide (via `jnp.where`) whether to
  commit it. When we are optimal or no valid pivot exists (degenerate / pval too
  small), we leave the tableau unchanged and set `done`, which stops the loop.
  This also guards against dividing by a near-zero pivot.

  Dual variables
  --------------
  At optimality the reduced cost of slack s_i (the objective-row entry in its
  column, `tab[m, n+i]`) equals the negative of the dual price of constraint i.
  Hence `ineq_dual[i] = -tab[m, n+i] >= 0` is the (nonnegative) dual / shadow
  price for `G[i] @ x <= h[i]`.

  Numerical notes and equilibration
  ---------------------------------
  `eps = sqrt(machine_eps)` is used both as the optimality threshold and as the
  guard for "is this entry effectively zero". A fixed small constant like 1e-9 is
  below float32 machine epsilon (~1.2e-7) and would fail to prevent tiny pivots,
  letting the tableau blow up to inf/NaN. For ill-conditioned problems prefer
  float64.

  Because that tolerance is ABSOLUTE, it implicitly assumes the data is O(1).
  The problem `max c^T x s.t. G x <= h` is invariant under rescaling the data
  (e.g. multiplying c, G, h by 1/1000 leaves the optimum unchanged), but a fixed
  absolute tolerance is not -- it would then reject legitimate pivots as "zero".
  To remove this dependence we first equilibrate: we pick positive diagonal
  scalings D_r, D_c (via `ruiz_equilibration`) so the scaled matrix
  G' = D_r G D_c has O(1) rows and columns, solve the equivalent LP

      max  c'^T x'   s.t.   G' x' <= h',   x' >= 0
      where  c' = D_c c,   h' = D_r h,   x = D_c x'

  and unscale the result: the primal is x = D_c x' and the inequality duals are
  u = D_r u' (column scaling is a change of variables and does not touch the
  duals; row scaling rescales each constraint, so its dual scales with D_r).
  Equilibration preserves h >= 0 (D_r > 0) and x >= 0 (D_c > 0), so the initial
  BFS is still valid. Pass equilibrate_iters=0 to disable.

  Args:
    c: Objective coefficients, shape [n]. The objective `c^T x` is MAXIMIZED.
    G: Inequality constraint matrix, shape [m, n], for `G x <= h`.
    h: Inequality right-hand side, shape [m]. MUST satisfy h >= 0.
    max_steps: Maximum number of pivot iterations before giving up. Acts as a
      safety bound; well-posed problems converge in O(m) pivots in practice.
    equilibrate_iters: Number of Ruiz equilibration iterations to apply to the
      problem data before solving (0 disables). See `ruiz_equilibration`.
    expected_dtype: If provided, assert that G has this dtype.

  Returns:
    A tuple (x_opt, ineq_dual, stats):
      x_opt: Optimal decision variables, shape [n].
      ineq_dual: Nonnegative dual variables, shape [m], one per inequality.
      stats: Dict with 'num_steps', the number of iterations actually taken.
  """
  m, n = G.shape
  dtype = G.dtype

  if expected_dtype is not None:
    assert dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {dtype}"

  # Equilibrate so the data is O(1) and the absolute tolerance below is meaningful.
  # We solve the scaled LP (c', G', h') and unscale the result at the end.
  if equilibrate_iters > 0:
    d_r, d_c = ruiz_equilibration(G, equilibrate_iters)  # [m], [n]
    c = d_c * c
    G = d_r[:, None] * G * d_c[None, :]
    h = d_r * h
  else:
    d_r = jnp.ones([m], dtype)
    d_c = jnp.ones([n], dtype)

  # Use a dtype-relative epsilon. For float32, 1e-9 < machine epsilon (~1.2e-7),
  # so it fails to guard against near-zero pivots, causing tableau overflow and NaN.
  # pow(eps_machine, 0.6) balances numerical safety with sensitivity (~7e-5 for f32).
  eps = jnp.pow(jnp.finfo(dtype).eps, 0.6).astype(dtype)
  inf_val = jnp.asarray(jnp.inf, dtype=dtype)

  # Standard form: max [c; 0]^T [x; s] s.t. [G | I][x; s] = h, x,s >= 0
  # Tableau rows 0..m-1: [G | I | h]; row m: [c | 0 | 0]
  tab = jnp.concatenate([
      jnp.concatenate([G, jnp.eye(m, dtype=dtype), h[:, None]], axis=1),  # [m, n+m+1]
      jnp.concatenate([c[None, :], jnp.zeros([1, m + 1], dtype=dtype)], axis=1),  # [1, n+m+1]
  ], axis=0)  # tab: [m+1, n+m+1]

  # Initial basis: slack variables (columns n..n+m-1)
  basis = jnp.arange(n, n + m, dtype=jnp.int32)  # [m]

  def cond_fun(carry: tuple) -> jax.Array:
    _, _, done, step = carry
    return ~done & (step < max_steps)

  def body_fun(carry: SimplexVariables) -> SimplexVariables:
    tab, basis, done, step = carry  # tab: [m+1, n+m+1], basis: [m]

    # Entering variable: column with the most positive reduced cost (Dantzig).
    # If even the best column can't improve the objective, we're optimal.
    obj = tab[m, :n + m]  # [n+m]
    entering = jnp.argmax(obj).astype(jnp.int32)  # scalar
    optimal = obj[entering] <= eps  # scalar bool

    # Leaving variable: min-ratio test. Among rows where the entering column is
    # positive (so increasing the variable decreases that basic var), pick the
    # one that hits zero first. Non-positive entries impose no bound (+inf).
    col = tab[:m, entering]  # [m]
    rhs = tab[:m, -1]  # [m]
    ratios = jnp.where(col > eps, rhs / col, inf_val)  # [m]
    leaving = jnp.argmin(ratios).astype(jnp.int32)  # scalar
    pval = tab[leaving, entering]  # scalar; pivot element

    # Gauss-Jordan pivot: rank-1 update that zeros the entering column in every
    # row except the pivot row, then normalize the pivot row to unit pivot.
    prow = tab[leaving, :]  # [n+m+1]
    entering_col_vals = tab[:, entering]  # [m+1]
    new_tab = tab - (entering_col_vals / pval)[:, None] * prow[None, :]  # [m+1, n+m+1]
    is_pivot_row = jnp.arange(m + 1) == leaving  # [m+1]
    new_tab = jnp.where(is_pivot_row[:, None], (prow / pval)[None, :], new_tab)

    new_basis = basis.at[leaving].set(entering)  # [m]

    # Only commit the pivot if it both improves and is numerically safe;
    # otherwise leave the tableau untouched and terminate the loop.
    can_pivot = ~optimal & (pval > eps)
    return (
        jnp.where(can_pivot, new_tab, tab),
        jnp.where(can_pivot, new_basis, basis),
        ~can_pivot,
        step + 1,
    )

  init = (tab, basis, jnp.zeros([], dtype=jnp.bool_), jnp.zeros([], dtype=jnp.int32))
  tab, basis, _, num_steps = jax.lax.while_loop(cond_fun, body_fun, init)

  # Extract primal: a decision variable x[j] that is basic equals the RHS of its
  # row; a nonbasic x[j] is zero. `eq[r, j]` marks the row (if any) basic in j,
  # so summing the masked RHS over rows picks out that value (0 if nonbasic).
  eq = basis[:, None] == jnp.arange(n)[None, :]  # [m, n]
  x = jnp.sum(jnp.where(eq, tab[:m, -1:], jnp.zeros_like(tab[:m, -1:])), axis=0)  # [n]

  # Dual variables: at optimality the shadow price of constraint i is the
  # negated reduced cost of its slack s_i, which sits at tab[m, n+i].
  ineq_dual = -tab[m, n:n + m]  # [m]

  # Undo equilibration: x = D_c x', u = D_r u' (no-op when disabled, d=1).
  x = d_c * x
  ineq_dual = d_r * ineq_dual

  return x, ineq_dual, {'num_steps': num_steps}


def solve_optimization_simplex_with_extras(
    problem: ConstrainedOptimizationProblem[Parameters, Variables],
    parameters: Parameters,
    *,
    max_steps: int = 200,
    expected_dtype: tp.Optional[jnp.dtype] = None,
    **_,
) -> tuple[Variables, jax.Array, Stats]:
  """Solve a linear program using simplex, also returning inequality dual variables.

  Assumes:
  - The objective and constraint_violations are linear in the variables.
  - No equality constraints (equality_violations must return an empty array).
  - Variables are implicitly non-negative (x >= 0); do NOT include -x <= 0 in
    constraint_violations — simplex enforces non-negativity structurally.
  - The initial variables returned by problem.initial_variables() are feasible,
    i.e. constraint_violations(initial) <= 0, so h = -violations(initial) >= 0.
  """
  variables = problem.initial_variables(parameters)
  flatten, unflatten, _ = _setup_flatten(variables)
  x0 = flatten(variables)

  dtype = x0.dtype
  if expected_dtype is not None:
    assert dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {dtype}"

  def obj_fn(x: jax.Array) -> jax.Array:
    return problem.objective(parameters, unflatten(x))

  def constr_fn(x: jax.Array) -> jax.Array:
    return problem.constraint_violations(parameters, unflatten(x))

  # For a linear LP, the Jacobian is constant (independent of x0).
  c_min = jax.grad(obj_fn)(x0)       # gradient of the objective to minimize
  G = jax.jacrev(constr_fn)(x0)      # [m, n]
  # constr_fn(x) = G @ x - h  =>  h = G @ x0 - constr_fn(x0)
  h = jnp.matvec(G, x0) - constr_fn(x0)

  x_opt, ineq_dual, stats = solve_lp_simplex(-c_min, G, h, max_steps=max_steps, expected_dtype=dtype)
  return unflatten(x_opt), ineq_dual, stats


def solve_optimization_simplex(
    problem: ConstrainedOptimizationProblem[Parameters, Variables],
    parameters: Parameters,
    *,
    max_steps: int = 200,
    expected_dtype: tp.Optional[jnp.dtype] = None,
    **_,
) -> tuple[Variables, Stats]:
  """Solve a linear program defined by a ConstrainedOptimizationProblem using simplex."""
  variables, _, stats = solve_optimization_simplex_with_extras(
      problem, parameters, max_steps=max_steps, expected_dtype=expected_dtype)
  return variables, stats


solve_feasibility_simplex = as_feasibility_solver(solve_optimization_simplex)

simplex_static_argnames = ('max_steps', 'expected_dtype')


def solve_lp_linrax(
    c: jax.Array,
    G: jax.Array,
    h: jax.Array,
    *,
    primal_tol: float = 1e-6,
    dual_tol: float = 1e-6,
    expected_dtype: tp.Optional[jnp.dtype] = None,
) -> tuple[jax.Array, jax.Array, Stats]:
  """Solve LP: max c^T x s.t. G x <= h, x >= 0 using linrax.

  Returns (x_opt, ineq_dual, stats) where ineq_dual[i] is the dual variable
  for constraint i (shadow price of G[i,:] x <= h[i]).
  """
  m, n = G.shape
  if expected_dtype is not None:
    assert G.dtype == expected_dtype, f'Expected dtype {expected_dtype}, got {G.dtype}'
  sol, sol_type = linprog(-c, A_ub=G, b_ub=h, primal_tol=primal_tol, dual_tol=dual_tol)
  # After _standard_form, the final tableau cols are [primal (n) | slack (m) | RHS].
  # Reduced costs of slack variables equal the dual variables for G x <= h.
  ineq_dual = sol.tableau[-1, n:n + m]
  stats = dict(feasible=sol_type.feasible, bounded=sol_type.bounded)
  return sol.x, ineq_dual, stats


linrax_static_argnames = ('primal_tol', 'dual_tol', 'expected_dtype')


def solve_optimization_linrax_with_extras(
    problem: ConstrainedOptimizationProblem[Parameters, Variables],
    parameters: Parameters,
    *,
    primal_tol: float = 1e-6,
    dual_tol: float = 1e-6,
    expected_dtype: tp.Optional[jnp.dtype] = None,
    **_,
) -> tuple[Variables, jax.Array, Stats]:
  """Solve a linear program using linrax, also returning inequality dual variables.

  Assumes:
  - The objective and constraint_violations are linear in the variables.
  - No equality constraints.
  - Variables are implicitly non-negative (x >= 0).
  - The initial variables are feasible (constraint_violations <= 0).
  """
  variables = problem.initial_variables(parameters)
  flatten, unflatten, _ = _setup_flatten(variables)
  x0 = flatten(variables)

  dtype = x0.dtype
  if expected_dtype is not None:
    assert dtype == expected_dtype, f'Expected dtype {expected_dtype}, got {dtype}'

  def obj_fn(x: jax.Array) -> jax.Array:
    return problem.objective(parameters, unflatten(x))

  def constr_fn(x: jax.Array) -> jax.Array:
    return problem.constraint_violations(parameters, unflatten(x))

  c_min = jax.grad(obj_fn)(x0)
  G = jax.jacrev(constr_fn)(x0)
  h = jnp.matvec(G, x0) - constr_fn(x0)

  x_opt, ineq_dual, stats = solve_lp_linrax(
      -c_min, G, h,
      primal_tol=primal_tol, dual_tol=dual_tol,
      expected_dtype=dtype)
  return unflatten(x_opt), ineq_dual, stats


def jitted_simplex_feasibility_solver(
    problem: FeasibilityProblem[Parameters, Variables],
):
  solver = jax_utils.partial(solve_feasibility_simplex, problem)
  return jax_utils.jit(solver, static_argnames=simplex_static_argnames)


def vmap1_simplex_feasibility_solver(
    problem: FeasibilityProblem[Parameters, Variables],
):
  solver = jax_utils.partial(solve_feasibility_simplex, problem)
  return jax_utils.vmap1(solver, static_argnames=simplex_static_argnames)


def solve_lp_mpax(
    c: jax.Array,
    G: jax.Array,
    h: jax.Array,
    *,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-4,
    expected_dtype: tp.Optional[jnp.dtype] = None,
) -> tuple[jax.Array, jax.Array, Stats]:
  """Solve LP: max c^T x s.t. G x <= h, x >= 0 using mpax's r2HPDHG.

  Returns (x_opt, ineq_dual, stats) where ineq_dual[i] is the dual variable
  for constraint i (shadow price of G[i,:] x <= h[i]).
  """
  from mpax.r2hpdhg import r2HPDHG
  from mpax.mp_io import create_lp

  m, n = G.shape
  if expected_dtype is not None:
    assert G.dtype == expected_dtype, f'Expected dtype {expected_dtype}, got {G.dtype}'

  # mpax minimizes c^T x s.t. G x >= h, l <= x <= u.
  # Convert: max c^T x s.t. G x <= h, x >= 0
  #       => min (-c)^T x s.t. (-G) x >= (-h), 0 <= x <= inf
  A_eq = jnp.zeros([0, n], dtype=G.dtype)
  b_eq = jnp.zeros([0], dtype=G.dtype)
  l = jnp.zeros([n], dtype=G.dtype)
  u = jnp.full([n], jnp.inf, dtype=G.dtype)

  lp = create_lp(-c, A_eq, b_eq, -G, -h, l, u, use_sparse_matrix=False)
  solver = r2HPDHG(eps_abs=eps_abs, eps_rel=eps_rel)
  output = solver.optimize(lp)

  # dual_solution[i] equals the KKT dual variable for (-G[i,:]) x >= -h[i],
  # which by complementarity equals the shadow price for G[i,:] x <= h[i].
  stats = dict(
      termination_status=output.termination_status,
      num_steps=output.iteration_count,
  )
  return output.primal_solution, output.dual_solution, stats


mpax_static_argnames = ('eps_abs', 'eps_rel', 'expected_dtype')


def solve_optimization_mpax_with_extras(
    problem: ConstrainedOptimizationProblem[Parameters, Variables],
    parameters: Parameters,
    *,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-4,
    expected_dtype: tp.Optional[jnp.dtype] = None,
    **_,
) -> tuple[Variables, jax.Array, Stats]:
  """Solve a linear program using mpax's r2HPDHG, also returning inequality dual variables.

  Assumes:
  - The objective and constraint_violations are linear in the variables.
  - No equality constraints (equality_violations must return an empty array).
  - Variables are implicitly non-negative (x >= 0).
  - The initial variables are feasible (constraint_violations <= 0).
  """
  variables = problem.initial_variables(parameters)
  flatten, unflatten, _ = _setup_flatten(variables)
  x0 = flatten(variables)

  dtype = x0.dtype
  if expected_dtype is not None:
    assert dtype == expected_dtype, f'Expected dtype {expected_dtype}, got {dtype}'

  def obj_fn(x: jax.Array) -> jax.Array:
    return problem.objective(parameters, unflatten(x))

  def constr_fn(x: jax.Array) -> jax.Array:
    return problem.constraint_violations(parameters, unflatten(x))

  c_min = jax.grad(obj_fn)(x0)
  G = jax.jacrev(constr_fn)(x0)
  h = jnp.matvec(G, x0) - constr_fn(x0)

  x_opt, ineq_dual, stats = solve_lp_mpax(
      -c_min, G, h,
      eps_abs=eps_abs, eps_rel=eps_rel,
      expected_dtype=dtype)
  return unflatten(x_opt), ineq_dual, stats
