import abc
import typing as tp

import attr
import numpy as np
import tensorflow as tf

Variables = tp.TypeVar('Variables')  # add bound on tree.Structure?

class FeasibilityProblem(abc.ABC, tp.Generic[Variables]):

  @abc.abstractmethod
  def initial_variables(self) -> Variables:
    """Returns initial variables for the feasibility problem."""

  @abc.abstractmethod
  def batch_size(self) -> int:
    """Returns the batch size of the problem."""

  @abc.abstractmethod
  def constraint_violations(self, variables: Variables) -> tf.Tensor:
    """Returns a tensor of violations of the constraints.

    The Tensor has shape [..., num_constraints] where num_constraints is the
    number of constraints in the problem. Positive values indicate violations,
    while negative values indicate that the constraint is satisfied.
    """

  @abc.abstractmethod
  def equality_violations(self, variables: Variables) -> tf.Tensor:
    """Returns a tensor of violations of equality constraints.

    The Tensor has shape [..., num_constraints] where num_constraints is the
    number of equality constraints in the problem. Positive or negative values
    indicate violations, while zero indicates that the constraint is satisfied.
    """

class ConstrainedOptimizationProblem(FeasibilityProblem[Variables]):
  """An optimization problem with an objective function."""

  @abc.abstractmethod
  def initial_variables(self) -> Variables:
    """Returns initial variables which satisfy the constraints."""

  @abc.abstractmethod
  def objective(self, variables: Variables) -> tf.Tensor:
    """Returns the objective function to minimize."""

# Unlike dataclasses, attrs is recognized by tf.nest as a structure.
@attr.s(auto_attribs=True)
class SlackVariables(tp.Generic[Variables]):
  """Variables for a slack feasibility problem.

  Attributes:
    variables: Variables for the original problem.
    slack: Slack variables for the constraints.
  """

  variables: Variables = attr.ib()
  slack: tf.Tensor = attr.ib()


class SlackFeasibilityProblem(ConstrainedOptimizationProblem[SlackVariables[Variables]]):

  def __init__(
      self,
      problem: FeasibilityProblem[Variables],
      initial_slack: float = 0.1,
  ):
    self.problem = problem
    self.initial_slack = initial_slack

  def batch_size(self) -> int:
    return self.problem.batch_size()

  def initial_variables(self) -> SlackVariables[Variables]:
    variables = self.problem.initial_variables()
    violations = self.problem.constraint_violations(variables)
    slack = tf.reduce_max(violations, axis=-1) + self.initial_slack
    return SlackVariables(variables=variables, slack=slack)

  def constraint_violations(self, variables: SlackVariables[Variables]) -> tf.Tensor:
    slack = tf.expand_dims(variables.slack, axis=-1)
    return self.problem.constraint_violations(variables.variables) - slack

  def equality_violations(self, variables: SlackVariables[Variables]) -> tf.Tensor:
    return self.problem.equality_violations(variables.variables)

  def objective(self, variables: SlackVariables[Variables]) -> tf.Tensor:
    return variables.slack

def jvp_bwd(
    f: tp.Callable[[tf.Tensor], tf.Tensor],
    x: tf.Tensor,
    dx: tf.Tensor,
) -> tf.Tensor:
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = f(x)
    z = tf.zeros_like(y)
    tape.watch(z)
    w = tf.reduce_sum(y * z)

    dw_dx = tape.gradient(w, x)
    dd = tape.gradient(dw_dx, z, output_gradients=dx)
    return y, dd

def jvp_fwd(
    f: tp.Callable[[tf.Tensor], tf.Tensor],
    x: tf.Tensor,
    dx: tf.Tensor,
) -> tf.Tensor:
  with tf.autodiff.ForwardAccumulator(x, dx) as acc:
    y = f(x)
  return y, acc.jvp(y)

def mat_inv(A: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
  return tf.squeeze(tf.linalg.solve(A, tf.expand_dims(b, -1)), -1)

def line_search(
    objective: tp.Callable[[tf.Tensor], tf.Tensor],
    condition: tp.Callable[[tf.Tensor], tf.Tensor],
    variables: tf.Tensor,
    direction: tf.Tensor,
    directional_derivative: tp.Optional[tf.Tensor] = None,
    initial_step_size: float = 1.0,
    alpha: float = 0.1,  # How much reduction in objective we want.
    beta: float = 0.5,  # How much to decrease step size by.
) -> tf.Tensor:

  alpha = tf.convert_to_tensor(alpha, dtype=variables.dtype)
  beta = tf.convert_to_tensor(beta, dtype=variables.dtype)
  one = tf.constant(1, dtype=variables.dtype)

  # Sadly both jvp_fwd and jvp_bwd don't work with jit_compile=True.
  # A workaround is to pass the directional_derivative if it is known.
  if directional_derivative is None:
    objective_value, directional_derivative = jvp_fwd(
        objective, variables, direction)
  else:
    objective_value = objective(variables)

  # TODO: assert that directional_derivative is negative
  batch_size = variables.shape[0]
  assert direction.shape[0] == batch_size

  step_size = tf.convert_to_tensor(initial_step_size, dtype=variables.dtype)
  step_size = tf.broadcast_to(step_size, [batch_size])

  def take_step(step_size):
    return variables + tf.expand_dims(step_size, -1) * direction

  def update_valid(step_size):
    valid = condition(take_step(step_size))
    step_size *= tf.where(valid, one, beta)
    return step_size, tf.reduce_all(valid)

  valid = tf.convert_to_tensor(False)
  while tf.logical_not(valid):
    step_size, valid = update_valid(step_size)

  # [step_size] = tf.while_loop(
  #     cond=not_valid, body=decrease_step_size, loop_vars=[step_size])

  def update_good_enough(step_size):
    new_objective = objective(take_step(step_size))
    decrease = new_objective - objective_value
    expected_decrease = step_size * directional_derivative
    good_enough = decrease <= alpha * expected_decrease
    step_size *= tf.where(good_enough, one, beta)
    return step_size, tf.reduce_all(good_enough)

  good_enough = tf.convert_to_tensor(False)
  while tf.logical_not(good_enough):
    step_size, good_enough = update_good_enough(step_size)

  # while not_valid(step_size):
  #   step_size = decrease_step_size(step_size)

  # [step_size] = tf.while_loop(
  #     cond=not_good_enough, body=decrease_step_size, loop_vars=[step_size])

  return take_step(step_size)

# https://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/primal-dual.pdf
def solve_optimization_interior_point_barrier(
    problem: ConstrainedOptimizationProblem[Variables],
    error: float = 1e-2,
    initial_constraint_weight: float = 1.0,
    constraint_weight_decay: float = 0.9,
    optimum: tp.Optional[float] = None,
    jit_compile: bool = False,
) -> Variables:
  batch_size = problem.batch_size()
  variables = problem.initial_variables()
  flat_vars: list[tf.Tensor] = tf.nest.flatten(variables)
  flat_shapes = [t.shape[1:] for t in flat_vars]
  flat_sizes = [s.num_elements() for s in flat_shapes]
  flat_size = sum(flat_sizes)

  def flatten(variables: Variables) -> tf.Tensor:
    return tf.concat([
        tf.reshape(v, [batch_size, -1])
        for v in tf.nest.flatten(variables)], axis=-1)

  def unflatten(flat_var: tf.Tensor) -> Variables:
    split = tf.split(flat_var, flat_sizes, axis=-1)
    reshaped = [
        tf.reshape(flat, v.shape)
        for flat, v in zip(split, flat_vars)]
    return tf.nest.pack_sequence_as(variables, reshaped)

  num_constraints = problem.constraint_violations(variables).shape[-1]
  num_equalities = problem.equality_violations(variables).shape[-1]

  def combine(variables: Variables, equality_vars: tf.Tensor) -> tf.Tensor:
    flat_vars = flatten(variables)
    return tf.concat([flat_vars, equality_vars], axis=-1)

  def uncombine(combined: tf.Tensor) -> tp.Tuple[Variables, tf.Tensor]:
    flat_vars, equality_vars = tf.split(combined, [flat_size, num_equalities], axis=-1)
    return unflatten(flat_vars), equality_vars

  def lagrangian(
      variables: Variables,
      equality_variables: tf.Tensor,
      mu: tf.Tensor,
  ) -> tf.Tensor:
    objective = problem.objective(variables)
    constraints = problem.constraint_violations(variables)
    equalities = problem.equality_violations(variables)

    return (
        objective
        - mu * tf.reduce_sum(tf.math.log(-constraints), axis=-1)
        + tf.reduce_sum(equality_variables * equalities, axis=-1)
    )

  def lagrangian_combined(combined: tf.Tensor, mu: tf.Tensor) -> tf.Tensor:
    variables, equality_vars = uncombine(combined)
    return lagrangian(variables, equality_vars, mu)

  def residual(combined: tf.Tensor, mu: tf.Tensor) -> tf.Tensor:
    variables, equality_vars = uncombine(combined)
    with tf.GradientTape() as tape:
      tape.watch([variables, equality_vars])
      L = lagrangian(variables, equality_vars, mu)
    grads = tape.gradient(L, (variables, equality_vars))
    return combine(*grads)

  def residual_combined(combined: tf.Tensor, mu: tf.Tensor) -> tf.Tensor:
    with tf.GradientTape() as tape:
      tape.watch(combined)
      L = lagrangian_combined(combined, mu)
    return tape.gradient(L, combined)

  def is_valid(combined: tf.Tensor):
    # variables = unflatten(combined[:flat_size])
    variables, _ = uncombine(combined)
    return tf.reduce_all(problem.constraint_violations(variables) < 0, axis=-1)

  @tf.function(jit_compile=jit_compile)
  def newton_step(combined: tf.Tensor, mu: tf.Tensor) -> tf.Tensor:
    with tf.GradientTape() as tape:
      tape.watch(combined)
      residual_value = residual(combined, mu)
    hessian = tape.batch_jacobian(residual_value, combined)

    delta = mat_inv(hessian, -residual_value)

    def residual_objective(combined: tf.Tensor):
      residual_value = residual(combined, mu)
      return 0.5 * tf.reduce_sum(tf.square(residual_value), axis=-1)

    # grad(residual_objective) = residual * grad(residual)
    # => grad(residual_objective) * delta
    # = residual * grad(residual) * delta
    # = residual * -residual = - |residual|^2

    new_combined = line_search(
        objective=residual_objective,
        condition=is_valid,
        variables=combined,
        direction=delta,
        directional_derivative=-tf.reduce_sum(tf.square(residual_value), axis=-1),
    )

    # tf.Assert(is_valid(new_combined), [new_combined])

    # Expected improvement in the Lagrangian
    # expected_improvement = 0.5 * tf.reduce_sum(residual_value * delta, axis=-1)

    return new_combined, residual_value

  @tf.function(jit_compile=jit_compile)
  def centering_step(
      combined: tf.Tensor,
      mu: tf.Tensor,
      tolerance: tf.Tensor,
      optimum: tp.Optional[tf.Tensor] = None,
  ) -> tuple[tf.Tensor, tf.Tensor]:
    assert combined.shape[0] == batch_size
    assert mu.shape[0] == batch_size
    # assert tolerance.shape[0] == batch_size

    def is_done_residual(residual_value) -> bool:
      residual_magnitude = tf.sqrt(tf.reduce_sum(tf.square(residual_value), axis=-1))
      return residual_magnitude < tolerance

    # TODO: check done-ness before starting the loop?
    num_steps = tf.fill([batch_size], 1)

    any_not_done = tf.constant(True)

    while any_not_done:
      combined, residual_value = newton_step(combined, mu)

      if optimum is not None:
        objective_value = problem.objective(uncombine(combined)[0])
        done = objective_value < optimum + tolerance
      else:
        done = is_done_residual(residual_value)

      not_done = tf.logical_not(done)
      assert not_done.shape == [batch_size]
      num_steps += tf.cast(not_done, num_steps.dtype)
      any_not_done = tf.reduce_any(not_done)

    return combined, num_steps

  flat_var = flatten(variables)
  equality_vars = tf.zeros([batch_size, num_equalities], dtype=flat_var.dtype)
  combined = tf.concat([flat_var, equality_vars], axis=-1)

  mu = tf.convert_to_tensor(initial_constraint_weight, dtype=flat_var.dtype)
  mu = tf.fill([batch_size], mu)
  constraint_weight_decay = tf.convert_to_tensor(
      constraint_weight_decay, dtype=flat_var.dtype)
  one = tf.constant(1, dtype=flat_var.dtype)

  def outer_step(combined: tf.Tensor, mu: tf.Tensor):
    duality_gap = num_constraints * mu
    if optimum is not None:
      perturbed_optimum = optimum + duality_gap
    else:
      perturbed_optimum = None

    combined, num_steps = centering_step(
        combined, mu, tolerance=error / 2, optimum=perturbed_optimum)

    if optimum is not None:
      # Check if we are close to the optimum.
      variables, _ = uncombine(combined)
      objective_value = problem.objective(variables)
      done = objective_value < optimum + error
    else:
      done = duality_gap < error / 2

    return combined, num_steps, done

  all_done = tf.convert_to_tensor(False)

  num_steps = tf.fill([batch_size], 1)
  centering_steps = tf.fill([batch_size], 0)

  # def body(combined, mu, _):
  #   combined, num_centering_steps, done = outer_step(combined, mu)
  #   # num_steps += tf.cast(tf.logical_not(done), num_steps.dtype)
  #   mu *= tf.where(done, one, constraint_weight_decay)
  #   # centering_steps += num_centering_steps
  #   all_done = tf.reduce_all(done)
  #   return combined, mu, all_done

  # cond = lambda combined, mu, all_done: tf.logical_not(all_done)

  # combined, mu, all_done = tf.while_loop(
  #     cond=cond, body=body, loop_vars=[combined, mu, all_done])

  while tf.logical_not(all_done):
    combined, num_centering_steps, done = outer_step(combined, mu)
    num_steps += tf.cast(tf.logical_not(done), num_steps.dtype)
    mu *= tf.where(done, one, constraint_weight_decay)
    # centering_steps.append(num_centering_steps)
    # centering_steps.append(num_centering_steps.numpy())
    centering_steps += num_centering_steps
    all_done = tf.reduce_all(done)

  stats = dict(
      num_steps=num_steps,
      centering_steps=centering_steps,
  )

  return uncombine(combined)[0], stats


def dot(x: tf.Tensor, y: tf.Tensor, axis=-1) -> tf.Tensor:
  return tf.reduce_sum(x * y, axis=axis)

# https://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/primal-dual.pdf
def solve_optimization_interior_point_primal_dual(
    problem: ConstrainedOptimizationProblem[Variables],
    error: float = 1e-2,
    initial_constraint_weight: float = 1.0,
    constraint_weight_decay: float = 0.9,
    optimum: tp.Optional[float] = None,
    jit_compile: bool = False,
    is_linear: bool = False,
    cholesky: bool = False,
) -> tuple[Variables, dict]:
  batch_size = problem.batch_size()
  variables = problem.initial_variables()
  flat_vars: list[tf.Tensor] = tf.nest.flatten(variables)
  flat_shapes = [t.shape[1:] for t in flat_vars]
  flat_sizes = [s.num_elements() for s in flat_shapes]
  flat_size = sum(flat_sizes)

  def flatten(variables: Variables) -> tf.Tensor:
    return tf.concat([
        tf.reshape(v, [batch_size, -1])
        for v in tf.nest.flatten(variables)], axis=-1)

  def unflatten(flat_var: tf.Tensor) -> Variables:
    split = tf.split(flat_var, flat_sizes, axis=-1)
    reshaped = [
        tf.reshape(flat, v.shape)
        for flat, v in zip(split, flat_vars)]
    return tf.nest.pack_sequence_as(variables, reshaped)

  flat_var = flatten(variables)
  dtype = flat_var.dtype

  initial_constraints = problem.constraint_violations(variables)
  num_constraints = initial_constraints.shape[-1]
  num_equalities = problem.equality_violations(variables).shape[-1]

  def split(combined: tf.Tensor) -> tp.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return tf.split(combined, [flat_size, num_constraints, num_equalities], axis=-1)

  def combine(variables: Variables, constraint_vars: tf.Tensor, equality_vars: tf.Tensor) -> tf.Tensor:
    flat_vars = flatten(variables)
    return tf.concat([flat_vars, constraint_vars, equality_vars], axis=-1)

  def uncombine(combined: tf.Tensor) -> tp.Tuple[Variables, tf.Tensor, tf.Tensor]:
    flat_vars, constraint_vars, equality_vars = split(combined)
    return unflatten(flat_vars), constraint_vars, equality_vars

  def is_valid(combined: tf.Tensor):
    # variables = unflatten(combined[:flat_size])
    variables = uncombine(combined)[0]
    return tf.reduce_all(problem.constraint_violations(variables) < 0, axis=-1)

  # Equality constraints are linear and so can be precomputed
  x = flat_var
  with tf.GradientTape() as tape:
    tape.watch(x)
    x_struct = unflatten(x)
    eq = problem.equality_violations(x_struct)
  A = tape.batch_jacobian(eq, x)  # [B, K, N]
  A_t = tf.transpose(A, perm=[0, 2, 1])  # [B, N, K]

  # If the constraints and objective are linear, we can precompute the gradients.
  if is_linear:
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(x)
      x_struct = unflatten(x)
      f = problem.objective(x_struct)
      f = tf.expand_dims(f, -1)  # [B, 1]
      g = problem.constraint_violations(x_struct)

    grad_f_linear = tf.squeeze(tape.batch_jacobian(f, x), -2)  # [B, N]
    grad_g_linear = tape.batch_jacobian(g, x)  # [B, M, N]
    grad_g_t_linear = tf.transpose(grad_g_linear, perm=[0, 2, 1])  # [B, N, M]

    def residuals(combined: tf.Tensor, epsilon: tf.Tensor) -> tp.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
      x, u, v = split(combined)
      x_struct = unflatten(x)
      g = problem.constraint_violations(x_struct)
      r_dual = grad_f_linear + tf.linalg.matvec(grad_g_t_linear, u) + tf.linalg.matvec(A_t, v)
      r_cent = u * g + tf.expand_dims(epsilon, -1)
      r_prim = problem.equality_violations(x_struct)
      return r_dual, r_cent, r_prim

  else:
    def residuals(combined: tf.Tensor, epsilon: tf.Tensor) -> tp.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
      flat_vars, constraint_variables, equality_variables = split(combined)
      with tf.GradientTape() as tape:
        tape.watch(flat_vars)
        variables = unflatten(flat_vars)
        objective = problem.objective(variables)
        constraints = problem.constraint_violations(variables)
        equalities = problem.equality_violations(variables)

        L = (
            objective
            + dot(constraint_variables, constraints)
            + dot(equality_variables, equalities)
        )

      r_dual = tape.gradient(L, flat_vars)
      r_cent = constraints * constraint_variables + tf.expand_dims(epsilon, -1)
      r_prim = equalities

      return r_dual, r_cent, r_prim

  def residual(combined: tf.Tensor, epsilon: tf.Tensor) -> tf.Tensor:
    return tf.concat(residuals(combined, epsilon), axis=-1)

  @tf.function(jit_compile=jit_compile)
  def newton_step(combined: tf.Tensor, epsilon: tf.Tensor) -> tf.Tensor:
    x, u, v = split(combined)
    N = flat_size
    M = num_constraints
    K = num_equalities

    with tf.GradientTape(persistent=True) as tape:
      tape.watch(x)
      x_struct = unflatten(x)
      f = problem.objective(x_struct)
      g = problem.constraint_violations(x_struct)
      eq = problem.equality_violations(x_struct)

      if not is_linear:
        L = f + dot(u, g) + dot(v, eq)
        r_dual = tape.gradient(L, x)

    if is_linear:
      grad_g = grad_g_linear
      grad_g_t = grad_g_t_linear
      r_dual = grad_f_linear + tf.linalg.matvec(grad_g_t, u) + tf.linalg.matvec(A_t, v)
      H = tf.zeros([batch_size, N, N], dtype)
    else:
      # The eq term vanishes as it should be linear.
      H = tape.batch_jacobian(r_dual, x)  # [B, N, N]

      grad_g = tape.batch_jacobian(g, x)  # [B, M, N]
      grad_g_t = tf.transpose(grad_g, perm=[0, 2, 1])  # [B, N, M]

    r_cent = u * g + tf.expand_dims(epsilon, -1)
    r_prim = eq
    residual_value = tf.concat([r_dual, r_cent, r_prim], axis=-1)

    # The goal of this Newton step is to set residual_value to zero.
    # We could just solve for this directly, but it is more efficient to
    # first elimate delta_u as it makes the linear system smaller.
    # TODO: show the math. See linked slides, pages 9-10 for details.

    diag_u = tf.expand_dims(u, -1)  # [B, M, 1]
    diag_g = tf.expand_dims(g, -1)  # [B, M, 1]

    J_xx = H - tf.matmul(grad_g_t, (diag_u / diag_g) * grad_g)  # [B, N, N]
    J_xv = A_t  # [B, N, K]
    J_x = tf.concat([J_xx, J_xv], axis=-1)  # [B, N, N + K]

    target_x = -r_dual + tf.linalg.matvec(grad_g_t, r_cent / g)

    J_vx = A  # [B, K, N]
    J_vv = tf.zeros([batch_size, K, K], dtype)
    J_v = tf.concat([J_vx, J_vv], axis=-1)  # [B, K, N + K]
    target_v = -r_prim

    if cholesky:
      # TODO: this is actually slightly slower than the non-cholesky version.
      chol_b = tf.linalg.cholesky(J_xx)
      b_inv_a = tf.linalg.cholesky_solve(chol_b, A_t)  # [B, N, K]
      a_b_inv_a = tf.linalg.matmul(A, b_inv_a)  # [B, K, K]
      chol_a_b_inv_a = tf.linalg.cholesky(a_b_inv_a)

      chol_inv = lambda M, z: tf.squeeze(
          tf.linalg.cholesky_solve(M, tf.expand_dims(z, -1)), -1)
      b_inv_target_x = chol_inv(chol_b, target_x)

      new_target_v = tf.linalg.matvec(A, b_inv_target_x) - target_v
      delta_v = chol_inv(chol_a_b_inv_a, new_target_v)
      # delta_v = mat_inv(a_b_inv_a, new_target_v)
      delta_x = b_inv_target_x - tf.linalg.matvec(b_inv_a, delta_v)
    else:
      J = tf.concat([J_x, J_v], axis=-2)
      target = tf.concat([target_x, target_v], axis=-1)
      delta_xv = mat_inv(J, target)
      delta_x, delta_v = tf.split(delta_xv, [N, K], axis=-1)

    delta_u = (-r_cent - u * tf.linalg.matvec(grad_g, delta_x)) / g
    delta = tf.concat([delta_x, delta_u, delta_v], axis=-1)

    # grad(residual) * delta = -residual

    # Make sure constraint vars remain positive.
    max_step_sizes = tf.where(delta_u < 0, -u / delta_u, 1.0)
    max_step_size = tf.reduce_min(max_step_sizes, axis=-1)
    max_step_size = tf.minimum(max_step_size, 1.0) * 0.99
    assert max_step_size.shape == [batch_size]

    def residual_objective(combined: tf.Tensor):
      residual_value = residual(combined, epsilon)
      return 0.5 * tf.reduce_sum(tf.square(residual_value), axis=-1)

    # grad(residual_objective) = residual * grad(residual)
    # => grad(residual_objective) * delta
    # = residual * grad(residual) * delta
    # = residual * -residual = - |residual|^2

    new_combined = line_search(
        objective=residual_objective,
        condition=is_valid,
        variables=combined,
        direction=delta,
        directional_derivative=-tf.reduce_sum(tf.square(residual_value), axis=-1),
        initial_step_size=max_step_size,
    )

    return new_combined, residual_value

  constraint_weight_decay = tf.convert_to_tensor(
      constraint_weight_decay, dtype=dtype)
  u = tf.convert_to_tensor(initial_constraint_weight, dtype=dtype)
  constraint_vars = tf.fill([batch_size, num_constraints], u)
  equality_vars = tf.zeros([batch_size, num_equalities], dtype=dtype)
  combined = tf.concat([flat_var, constraint_vars, equality_vars], axis=-1)

  eta = - dot(constraint_vars, initial_constraints)

  num_steps = tf.fill([batch_size], 1)
  done = tf.fill([batch_size], False)

  while tf.logical_not(tf.reduce_all(done)):
    # print('eta', eta.numpy())
    # TODO: don't take steps for done ones
    epsilon = constraint_weight_decay * eta / num_constraints
    combined, residual_value = newton_step(combined, epsilon)

    variables, constraint_vars, equality_vars = uncombine(combined)
    constraints = problem.constraint_violations(variables)
    eta = - dot(constraints, constraint_vars)

    # print(tf.nest.map_structure(lambda x: x.numpy(), variables))
    # print(combined.numpy())

    if optimum is not None:
      objective_value = problem.objective(variables)
      done = objective_value <= optimum + error
    else:
      r_dual, _, r_prim = split(residual_value)

      feasible = tf.sqrt(
          tf.reduce_sum(tf.square(r_dual), axis=-1)
          + tf.reduce_sum(tf.square(r_prim), axis=-1)) <= error

      done = tf.logical_and(eta <= error, feasible)

    num_steps += tf.cast(tf.logical_not(done), num_steps.dtype)

  stats = dict(
      num_steps=num_steps,
  )

  return variables, stats


# This should actually be a "forall" type but I don't think python has those.
V = tp.TypeVar('V')
Stats = dict
Solver = tp.Callable[[ConstrainedOptimizationProblem[V]], tuple[V, Stats]]

def solve_feasibility(
    problem: FeasibilityProblem[Variables],
    optimization_solver: Solver[SlackVariables[Variables]] = solve_optimization_interior_point_barrier,
    **solver_kwargs,
) -> tuple[Variables, Stats]:
  slack_problem = SlackFeasibilityProblem(problem)
  variables, stats = optimization_solver(slack_problem, **solver_kwargs)
  stats['slack'] = variables.slack
  return variables.variables, stats
