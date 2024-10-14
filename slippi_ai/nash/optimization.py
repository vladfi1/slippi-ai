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

# @tf.function()
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

  if directional_derivative is None:
    with tf.autodiff.ForwardAccumulator(variables, direction) as acc:
      objective_value = objective(variables)
    directional_derivative = acc.jvp(objective_value)
  else:
    objective_value = objective(variables)

  # TODO: assert that directional_derivative is negative

  step_size = tf.convert_to_tensor(initial_step_size, dtype=variables.dtype)
  # print('Initial step size', step_size)

  def not_valid(step_size):
    return tf.logical_not(condition(variables + step_size * direction))

  decrease_step_size = lambda x: x * beta

  # while not_valid(step_size):
  #   step_size = decrease_step_size(step_size)

  [step_size] = tf.while_loop(
      cond=not_valid, body=decrease_step_size, loop_vars=[step_size])

  def not_good_enough(step_size):
    new_objective = objective(variables + step_size * direction)
    decrease = new_objective - objective_value
    expected_decrease = step_size * directional_derivative
    return decrease > alpha * expected_decrease

  # while not_valid(step_size):
  #   step_size = decrease_step_size(step_size)

  [step_size] = tf.while_loop(
      cond=not_good_enough, body=decrease_step_size, loop_vars=[step_size])

  return variables + step_size * direction

# https://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/primal-dual.pdf
def solve_optimization_interior_point_barrier(
    problem: ConstrainedOptimizationProblem[Variables],
    error: float = 1e-2,
    initial_constraint_weight: float = 1.0,
    constraint_weight_decay: float = 0.9,
) -> Variables:
  variables = problem.initial_variables()
  flat_vars: list[tf.Tensor] = tf.nest.flatten(variables)
  flat_shapes = [t.shape for t in flat_vars]
  flat_sizes = [s.num_elements() for s in flat_shapes]
  flat_size = sum(flat_sizes)

  def flatten(variables: Variables) -> tf.Tensor:
    return tf.concat([tf.reshape(v, [-1]) for v in tf.nest.flatten(variables)], axis=-1)

  def unflatten(flat_var: tf.Tensor) -> Variables:
    split = tf.split(flat_var, flat_sizes, axis=-1)
    reshaped = [tf.reshape(v, s) for v, s in zip(split, flat_shapes)]
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
    # variables = unflatten(combined[:flat_size])
    # equality_vars = combined[flat_size:]
    flat_vars, equality_vars = tf.split(combined, [flat_size, num_equalities], axis=-1)
    variables = unflatten(flat_vars)
    return lagrangian(variables, equality_vars, mu)

  def residual(combined: tf.Tensor, mu: tf.Tensor) -> tf.Tensor:
    variables, equality_vars = uncombine(combined)
    with tf.GradientTape() as tape:
      tape.watch([variables, equality_vars])
      L = lagrangian(variables, equality_vars, mu)
    grads = tape.gradient(L, (variables, equality_vars))
    return combine(*grads)

  def is_valid(combined: tf.Tensor):
    # variables = unflatten(combined[:flat_size])
    variables, equality_vars = uncombine(combined)
    return tf.reduce_all(problem.constraint_violations(variables) < 0, axis=-1)

  @tf.function(jit_compile=False)
  def newton_step(combined: tf.Tensor, mu: tf.Tensor) -> tf.Tensor:
    with tf.GradientTape() as tape:
      tape.watch(combined)
      residual_value = residual(combined, mu)
    hessian = tape.jacobian(residual_value, combined)

    target = -tf.expand_dims(residual_value, axis=-1)
    delta = tf.linalg.solve(hessian, target)
    delta = tf.squeeze(delta, axis=-1)

    def residual_objective(combined: tf.Tensor):
      residual_value = residual(combined, mu)
      return 0.5 * tf.reduce_sum(tf.square(residual_value), axis=-1)

    # grad(residual_objective) = residual * grad(residual)
    # => grad(residual_objective) * delta = |residual|^2

    new_combined = line_search(
        objective=residual_objective,
        condition=is_valid,
        variables=combined,
        direction=delta,
        # directional_derivative=tf.reduce_sum(tf.square(residual_value), axis=-1),
    )

    tf.Assert(is_valid(new_combined), [new_combined])

    # Expected improvement in the Lagrangian
    # expected_improvement = 0.5 * tf.reduce_sum(residual_value * delta, axis=-1)

    return new_combined, residual_value

  @tf.function(jit_compile=False)
  def centering_step(
      combined: tf.Tensor,
      mu: tf.Tensor,
      tolerance: tf.Tensor,
  ) -> tf.Tensor:
    combined, residual_value = newton_step(combined, mu)

    def is_done(residual_value) -> bool:
      residual_magnitude = tf.sqrt(tf.reduce_sum(tf.square(residual_value), axis=-1))
      return residual_magnitude < tolerance

    num_steps = 1
    while not is_done(residual_value):
      combined, residual_value = newton_step(combined, mu)
      num_steps += 1

    return combined, num_steps

  flat_var = tf.concat([tf.reshape(v, [-1]) for v in flat_vars], axis=-1)
  equality_vars = tf.zeros([num_equalities], dtype=flat_var.dtype)
  combined = tf.concat([flat_var, equality_vars], axis=-1)

  mu = tf.convert_to_tensor(initial_constraint_weight, dtype=flat_var.dtype)

  def step(combined: tf.Tensor, mu: tf.Tensor):
    combined, num_steps = centering_step(combined, mu, error / 2)

    # assert is_valid(combined)
    # residual_value = residual(combined, mu)
    # residual_magnitude = tf.sqrt(tf.reduce_sum(tf.square(residual_value), axis=-1))

    duality_gap = num_constraints * mu
    done = duality_gap < error / 2

    return combined, num_steps, done

  done = tf.convert_to_tensor(False)

  num_steps = 0
  centering_steps = 0
  while not done:
    combined, num_centering_steps, done = step(combined, mu)
    mu *= constraint_weight_decay
    num_steps += 1
    # centering_steps.append(num_centering_steps)
    # centering_steps.append(num_centering_steps.numpy())
    centering_steps += num_centering_steps

  # print('Number of steps:', num_steps)
  # print('Centering steps:', centering_steps)

  stats = dict(
      num_steps=num_steps,
      centering_steps=centering_steps,
  )

  return unflatten(combined[:flat_size])

# This should actually be a "forall" type but I don't think python has those.
V = tp.TypeVar('V')
Solver = tp.Callable[[ConstrainedOptimizationProblem[V]], V]

def solve_feasibility(
    problem: FeasibilityProblem[Variables],
    optimization_solver: Solver[SlackVariables[Variables]] = solve_optimization_interior_point_barrier,
    **solver_kwargs,
) -> Variables:
  slack_problem = SlackFeasibilityProblem(problem)
  slack_variables = optimization_solver(slack_problem, **solver_kwargs)
  return slack_variables.variables
