import typing as tp

import numpy as np
import tensorflow as tf

from slippi_ai.nash import optimization

class NashLogits(tp.NamedTuple):
  p1_logits: tf.Tensor
  p2_logits: tf.Tensor
  p1_nash_value: tf.Tensor

  def to_nash_variables(self) -> 'NashVariables':
    return NashVariables(
        p1=tf.nn.softmax(self.p1_logits),
        p2=tf.nn.softmax(self.p2_logits),
        p1_nash_value=self.p1_nash_value,
    )

class ZeroSumNashProblemWithLogits(optimization.FeasibilityProblem[NashLogits]):

  def __init__(self, payoff_matrix: tf.Tensor):
    self.payoff_matrix = tf.convert_to_tensor(payoff_matrix)

  def initial_variables(self) -> NashLogits:
    d1, d2 = self.payoff_matrix.shape.as_list()
    dtype = self.payoff_matrix.dtype

    return NashLogits(
        p1_logits=tf.zeros([d1], dtype=dtype),
        p2_logits=tf.zeros([d2], dtype=dtype),
        p1_nash_value=tf.constant(0.0, dtype=dtype),
    )

  def constraint_violations(self, variables: NashLogits) -> tf.Tensor:
    p1 = tf.expand_dims(tf.nn.softmax(variables.p1_logits), 0)
    p2 = tf.expand_dims(tf.nn.softmax(variables.p2_logits), 1)

    # No strategy for p1 does better than the nash value
    p1_payoffs = tf.squeeze(tf.matmul(self.payoff_matrix, p2), 1)
    p1_optimality = p1_payoffs - variables.p1_nash_value  # <= 0

    # No strategy for p2 does better than the nash value
    p2_payoffs = -tf.squeeze(tf.matmul(p1, self.payoff_matrix), 0)
    p2_optimality = p2_payoffs + variables.p1_nash_value  # <= 0

    return tf.concat([p1_optimality, p2_optimality], axis=0)

def normalize_probs(probs: tf.Tensor) -> tf.Tensor:
  probs = tf.maximum(probs, 0)
  return probs / tf.reduce_sum(probs, axis=-1, keepdims=True)

class NashVariables(tp.NamedTuple):
  p1: tf.Tensor
  p2: tf.Tensor
  p1_nash_value: tf.Tensor


class ZeroSumNashProblem(optimization.FeasibilityProblem[NashVariables]):

  def __init__(self, payoff_matrices: tf.Tensor):
    self.payoff_matrices = tf.convert_to_tensor(payoff_matrices)

  def batch_size(self) -> int:
    return self.payoff_matrices.shape[0]

  def initial_variables(self) -> NashLogits:
    b, d1, d2 = self.payoff_matrices.shape.as_list()
    dtype = self.payoff_matrices.dtype

    return NashVariables(
        p1=tf.ones([b, d1], dtype=dtype) / d1,
        p2=tf.ones([b, d2], dtype=dtype) / d2,
        p1_nash_value=tf.zeros([b], dtype=dtype),
    )

  def constraint_violations(self, variables: NashVariables) -> tf.Tensor:
    constraints = [
        -variables.p1,  # p1 >= 0
        -variables.p2,  # p2 >= 0
    ]

    # No strategy for p1 does better than the nash value
    p1_payoffs = tf.linalg.matvec(self.payoff_matrices, variables.p2)
    # p1_payoffs = tf.squeeze(tf.matmul(self.payoff_matrix, p2), 1)
    p1_nash_value = tf.expand_dims(variables.p1_nash_value, -1)
    p1_optimality = p1_payoffs - p1_nash_value  # <= 0

    # No strategy for p2 does better than the nash value
    p2_payoffs = -tf.linalg.matvec(
        self.payoff_matrices, variables.p1, transpose_a=True)
    p2_nash_value = -tf.expand_dims(variables.p1_nash_value, -1)
    p2_optimality = p2_payoffs - p2_nash_value  # <= 0

    constraints.extend([p1_optimality, p2_optimality])

    return tf.concat(constraints, axis=-1)

  def equality_violations(self, variables: NashVariables) -> tf.Tensor:
    return tf.stack([
        tf.reduce_sum(variables.p1, axis=-1) - 1.0,
        tf.reduce_sum(variables.p2, axis=-1) - 1.0,
    ], axis=-1)

def solve_zero_sum_nash_pulp(
    payoff_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
  import pulp

  d1, d2 = payoff_matrix.shape

  problem = pulp.LpProblem('zero_sum_nash')

  constraints = {}

  p1 = [pulp.LpVariable(f'p1_{i}', lowBound=0) for i in range(d1)]
  constraints['p1_sum_one'] = pulp.lpSum(p1) == 1

  p2 = [pulp.LpVariable(f'p2_{i}', lowBound=0) for i in range(d2)]
  constraints['p2_sum_one'] = pulp.lpSum(p2) == 1

  p1_nash = pulp.LpVariable('p1_nash')
  p2_nash = -p1_nash

  # No strategy for p1 does better than the nash value
  for i in range(d1):
    payoff_i = pulp.lpDot(p2, payoff_matrix[i])
    # payoff_i = pulp.LpAffineExpression([
    #     (p2[j], payoff_matrix[i, j]) for j in range(d2)])
    constraints[f'p1_{i}'] = payoff_i <= p1_nash

  # No strategy for p2 does better than the nash value
  for j in range(d2):
    payoff_j = pulp.lpDot(p1, -payoff_matrix[:, j])
    # payoff_j = pulp.LpAffineExpression([
    #     (p1[i], -payoff_matrix[i, j]) for i in range(d1)])
    constraints[f'p2_{j}'] = payoff_j <= p2_nash

  for name, constraint in constraints.items():
    problem.addConstraint(constraint, name=name)

  problem.solve(pulp.PULP_CBC_CMD(msg=0))

  p1_values = np.array([p.value() for p in p1])
  p2_values = np.array([p.value() for p in p2])

  return p1_values, p2_values, p1_nash.value()

def solve_zero_sum_nash_gambit(
    payoff_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
  from pygambit.gambit import Game
  from pygambit import nash

  game = Game.from_arrays(payoff_matrix, -payoff_matrix)
  result = nash.lp_solve(game, rational=False)

  equilibrium = result.equilibria[0]
  strategies = {}

  for player, strategy in equilibrium.mixed_strategies():
    strategies[player.label] = np.array([x for _, x in strategy])

  return strategies['1'], strategies['2'], equilibrium.payoff('1')
