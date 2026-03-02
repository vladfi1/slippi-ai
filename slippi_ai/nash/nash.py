import numpy as np

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

  problem.solve(pulp.PULP_CBC_CMD(msg=False))

  p1_values = np.array([p.value() for p in p1])
  p2_values = np.array([p.value() for p in p2])

  p1_nash_value = p1_nash.value()
  assert p1_nash_value is not None

  return p1_values, p2_values, p1_nash_value

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
