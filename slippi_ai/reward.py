"""Calculate rewards for off-policy reinforcement learning."""

import numpy as np

def is_dying(player_action):
  # See https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
  return player_action <= 0xA

def process_deaths(deaths):
  # Players are in a dead action-state for many consecutive frames.
  # Prune all but the first frame of death
  return np.logical_and(np.logical_not(deaths[:-1]), deaths[1:])

def process_damages(damages):
  return np.maximum(damages[1:] - damages[:-1], 0)

def compute_rewards(game, enemies=[2], allies=[1], damage_ratio=0.01):
  '''
    Args:
      game: nest of np.arrays of length T, from make_dataset.py
      enemies: List of controller ports for the enemy team
      allies: List of controller ports for our team
      damage_ratio: How much damage (percent) counts relative to stocks
    Returns:
      A length (T-1) np.array of rewards
  '''
  pids = enemies + allies

  deaths = {p : process_deaths(is_dying(game['player'][p]['action'])) for p in pids}
  damages = {p : process_damages(game['player'][p]['percent']) for p in pids}

  losses = {p : deaths[p] + damage_ratio * damages[p] for p in pids}

  return sum(losses[p] for p in enemies) - sum(losses[p] for p in allies)
