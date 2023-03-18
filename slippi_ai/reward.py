"""Calculate rewards for off-policy reinforcement learning."""

import numpy as np

from slippi_ai.types import Game, Player

def is_dying(player_action: np.ndarray) -> np.ndarray:
  # See https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
  return player_action <= 0xA

def process_deaths(deaths: np.ndarray) -> np.ndarray:
  # Players are in a dead action-state for many consecutive frames.
  # Prune all but the first frame of death
  return np.logical_and(np.logical_not(deaths[:-1]), deaths[1:])

def process_damages(damages: np.ndarray) -> np.ndarray:
  damages = damages.astype(np.float32)
  return np.maximum(damages[1:] - damages[:-1], 0)

def compute_rewards(game: Game, damage_ratio=0.01):
  '''
    Args:
      game: nest of np.arrays of length T, from make_dataset.py
      damage_ratio: How much damage (percent) counts relative to stocks
    Returns:
      A length (T-1) np.array of rewards
  '''

  def player_reward(player: Player):
    dying = is_dying(player.action)
    deaths = process_deaths(dying).astype(np.float32)
    damage = damage_ratio * process_damages(player.percent)
    return - (deaths + damage)

  rewards = player_reward(game.p0) - player_reward(game.p1)

  # sanity checks
  assert np.all(rewards > -2)
  assert np.all(rewards < 2)

  return rewards
