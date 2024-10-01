"""Calculate rewards."""

import dataclasses

import numpy as np
import tree

import melee
from slippi_ai.types import Game, Player

def is_dying(player_action: np.ndarray) -> np.ndarray:
  # See https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
  return player_action <= 0xA

def process_deaths(player_action: np.ndarray) -> np.ndarray:
  deaths = is_dying(player_action)
  # Players are in a dead action-state for many consecutive frames.
  # Prune all but the first frame of death
  return np.logical_and(np.logical_not(deaths[:-1]), deaths[1:])

def process_damages(damages: np.ndarray) -> np.ndarray:
  damages = damages.astype(np.float32)
  return np.maximum(damages[1:] - damages[:-1], 0)

def grabbed_ledge(player_action: np.ndarray) -> np.ndarray:
  is_ledge_grab = player_action == 0xFC  # "CliffCatch"
  return np.logical_and(np.logical_not(is_ledge_grab[:-1]), is_ledge_grab[1:])

@dataclasses.dataclass
class RewardConfig:
  damage_ratio: float = 0.01
  ledge_grab_penalty: float = 0

def compute_rewards(
    game: Game,
    damage_ratio: float = 0.01,
    ledge_grab_penalty: float = 0,
) -> np.ndarray:
  '''
    Args:
      game: nest of np.arrays of length T, from make_dataset.py
      damage_ratio: How much damage (percent) counts relative to stocks
    Returns:
      A length (T-1) np.array of rewards
  '''

  def player_reward(player: Player):
    deaths = process_deaths(player.action).astype(np.float32)
    damages = damage_ratio * process_damages(player.percent)

    ledge_grabs = grabbed_ledge(player.action).astype(np.float32)
    ledge_grab_penalties = ledge_grab_penalty * ledge_grabs

    return - (deaths + damages + ledge_grab_penalties)

  # Zero-sum rewards ensure there can be no collusion.
  rewards = player_reward(game.p0) - player_reward(game.p1)

  # sanity checks
  assert np.all(rewards > -2)
  assert np.all(rewards < 2)
  assert rewards.dtype == np.float32

  return rewards

def player_stats(player: Player) -> dict:
  FPM = 60 * 60
  stats = dict(
      deaths=process_deaths(player.action),
      damages=process_damages(player.percent),
      ledge_grabs=grabbed_ledge(player.action),
  )
  return tree.map_structure(lambda x: np.mean(x) * FPM, stats)

# TODO: test that the two ways of getting reward yield the same results
def get_reward(
    prev_state: melee.GameState,
    next_state: melee.GameState,
    own_port: int,
    opponent_port: int,
    damage_ratio: float = 0.01,
) -> float:
  """Reward implemented directly on gamestates."""

  def player_reward(port: int):
    players = [prev_state.players[port], next_state.players[port]]
    actions = np.array([p.action for p in players])
    deaths = process_deaths(actions).astype(np.float32).item()

    percents = np.array([p.percent for p in players])
    damage = damage_ratio * process_damages(percents).item()
    return - (deaths + damage)

  return player_reward(own_port) - player_reward(opponent_port)
