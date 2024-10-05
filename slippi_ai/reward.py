"""Calculate rewards."""

import dataclasses

import numpy as np

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

def get_bad_ledge_grabs(player: Player, opponent: Player) -> np.ndarray:
  ledge_grabs = grabbed_ledge(player.action)

  # Don't penalize if opponent is offstage
  opponent_direction = player.x < opponent.x  # True if opponent is right
  center_direction = player.x < 0  # True if center is right
  opponent_towards_center = opponent_direction == center_direction
  bad_ledge_grabs = np.logical_and(ledge_grabs, opponent_towards_center[:-1])

  # Also ok if opponent is invincible (like after respawn)
  bad_ledge_grabs = np.logical_and(
      bad_ledge_grabs, np.logical_not(opponent.invulnerable[:-1]))

  return bad_ledge_grabs

def normalize(xys, epsilon=1e-6):
  r = np.sqrt(np.sum(np.square(xys), axis=-1, keepdims=True))
  return xys / (r + epsilon)

def compute_approaching_factor(
    player: Player, opponent: Player) -> np.ndarray:
  """Measures how much we are approaching the opponent on each frame."""
  xy = np.stack([player.x, player.y], axis=-1)
  v = xy[1:] - xy[:-1]


  opp_xy = np.stack([opponent.x, opponent.y], axis=-1)
  dxy = normalize(opp_xy - xy)

  approach_factor = np.sum(v * dxy[:-1], axis=-1)

  # Player teleports when respawning.
  dying = is_dying(player.action)
  respawning = np.logical_and(dying[:-1], np.logical_not(dying[1:]))
  approach_factor = np.where(respawning, 0, approach_factor)

  return approach_factor

stage_to_edge_x = {
    stage.value: x for stage, x in melee.stages.EDGE_POSITION.items()
}
get_edge_x = np.vectorize(lambda x: stage_to_edge_x.get(x, 100))

def amount_offstage(player: Player, stage: np.ndarray) -> np.ndarray:
  stage_xs = get_edge_x(stage[0])
  below = np.minimum(player.y, 0)
  offstage = np.maximum(np.abs(player.x) - stage_xs, 0)
  return np.sqrt(np.square(below) + np.square(offstage))

def is_stalling_offstage(player: Player, stage: np.ndarray) -> np.ndarray:
  return amount_offstage(player, stage) > 20  # arbitrary

def is_aerial_shine(player: Player):
  is_fox = player.character == melee.Character.FOX.value
  is_falco = player.character == melee.Character.FALCO.value
  is_spacie = np.logical_or(is_fox, is_falco)

  # We only care about aerial shines.
  is_shine = player.action == melee.Action.DOWN_B_AIR

  return np.logical_and(is_spacie, is_shine)

def find_offstage_shine_stalls(player: Player, stage: np.ndarray):
  offstage = is_stalling_offstage(player, stage) > 20
  return np.logical_and(offstage, is_aerial_shine(player))

@dataclasses.dataclass
class RewardConfig:
  damage_ratio: float = 0.01
  ledge_grab_penalty: float = 0
  approaching_factor: float = 0
  stalling_penalty: float = 0  # per second

def compute_rewards(
    game: Game,
    damage_ratio: float = 0.01,
    ledge_grab_penalty: float = 0,
    approaching_factor: float = 0,
    stalling_penalty: float = 0  # per second
) -> np.ndarray:
  '''
    Args:
      game: nest of np.arrays of length T, from make_dataset.py
      damage_ratio: How much damage (percent) counts relative to stocks
    Returns:
      A length (T-1) np.array of rewards
  '''

  def player_reward(player: Player, opponent: Player):
    deaths = process_deaths(player.action).astype(np.float32)
    damages = damage_ratio * process_damages(player.percent)

    bad_ledge_grabs = get_bad_ledge_grabs(player, opponent).astype(np.float32)
    ledge_grab_penalties = ledge_grab_penalty * bad_ledge_grabs

    stalling = is_stalling_offstage(player, game.stage)[1:]
    stalling_penalties = (stalling_penalty / 60) * stalling.astype(np.float32)

    reward = approaching_factor * compute_approaching_factor(player, opponent)
    reward -= (deaths + damages + ledge_grab_penalties + stalling_penalties)

    return reward

  # Zero-sum rewards ensure there can be no collusion.
  rewards = player_reward(game.p0, game.p1) - player_reward(game.p1, game.p0)

  # sanity checks
  assert np.all(rewards > -2)
  assert np.all(rewards < 2)
  assert rewards.dtype == np.float32

  return rewards

def player_stats(player: Player, opponent: Player, stage: np.ndarray) -> dict:
  FPM = 60 * 60
  return dict(
      deaths=process_deaths(player.action).mean() * FPM,
      damages=process_damages(player.percent).mean() * FPM,
      ledge_grabs=get_bad_ledge_grabs(player, opponent).mean() * FPM,
      approaching_factor=compute_approaching_factor(player, opponent).mean(),
      stalling=is_stalling_offstage(player, stage).mean(),
  )

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
