"""Preprocessing of .slp files."""

from typing import List, Optional
import typing

import numpy as np
import tree

from melee import enums, Character
import peppi_py

from slippi_db import parse_libmelee
from slippi_db import parse_peppi
from slippi_ai import types

def assert_same_parse(game_path: str):
  peppi_game_raw = parse_peppi.read_slippi(game_path)
  peppi_game = parse_peppi.from_peppi(peppi_game_raw)
  peppi_game = types.array_to_nt(types.Game, peppi_game)

  libmelee_game = parse_libmelee.get_slp(game_path)
  libmelee_game = types.array_to_nt(types.Game, libmelee_game)

  # def assert_equal(path, pv):
  #   lv = libmelee_game
  #   for p in path:
  #     lv = lv[p]
  #   path_str = '.'.join(map(str, path))
  #   neq = lv != pv
  #   if np.any(neq):
  #     game = peppi_py.game(game_path)
  #     lv_neq = lv[neq]
  #     pv_neq = pv[neq]
  #     print(path_str, lv_neq[:5], pv_neq[:5])
  #     raise AssertionError(path_str)

  # tree.map_structure_with_path(assert_equal, peppi_game)

  def assert_equal(path, x, y):
    diff_indices = np.arange(len(x))[x != y]
    if diff_indices.size > 0:
      raise AssertionError(f'{path} differs at indices {diff_indices[:5].tolist()}')

  tree.map_structure_with_path(assert_equal, peppi_game, libmelee_game)

class PlayerMeta(typing.NamedTuple):
  name_tag: str
  # most played character
  character: int
  # 1-indexed
  port: int
  # 0 is human, 1 is cpu
  type: int
  # netplay info
  netplay: dict
  # amount of damage taken during the game
  damage_taken: Optional[int] = None

class Metadata(typing.NamedTuple):
  lastFrame: int
  slippi_version: List[int]
  num_players: int
  players: List[PlayerMeta]
  stage: int
  timer: int
  is_teams: bool
  # Attempt to guess winner by looking at last frame.
  # Note: indexes into player array, NOT port number.
  winner: Optional[int]

  # These are missing from ranked-anonymized replays
  startAt: Optional[str] = None
  playedOn: Optional[str] = None

  # Game hash
  key: Optional[str] = None

  match: Optional[dict] = None

  @staticmethod
  def from_dict(d: dict) -> 'Metadata':
    players = [PlayerMeta(**p) for p in d['players']]
    kwargs = dict(d, players=players)
    return Metadata(**kwargs)

def mode(xs: np.ndarray):
  unique, counts = np.unique(xs, return_counts=True)
  i = np.argmax(counts)
  return unique[i]

_OPPONENT_IDX = {0: 1, 1: 0}

def winner_from_lras(game: peppi_py.Game) -> Optional[int]:
  """Assume that the LRAS player lost."""
  end = game.end
  if end is None or end.lras_initiator is None:
    return None

  port_to_index = {
      player.port: idx for idx, player
      in enumerate(game.start.players)
  }

  loser_idx = port_to_index[end.lras_initiator]
  return _OPPONENT_IDX[loser_idx]

def compute_winner(game: peppi_py.Game) -> Optional[int]:
  if len(game.start.players) > 2:
    # TODO: handle more than 2 players
    return None

  end = game.end
  if end is not None and end.players is not None:
    for i, player_end in enumerate(end.players):
      if player_end.placement == 0:
        return i
    raise ValueError('No player with placement = 0')

  # Try to see who had nonzero stocks left on the last frame.
  stock_counts = {}
  for i, port in enumerate(game.frames.ports):
    stock_counts[i] = port.leader.post.stocks[-1].as_py()

  # s is 0 or None for eliminated players
  losers = [p for p, s in stock_counts.items() if not s]
  if losers:
    winners = [p for p, s in stock_counts.items() if s]
    if len(winners) == 1:
      return winners[0]

  lras_winner = winner_from_lras(game)
  if lras_winner is not None:
    return lras_winner

  return None

def get_dict(obj: object, *keys: str) -> dict:
  result = {}
  for key in keys:
    result[key] = getattr(obj, key)
  return result

def get_metadata(game: peppi_py.Game) -> dict:
  result = {}

  # Metadata section may be empty for ranked-anonymized replays.
  metadata = game.metadata
  for key in ['startAt', 'playedOn']:
    if key in metadata:
      result[key] = metadata[key]
  del metadata

  result['lastFrame'] = game.frames.id[-1].as_py()

  start = game.start
  result['slippi_version'] = start.slippi.version

  if result['slippi_version'] >= (3, 14, 0):
    result['match'] = get_dict(start.match, 'id', 'game', 'tiebreaker')

  players = start.players
  player_metas = []

  for player, port in zip(players, game.frames.ports):
    leader = port.leader

    # `player.character` is an "external ID", would need to map to internal ID
    # See https://github.com/project-slippi/slippi-wiki/blob/master/SPEC.md#melee-ids
    # assert int(cs[0]) == player.character

    # get most-played character, in case of Zelda/Sheik switches
    if len(players) == 2:
      character = int(mode(leader.post.character.to_numpy()))
      percent = leader.post.percent.to_numpy()
      damage = float(np.maximum(percent[1:] - percent[:-1], 0).sum())
    else: # Non-1v1 games will have nulls when players are eliminated
      character = leader.post.character[0].as_py()
      damage = None

    netplay = None
    if player.netplay is not None:
      netplay = get_dict(player.netplay, 'name', 'code', 'suid')

    player_metas.append(dict(
        port=player.port.value,
        character=character,
        type=player.type.value,  # 0 is human
        name_tag=player.name_tag,
        netplay=netplay,
        damage_taken=damage,
    ))
  result.update(
      num_players=len(player_metas),
      players=player_metas,
  )

  for key in ['stage', 'timer', 'is_teams']:
    result[key] = getattr(start, key)

  # compute winner
  result['winner'] = compute_winner(game)

  return result

def get_metadata_safe(path: str) -> dict:
  try:
    # Sadly skip_frames=True crashed on some replays
    game = peppi_py.read_slippi(path)
    return get_metadata(game)
  except BaseException as e:
    return dict(
        invalid=True,
        reason=str(e),
    )
  except:  # should be a catch-all
    return dict(invalid=True, reason='uncaught exception')

# def get_metadata(game: data.Game) -> Metadata:
#   return Metadata(
#       characters={port: xs['character'][0] for port, xs in game['players'].items()},
#       stage=game['stage'][0],
#       num_frames=len(game['stage']),
#   )


BANNED_CHARACTERS = set([
    # Kirby's actions aren't fully mapped out yet
    Character.KIRBY,

    Character.UNKNOWN_CHARACTER,
])

ALLOWED_CHARACTERS = set(Character) - BANNED_CHARACTERS
ALLOWED_CHARACTER_VALUES = set(c.value for c in ALLOWED_CHARACTERS)

MIN_SLP_VERSION = (2, 1, 0)

MIN_FRAMES = 60 * 60  # one minute
GAME_TIME = 60 * 8  # eight minutes

def is_training_replay(meta_dict: dict) -> tuple[bool, str]:
  if meta_dict.get('invalid') or meta_dict.get('failed'):
    return False, meta_dict.get('reason', 'invalid or failed')

  meta = Metadata.from_dict(meta_dict)

  if meta.slippi_version < MIN_SLP_VERSION:
    return False, 'slippi version too low'
  if meta.num_players != 2:
    return False, 'not 1v1'
  if meta.lastFrame < MIN_FRAMES:
    return False, 'game length too short'
  if meta.timer != GAME_TIME:
    return False, 'timer not set to 8 minutes'
  if enums.to_internal_stage(meta.stage) == enums.Stage.NO_STAGE:
    return False, 'invalid stage'

  for player in meta.players:
    if player.type != 0:
      return False, 'not human'
    if player.character not in ALLOWED_CHARACTER_VALUES:
      return False, 'invalid character'

  return True, ''
