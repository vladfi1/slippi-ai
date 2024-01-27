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
  peppi_game_raw = peppi_py.read_slippi(game_path)
  peppi_game = parse_peppi.from_peppi(peppi_game_raw)
  peppi_game = types.array_to_nest(peppi_game)

  libmelee_game = parse_libmelee.get_slp(game_path)
  libmelee_game = types.array_to_nest(libmelee_game)

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
    try:
      np.testing.assert_equal(x, y)
    except AssertionError as e:
      raise AssertionError(path) from e

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

class Metadata(typing.NamedTuple):
  lastFrame: int
  slippi_version: List[int]
  num_players: int
  players: List[PlayerMeta]
  stage: int
  timer: int
  is_teams: bool
  # attempt to guess winner by looking at last frame
  # index into player array, NOT port number
  winner: Optional[int]

  # These are missing from ranked-anonymized replays
  startAt: Optional[str] = None
  playedOn: Optional[str] = None

  # Game hash
  key: Optional[str] = None

  @staticmethod
  def from_dict(d: dict) -> 'Metadata':
    players = [PlayerMeta(**p) for p in d['players']]
    kwargs = dict(d, players=players)
    return Metadata(**kwargs)

def mode(xs: np.ndarray):
  unique, counts = np.unique(xs, return_counts=True)
  i = np.argmax(counts)
  return unique[i]

def port_to_int(port: str) -> int:
  assert port.startswith('P')
  return int(port[1])

def compute_winner(game: peppi_py.Game) -> Optional[int]:
  if len(game.start['players']) > 2:
    # TODO: handle more than 2 players
    return None

  last_frame = game.frames[-1]
  stock_counts = {}
  for port, player in last_frame['ports'].items():
    stock_counts[port] = player['leader']['post']['stocks'].as_py()

  # s is 0 or None for eliminated players
  losers = [p for p, s in stock_counts.items() if not s]
  if losers:
    winners = [p for p, s in stock_counts.items() if s]
    if len(winners) == 1:
      return port_to_int(winners[0])

  return None

def get_metadata(game: peppi_py.Game) -> dict:
  result = {}

  # Metadata section may be empty for ranked-anonymized replays.
  metadata = game.metadata
  for key in ['startAt', 'playedOn']:
    if key in metadata:
      result[key] = metadata[key]
  del metadata

  result['lastFrame'] = game.frames.field('id')[-1].as_py()

  start = game.start
  result['slippi_version'] = start['slippi']['version']

  players = start['players']
  player_metas = []

  for player in players:
    port = player['port']  # P[1-4]

    # get most-played character
    if len(players) == 2:
      leader = game.frames.field('ports').field(port).field('leader')
      cs = leader.field('post').field('character').to_numpy()
      character = int(mode(cs))
    else: # Non-1v1 games will have nulls when players are eliminated
      leader = game.frames[0]['ports'][port]['leader']
      character = leader['post']['character'].as_py()

    player_metas.append(dict(
        port=port_to_int(port),
        character=character,
        type=0 if player['type'] == 'Human' else 1,
        name_tag=player['name_tag'],
        netplay=player.get('netplay'),
    ))
  result.update(
      num_players=len(player_metas),
      players=player_metas,
  )

  for key in ['stage', 'timer', 'is_teams']:
    result[key] = start[key]

  # compute winner
  result['winner'] = compute_winner(game)

  return result

def get_metadata_safe(path: str) -> dict:
  try:
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

    # peppi-py bug with ICs
    # Character.NANA.value,
    # Character.POPO.value,

    Character.UNKNOWN_CHARACTER,
])

ALLOWED_CHARACTERS = set(Character) - BANNED_CHARACTERS
ALLOWED_CHARACTER_VALUES = set(c.value for c in ALLOWED_CHARACTERS)

MIN_SLP_VERSION = [2, 1, 0]

MIN_FRAMES = 60 * 60  # one minute
GAME_TIME = 60 * 8  # eight minutes

def is_training_replay(meta_dict: dict) -> tuple[bool, str]:
  if meta_dict.get('invalid') or meta_dict.get('failed'):
    return False, meta_dict.get('reason', 'invalid or failed')

  # del meta_dict['_id']
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
      # import ipdb; ipdb.set_trace()
      return False, 'not human'
    if player.character not in ALLOWED_CHARACTER_VALUES:
      return False, 'invalid character'

  return True, ''
