"""Preprocessing of .slp files."""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import typing

import numpy as np
import pyarrow as pa
import tree

from melee import enums, Character
import peppi_py

from slippi_db import parse_libmelee
from slippi_db import parse_peppi
from slippi_ai import types

def assert_same_parse(game_path: str):
  peppi_game = parse_peppi.get_slp(game_path)
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

class Metadata(typing.NamedTuple):
  key: str
  startAt: str
  lastFrame: int
  playedOn: str
  slippi_version: List[int]
  num_players: int
  players: List[PlayerMeta]
  stage: int
  timer: int
  is_teams: bool
  # attempt to guess winner by looking at last frame
  # index into player array, NOT port number
  winner: Optional[int]

  @staticmethod
  def from_dict(d: dict) -> 'Metadata':
    players = [PlayerMeta(**p) for p in d['players']]
    kwargs = dict(d, players=players)
    return Metadata(**kwargs)

def mode(xs: np.ndarray):
  unique, counts = np.unique(xs, return_counts=True)
  i = np.argmax(counts)
  return unique[i]

def get_metadata(path: str) -> dict:
  try:
    game = peppi_py.game(path)
  except OSError as e:
    return dict(
        invalid=True,
        reason=str(e),
    )

  metadata = game['metadata']
  start = game['start']

  result = {}
  for key in ['startAt', 'lastFrame', 'playedOn']:
    result[key] = metadata[key]
  result['slippi_version'] = start['slippi']['version']

  players = []
  ports = []

  for i, player in enumerate(start['players']):
    port = int(player['port'][1])
    ports.append(port)

    # get most-played character
    p = game['frames'].field('ports').field(str(i))
    c = p.field('leader').field('post').field('character').to_numpy()
    character = mode(c)

    players.append(dict(
        port=port,
        character=int(character),
        type=player['type'],
        name_tag=player['name_tag'],
    ))
  result.update(
      num_players=len(players),
      players=players,
  )

  for key in ['stage', 'timer', 'is_teams']:
    result[key] = start[key]

  # compute winner
  last_frame = game['frames'][-1]
  winner = None
  stock_counts = {}
  for index, player in last_frame['ports'].items():
    stock_counts[int(index)] = player['leader']['post']['stocks'].as_py()

  losers = [p for p, s in stock_counts.items() if s == 0]
  if losers:
    winners = [p for p, s in stock_counts.items() if s > 0]
    if len(winners) == 1:
      winner = winners[0]
  result['winner'] = winner

  return result

def get_metadata_safe(path: str) -> dict:
  try:
    return get_metadata(path)
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
    Character.KIRBY.value,
    # peppi-py bug with ICs
    # Character.NANA.value,
    # Character.POPO.value,
])

MIN_SLP_VERSION = [2, 0, 0]

MIN_TIME = 60 * 60  # one minute

def is_training_replay(meta_dict: dict) -> bool:
  if meta_dict.get('invalid', False):
    return False

  del meta_dict['_id']
  meta = Metadata.from_dict(meta_dict)

  if meta.num_players != 2:
    return False

  if meta.slippi_version < MIN_SLP_VERSION:
    return False

  for player in meta.players:
    if player.type != 0:
      return False

    if player.character in BANNED_CHARACTERS:
      return False

  if meta.lastFrame < MIN_TIME:
    return False

  if enums.to_internal_stage(meta.stage) == enums.Stage.NO_STAGE:
    return False

  return True
