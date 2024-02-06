import atexit
import dataclasses
import itertools
import json
import multiprocessing as mp
import os
import random
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple, Iterator, NamedTuple
import zlib

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import tree

import melee

from slippi_ai import embed, reward, utils
from slippi_ai.types import Controller, Game, Nest, game_array_to_nt

from slippi_ai.embed import StateActionReward

class Batch(NamedTuple):
  game: StateActionReward
  needs_reset: bool

class PlayerMeta(NamedTuple):
  character: int
  name: str

class ReplayMetadata(NamedTuple):
  p0: PlayerMeta
  p1: PlayerMeta
  stage: int

class ReplayInfo(NamedTuple):
  path: str
  swap: bool
  metadata: Optional[ReplayMetadata] = None

def _charset(chars: Optional[Iterable[melee.Character]]) -> Set[int]:
  if chars is None:
    chars = list(melee.Character)
  return set(c.value for c in chars)


@dataclasses.dataclass
class DatasetConfig:
  data_dir: Optional[str] = None  # required
  meta_path: Optional[str] = None
  test_ratio: float = 0.1
  # comma-separated lists of characters, or "all"
  allowed_characters: str = 'all'
  allowed_opponents: str = 'all'
  seed: int = 0


def train_test_split(
    config: DatasetConfig,
) -> Tuple[List[ReplayInfo], List[ReplayInfo]]:
  filenames = sorted(os.listdir(config.data_dir))
  print(f"Found {len(filenames)} files.")

  replays: list[ReplayInfo] = []

  if config.meta_path is not None:
    with open(config.meta_path) as f:
      meta_rows: list[dict] = json.load(f)

    # check that we have the right metadata
    filenames_set = set(filenames)
    assert all(row['slp_md5'] in filenames_set for row in meta_rows)

    allowed_characters = _charset(chars_from_string(config.allowed_characters))
    allowed_opponents = _charset(chars_from_string(config.allowed_opponents))

    for row in meta_rows:
      player_metas = []
      for player in row['players']:
        netplay = player['netplay']
        if netplay is None:
          name = player['name_tag']
        else:
          name = netplay['name']
        player_metas.append(PlayerMeta(
            character=player['character'],
            name=name))

      replay_meta = ReplayMetadata(
          p0=player_metas[0],
          p1=player_metas[1],
          stage=row['stage'])

      c0 = replay_meta.p0.character
      c1 = replay_meta.p1.character
      replay_path = os.path.join(config.data_dir, row['slp_md5'])

      if c0 in allowed_characters and c1 in allowed_opponents:
        replays.append(ReplayInfo(replay_path, False, replay_meta))

      if c0 in allowed_opponents and c1 in allowed_characters:
        replays.append(ReplayInfo(replay_path, True, replay_meta))
  else:
    if not (config.allowed_characters == 'all'
            and config.allowed_opponents == 'all'):
      raise ValueError(
          "Can't filter by character without metadata. "
          "Please provide a metadata file.")

    for filename in filenames:
      replay_path = os.path.join(config.data_dir, filename)
      replays.append(ReplayInfo(replay_path, False))
      replays.append(ReplayInfo(replay_path, True))

  # reproducible train/test split
  rng = random.Random(config.seed)
  rng.shuffle(replays)
  num_test = int(config.test_ratio * len(replays))

  train_replays = replays[num_test:]
  test_replays = replays[:num_test]

  return train_replays, test_replays

_name_to_character = {c.name.lower(): c for c in melee.Character}

def chars_from_string(chars: str) -> Optional[List[melee.Character]]:
  if chars == 'all':
    return None
  chars = chars.split(',')
  return [_name_to_character[c] for c in chars]


def game_len(game: StateActionReward):
  # We use the reward length, effectively ignoring the last frame.
  return len(game.reward)

class TrajectoryManager:
  # TODO: manage recurrent state? can also do it in the learner

  def __init__(self, source: Iterator[StateActionReward]):
    self.source = source
    self.game: StateActionReward = None
    self.frame = None

  def find_game(self, n):
    while True:
      game = next(self.source)
      if game_len(game) >= n: break
    self.game = game
    self.frame = 0

  def grab_chunk(self, n) -> Tuple[StateActionReward, bool]:
    """Grabs a chunk from a trajectory."""
    # TODO: write a unit test for this
    needs_reset = self.game is None or self.frame + n > game_len(self.game)

    if needs_reset:
      self.find_game(n)

    new_frame = self.frame + n
    slice = lambda a: a[self.frame:new_frame]
    # faster than tree.map_structure
    chunk = utils.map_nt(slice, self.game)
    self.frame = new_frame

    return Batch(chunk, needs_reset)

def swap_players(game: Game) -> Game:
  return game._replace(p0=game.p1, p1=game.p0)

def read_table(path: str, compressed: bool) -> Game:
  if compressed:
    with open(path, 'rb') as f:
      contents = f.read()
    contents = zlib.decompress(contents)
    reader = pyarrow.BufferReader(contents)
    table = pq.read_table(reader)
  else:
    table = pq.read_table(path)

  game_struct = table['root'].combine_chunks()
  return game_array_to_nt(game_struct)

class DataSource:
  def __init__(
      self,
      replays: List[ReplayInfo],
      # preprocesses (discretizes) actions before repeat detection
      embed_controller: embed.Embedding[Controller, Any],
      compressed=True,
      batch_size: int = 64,
      unroll_length: int = 64,
      max_action_repeat=15,
      # None means all allowed.
      allowed_characters: Optional[list[melee.Character]] = None,
      allowed_opponents: Optional[list[melee.Character]] = None,
  ):
    self.replays = replays
    self.batch_size = batch_size
    self.unroll_length = unroll_length
    self.compressed = compressed
    self.max_action_repeat = max_action_repeat
    self.embed_controller = embed_controller
    trajectories = self.produce_trajectories()
    self.managers = [
        TrajectoryManager(trajectories)
        for _ in range(batch_size)]

    self.allowed_characters = _charset(allowed_characters)
    self.allowed_opponents = _charset(allowed_opponents)

  def produce_trajectories(self) -> Iterator[StateActionReward]:
    games = self.produce_raw_games()
    games = filter(self.is_allowed, games)
    games = map(self.process_game, games)
    return games

  def process_game(self, game: Game) -> StateActionReward:
    rewards = reward.compute_rewards(game)
    controllers = self.embed_controller.from_state(game.p0.controller)

    # TODO: configure game embedding
    states = embed.default_embed_game.from_state(game)
    result = StateActionReward(
        state=states,
        action=embed.ActionWithRepeat(
            action=controllers,
            repeat=np.zeros(len(game.stage), dtype=np.int64)),
        reward=rewards)
    return result

  def produce_raw_games(self) -> Iterator[Game]:
    """Raw games without post-processing."""
    self.replay_counter = 0
    for replay in itertools.cycle(self.replays):
      self.replay_counter += 1
      game = read_table(replay.path, self.compressed)
      if replay.swap:
        game = swap_players(game)
      # assert self.is_allowed(game)
      yield game

  def is_allowed(self, game: Game) -> bool:
    # TODO: handle Zelda/Sheik transformation
    return (
        game.p0.character[0] in self.allowed_characters
        and
        game.p1.character[0] in self.allowed_opponents)

  def __next__(self) -> Tuple[Batch, float]:
    next_batch: Batch = utils.batch_nest_nt(
        [m.grab_chunk(self.unroll_length) for m in self.managers])
    epoch = self.replay_counter / len(self.replays)
    assert next_batch.game.action.repeat.shape[-1] == self.unroll_length
    return next_batch, epoch

def produce_batches(data_source_kwargs, batch_queue):
  data_source = DataSource(**data_source_kwargs)
  while True:
    batch_queue.put(next(data_source))

class DataSourceMP:
  def __init__(self, buffer=4, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
    self.batch_queue = mp.Queue(buffer)
    self.process = mp.Process(
        target=produce_batches, args=(kwargs, self.batch_queue))
    self.process.start()

    atexit.register(self.batch_queue.close)
    atexit.register(self.process.terminate)

  def __next__(self) -> Tuple[Batch, float]:
    return self.batch_queue.get()

CONFIG = dict(
    batch_size=32,
    unroll_length=64,
    compressed=True,
    max_action_repeat=0,
    in_parallel=True,
)

def make_source(
    in_parallel: bool,
    **kwargs):
  constructor = DataSourceMP if in_parallel else DataSource
  return constructor(**kwargs)
