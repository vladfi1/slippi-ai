import atexit
import collections
import dataclasses
import functools
import itertools
import json
import logging
import multiprocessing as mp
import os
import random
from typing import (
    Any, Callable, Iterable, List, Optional, Set, Tuple, Iterator, NamedTuple,
    Union,
)
import zlib

import numpy as np
import pyarrow
import pyarrow.parquet as pq

import melee

from slippi_ai import reward, utils, nametags, paths, observations
from slippi_ai.types import Game, game_array_to_nt, Controller
from slippi_ai.mirror import mirror_game

class PlayerMeta(NamedTuple):
  character: int
  name: str

  @classmethod
  def from_metadata(cls, player_meta: dict, raw: str) -> 'PlayerMeta':
    return cls(
        character=player_meta['character'],
        name=nametags.name_from_metadata(player_meta, raw=raw))

class ReplayMeta(NamedTuple):
  p0: PlayerMeta
  p1: PlayerMeta
  stage: int
  slp_md5: str

  @classmethod
  def from_metadata(cls, metadata: dict) -> 'ReplayMeta':
    raw = metadata['raw']
    return cls(
        p0=PlayerMeta.from_metadata(metadata['players'][0], raw),
        p1=PlayerMeta.from_metadata(metadata['players'][1], raw),
        stage=metadata['stage'],
        slp_md5=metadata['slp_md5'])

class ReplayInfo(NamedTuple):
  path: str
  swap: bool
  # We use empty tuple instead of None to play nicely with Tensorflow.
  meta: Union[ReplayMeta, Tuple[()]] = ()

  mirror: bool = False

  @property
  def main_player(self) -> PlayerMeta:
    if isinstance(self.swap, np.ndarray):
      return utils.map_nt(
          lambda p0, p1: np.where(self.swap, p1, p0),
          self.meta.p0, self.meta.p1)
    return self.meta.p1 if self.swap else self.meta.p0

class ChunkMeta(NamedTuple):
  start: int
  end: int
  info: ReplayInfo

# Action = TypeVar('Action')
Action = Controller

class StateAction(NamedTuple):
  state: Game
  # The action could actually be an "encoded" action type,
  # which might discretize certain components of the controller
  # such as the sticks and shoulder. Unfortunately NamedTuples can't be
  # generic. We could use a dataclass instead, but TF can't trace them.
  # Note that this is the action taken on the _previous_ frame.
  action: Action

  # Encoded name
  name: int

class Frames(NamedTuple):
  state_action: StateAction
  is_resetting: bool
  # The reward will have length one less than the states and actions.
  reward: np.float32

class Chunk(NamedTuple):
  frames: Frames
  meta: ChunkMeta

class Batch(NamedTuple):
  frames: Frames
  count: int  # For reproducing batches
  meta: ChunkMeta

def _charset(chars: Optional[Iterable[melee.Character]]) -> Set[int]:
  if chars is None:
    chars = list(melee.Character)
  return set(c.value for c in chars)

ALL = 'all'
NONE = 'none'

@dataclasses.dataclass
class DatasetConfig:
  data_dir: Optional[str] = None  # required
  meta_path: Optional[str] = None
  test_ratio: float = 0.1
  # comma-separated lists of characters, or "all"
  allowed_characters: str = ALL
  allowed_opponents: str = ALL
  # Filter by player
  allowed_names: str = ALL
  banned_names: str = NONE

  swap: bool = True  # yield swapped versions of each replay
  mirror: bool = False  # mirror left/right in each replay
  seed: int = 0

def create_name_filter(
    allowed_names: str,
    banned_names: str = NONE,
) -> Callable[[str], bool]:
  """Creates a function that filters names based on the allowed names."""
  if allowed_names != ALL:
    allowed_names_set = set(allowed_names.split(','))

  if banned_names == NONE:
    banned_names_set = set()
  else:
    banned_names_set = set(banned_names.split(','))

  def is_allowed(name: str) -> bool:
    name = nametags.normalize_name(name)
    if name in banned_names_set:
      return False
    if allowed_names == ALL:
      return True
    return name in allowed_names_set

  return is_allowed

def replays_from_meta(config: DatasetConfig) -> List[ReplayInfo]:
  replays = []

  with open(config.meta_path) as f:
    meta_rows: list[dict] = json.load(f)

  allowed_characters = _charset(chars_from_string(config.allowed_characters))
  allowed_opponents = _charset(chars_from_string(config.allowed_opponents))
  name_filter = create_name_filter(config.allowed_names, config.banned_names)

  banned_counts = collections.Counter()

  for row in meta_rows:
    replay_meta = ReplayMeta.from_metadata(row)
    replay_path = os.path.join(config.data_dir, replay_meta.slp_md5)

    if not config.swap:
      is_banned = False
      for name in [replay_meta.p0.name, replay_meta.p1.name]:
        if nametags.is_banned_name(name):
          banned_counts[name] += 1
          is_banned = True

      if is_banned:
        continue

      if (replay_meta.p0.character not in allowed_characters
          or replay_meta.p1.character not in allowed_opponents):
        continue

      replays.append(ReplayInfo(replay_path, False, replay_meta))

      continue

    for swap in [False, True]:
      players = [replay_meta.p0, replay_meta.p1]
      if swap:
        players = reversed(players)
      p0, p1 = players

      if (p0.character not in allowed_characters
          or p1.character not in allowed_opponents):
        continue

      if nametags.is_banned_name(p0.name):
        banned_counts[p0.name] += 1
        continue

      if not name_filter(p0.name):
        banned_counts[p0.name] += 1
        continue

      replays.append(ReplayInfo(replay_path, swap, replay_meta))

  print('Banned names:', banned_counts)

  return replays


def train_test_split(
    config: DatasetConfig,
) -> Tuple[List[ReplayInfo], List[ReplayInfo]]:
  filenames = sorted(os.listdir(config.data_dir))
  print(f"Found {len(filenames)} files.")

  replays: list[ReplayInfo] = []

  if config.meta_path is not None:
    replays = replays_from_meta(config)

    # check that we have the right metadata
    filenames_set = set(filenames)
    assert all(info.meta.slp_md5 in filenames_set for info in replays)
  else:
    if not (config.allowed_characters == ALL
            and config.allowed_opponents == ALL):
      raise ValueError(
          "Can't filter by character without metadata. "
          "Please provide a metadata file.")

    for filename in filenames:
      replay_path = os.path.join(config.data_dir, filename)
      replays.append(ReplayInfo(replay_path, False))
      if config.swap:
        replays.append(ReplayInfo(replay_path, True))

  # TODO: stable partition
  rng = random.Random(config.seed)
  rng.shuffle(replays)
  num_test = int(config.test_ratio * len(replays))

  train_replays = replays[num_test:]
  test_replays = replays[:num_test]

  def add_mirrored(unmirrored: List[ReplayInfo]):
    mirrored = []
    for info in unmirrored:
      mirrored.append(info._replace(mirror=True))
    unmirrored.extend(mirrored)
    rng.shuffle(unmirrored)

  # Add mirrored versions of each replay.
  # We do this here to avoid contamination between train and test sets.
  if config.mirror:
    add_mirrored(train_replays)
    # TODO: test on mirrored too, but keep separate from original test replays.

  return train_replays, test_replays

name_to_character = {c.name.lower(): c for c in melee.Character}

def chars_from_string(chars: str) -> Optional[List[melee.Character]]:
  if chars == ALL:
    return None
  return [name_to_character[c] for c in chars.split(',')]


def game_len(game: Game):
  return len(game.stage)

class TrajectoryManager:
  # TODO: manage recurrent state? can also do it in the learner

  def __init__(
      self,
      source: Iterator[ReplayInfo],
      unroll_length: int,
      encode_name: Callable[[str], int],
      overlap: int = 1,
      compressed: bool = True,
      game_filter: Optional[Callable[[Game], bool]] = None,
      observation_filter: Optional[observations.ObservationFilter] = None,
      reward_kwargs: dict = {},
  ):
    self.source = source
    self.compressed = compressed
    self.unroll_length = unroll_length
    self.overlap = overlap
    self.game_filter = game_filter or (lambda _: True)
    self.observation_filter = observation_filter
    self.reward_kwargs = reward_kwargs
    self.encode_name = encode_name

    self.game: Game = None
    self.reward: np.ndarray = None
    self.frame: int = None
    self.info: ReplayInfo = None

  def load_game(self, info: ReplayInfo) -> Game:
    game = read_table(info.path, compressed=self.compressed)
    if info.swap:
      game = swap_players(game)
    if info.mirror:
      # Also mirrors the controller inputs, which is what we want.
      game = mirror_game(game)
    return game

  def find_game(self):
    while True:
      info = next(self.source)
      game = self.load_game(info)
      if game_len(game) < self.unroll_length:
        continue
      if not self.game_filter(game):
        continue

      self.reward = reward.compute_rewards(game)
      break

    if self.observation_filter is not None:
      self.observation_filter.reset()
      game = self.observation_filter.filter_time(game)

    self.game = game
    self.frame = 0
    self.info = info
    self.name_code = self.encode_name(info.main_player.name)

  def grab_chunk(self) -> Chunk:
    """Grabs a chunk from a trajectory."""
    # TODO: write a unit test for this

    needs_reset = (
        self.game is None or
        self.frame + self.unroll_length > game_len(self.game))

    if needs_reset:
      self.find_game()

    start = self.frame
    end = start + self.unroll_length
    slice = lambda a: a[start:end]
    # faster than tree.map_structure
    states = utils.map_nt(slice, self.game)
    self.frame = end - self.overlap

    # Rewards could be deferred to the learner.
    rewards = self.reward[start:end - 1]
    name_codes = np.full([self.unroll_length], self.name_code, np.int32)
    state_action = StateAction(states, states.p0.controller, name_codes)
    is_resetting = np.full([self.unroll_length], False)
    is_resetting[0] = needs_reset
    frames = Frames(
        state_action=state_action,
        reward=rewards,
        is_resetting=is_resetting,
    )

    return Chunk(frames, ChunkMeta(start, end, self.info))

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
      compressed: bool = True,
      batch_size: int = 64,
      unroll_length: int = 64,
      extra_frames: int = 1,
      damage_ratio: float = 0.01,
      # None means all allowed.
      allowed_characters: Optional[list[melee.Character]] = None,
      allowed_opponents: Optional[list[melee.Character]] = None,
      balance_characters: bool = False,
      name_map: Optional[dict[str, int]] = None,
      observation_config: Optional[observations.ObservationConfig] = None,
  ):
    self.replays = replays
    self.batch_size = batch_size
    self.unroll_length = unroll_length
    self.chunk_size = unroll_length + extra_frames
    self.damage_ratio = damage_ratio
    self.compressed = compressed
    self.batch_counter = 0
    self.balance_characters = balance_characters

    def build_observation_filter():
      if observation_config is None:
        return None
      return observations.build_observation_filter(observation_config)

    self.allowed_characters = _charset(allowed_characters)
    self.allowed_opponents = _charset(allowed_opponents)
    self.name_map = name_map or {}
    self.encode_name = nametags.name_encoder(self.name_map)
    self.observation_config = observation_config

    self.replay_counter = 0
    replay_iter = self.iter_replays()
    self.managers = [
        TrajectoryManager(
            replay_iter,
            unroll_length=self.chunk_size,
            overlap=extra_frames,
            compressed=compressed,
            game_filter=self.is_allowed,
            observation_filter=build_observation_filter(),
            reward_kwargs=dict(damage_ratio=damage_ratio),
            encode_name=self.encode_name,
        ) for _ in range(batch_size)
    ]

  def iter_replays(self) -> Iterator[ReplayInfo]:
    replay_iter = itertools.cycle(self.replays)

    if self.balance_characters:
      # TODO: balance by opponent (i.e. matchup) too?
      by_character = collections.defaultdict(list)
      for replay in self.replays:
        by_character[replay.main_player.character].append(replay)

      num_per_character = {
          melee.Character(c).name: len(vs)
          for c, vs in by_character.items()
      }

      logging.info(f'Character balance: {num_per_character}')

      if len(by_character) > 1:
        iterators = [itertools.cycle(replays) for replays in by_character.values()]
        balanced_iterator = utils.interleave(*iterators)
        replay_iter = utils.interleave(balanced_iterator, replay_iter)
      else:
        logging.info("Only one character present, balancing not needed.")

    for replay in replay_iter:
      self.replay_counter += 1
      yield replay

  def is_allowed(self, game: Game) -> bool:
    # TODO: handle Zelda/Sheik transformation
    return (
        game.p0.character[0] in self.allowed_characters
        and
        game.p1.character[0] in self.allowed_opponents)

  def process_batch(self, chunks: list[Chunk]) -> Batch:
    batches: List[Batch] = []

    for chunk in chunks:
      batches.append(Batch(
          frames=chunk.frames,
          count=self.batch_counter,
          meta=chunk.meta))

    return utils.batch_nest_nt(batches)

  def __next__(self) -> Tuple[Batch, float]:
    batch: Batch = self.process_batch(
        [m.grab_chunk() for m in self.managers])
    # TODO: the epoch isn't quite correct if we are balancing replays
    epoch = self.replay_counter / len(self.replays)
    self.batch_counter += 1
    assert batch.frames.state_action.state.stage.shape[-1] == self.chunk_size
    assert batch.frames.reward.shape[-1] == self.chunk_size - 1
    return batch, epoch

def produce_batches(data_source_kwargs: dict, batch_queue: mp.Queue):
  data_source = DataSource(**data_source_kwargs)
  while True:
    batch_queue.put(next(data_source))

class DataSourceMP:
  def __init__(self, buffer=16, **kwargs):
    for k, v in kwargs.items():
      if k == 'replays':
        continue
      setattr(self, k, v)

    # 'spawn' uses much less memory than 'fork'
    context = mp.get_context('spawn')

    self.batch_queue = context.Queue(buffer)
    self.process = context.Process(
        target=produce_batches, args=(kwargs, self.batch_queue),
        name='DataSourceMP')
    self.process.start()

    atexit.register(self.batch_queue.close)
    atexit.register(self.process.terminate)

  def __next__(self) -> Tuple[Batch, float]:
    return self.batch_queue.get()

  def __del__(self):
    self.process.terminate()

class MultiDataSourceMP:

  def __init__(
      self,
      replays: List[ReplayInfo],
      num_workers: int,
      batch_size: int,
      **kwargs,
  ):
    if num_workers > len(replays):
      raise ValueError(
          f"num_workers ({num_workers}) must be less than the number of "
          f"replays ({len(replays)})")

    if batch_size % num_workers != 0:
      raise ValueError(
          f"batch_size ({batch_size}) must be divisible by num_workers "
          f"({num_workers})")

    self.sources: list[DataSourceMP] = []
    for i in range(num_workers):
      self.sources.append(DataSourceMP(
          replays=replays[i::num_workers],
          batch_size=batch_size // num_workers,
          **kwargs
      ))

    self.batch_size = batch_size
    for k in kwargs:
      setattr(self, k, kwargs[k])

  def __next__(self) -> Tuple[Batch, float]:
    results = [next(source) for source in self.sources]
    batches, epochs = zip(*results)
    return utils.concat_nest_nt(batches), np.mean(epochs)

class CachedDataSource(DataSource):
  """Guaranteed fast, useful for performance benchmarking."""

  @functools.cache
  def _get_batch(self) -> tuple[Batch, float]:
    return super().__next__()

  def __next__(self) -> Tuple[Batch, float]:
    return self._get_batch()

@dataclasses.dataclass
class DataConfig:
  batch_size: int = 32
  unroll_length: int = 64
  damage_ratio: float = 0.01
  compressed: bool = True
  num_workers: int = 0
  balance_characters: bool = False
  cached: bool = False

def make_source(
    num_workers: int,
    cached: bool = False,
    **kwargs):
  if num_workers == 0:
    if cached:
      return CachedDataSource(**kwargs)

    return DataSource(**kwargs)

  if num_workers == 1:
    return DataSourceMP(**kwargs)

  return MultiDataSourceMP(num_workers=num_workers, **kwargs)

def toy_data_source(**kwargs) -> DataSource:
  dataset_config = DatasetConfig(
      data_dir=str(paths.TOY_DATA_DIR),
      meta_path=str(paths.TOY_META_PATH),
  )
  return DataSource(
      replays=replays_from_meta(dataset_config),
      compressed=True,
      **kwargs,
  )
