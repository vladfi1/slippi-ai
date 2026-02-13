import abc
import atexit
import collections
import dataclasses
import functools
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
import random
from typing import (
    Any, Callable, Iterable, List, Optional, Set, Tuple, Iterator, NamedTuple,
    Union,
)
import typing as tp
import zlib
import zipfile

import numpy as np
import pyarrow
import pyarrow.parquet as pq

import melee

from slippi_ai import reward, utils, nametags, paths, observations
from slippi_ai.types import (
    S, Game, game_array_to_nt,
    BoolArray, FloatArray, Int32Array,
    # Re-exported for backward compatibility; canonical home is types.py.
    Action, NAME_DTYPE, StateAction, Frames,
)
from typing import Generic
from slippi_ai.mirror import mirror_game

from slippi_db import utils as file_utils
from slippi_db.utils import is_remote, FsspecFile

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
  zlib: bool

  @classmethod
  def from_metadata(cls, metadata: dict) -> 'ReplayMeta':
    raw = metadata['raw']
    return cls(
        p0=PlayerMeta.from_metadata(metadata['players'][0], raw),
        p1=PlayerMeta.from_metadata(metadata['players'][1], raw),
        stage=metadata['stage'],
        slp_md5=metadata['slp_md5'],
        zlib=metadata['compression'] == 'zlib',
    )

class ReplayInfo(NamedTuple):
  path: file_utils.LocalFile | str
  swap: bool
  meta: ReplayMeta

  mirror: bool = False

  @property
  def main_player(self) -> PlayerMeta:
    if isinstance(self.swap, np.ndarray):
      return utils.map_nt(
          lambda p0, p1: np.where(self.swap, p1, p0),
          self.meta.p0, self.meta.p1)
    return self.meta.p1 if self.swap else self.meta.p0

  def read_game(self) -> Game:
    if isinstance(self.path, str):
      with open(self.path, 'rb') as f:
        contents = f.read()
    else:
      contents = self.path.read()

    if self.meta.zlib:
      contents = zlib.decompress(contents)
    reader = pyarrow.BufferReader(contents)
    table = pq.read_table(reader)

    game_struct = table['root'].combine_chunks()
    game = game_array_to_nt(game_struct)

    if self.swap:
      game = swap_players(game)

    if self.mirror:
      game = mirror_game(game)

    return game

class ChunkMeta(NamedTuple):
  start: int
  end: int
  info: ReplayInfo


class Batch(NamedTuple, Generic[S]):
  game: Game[S]
  name: Int32Array[S]
  is_resetting: BoolArray[S]
  reward: FloatArray[S]
  meta: ChunkMeta

def _charset(chars: Optional[Iterable[melee.Character]]) -> Set[int]:
  if chars is None:
    chars = list(melee.Character)
  return set(c.value for c in chars)

ALL = 'all'
NONE = 'none'

# Within a dataset archive
GAMES_DIR = 'games'
META_PATH = 'meta.json'

@dataclasses.dataclass
class DatasetConfig:
  data_dir: Optional[str] = None  # required
  meta_path: Optional[str] = None
  archive: Optional[str] = None
  dataset_path: Optional[str] = None

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

  def validate(self):
    if self.dataset_path is not None:
      if self.data_dir is not None or self.meta_path is not None or self.archive is not None:
        logging.warning("dataset_path specified, ignoring data_dir, meta_path, and archive.")
    elif self.archive is not None:
      if not self.archive.endswith('.zip'):
        raise ValueError(f"Archive must be a .zip file, got: {self.archive}")

      # TODO: validate archive structure

      if self.data_dir is not None or self.meta_path is not None:
        logging.warning("Archive specified, ignoring data_dir and meta_path.")
    else:
      if self.data_dir is None:
        raise ValueError("Missing data_dir.")

      if self.meta_path is None:
        raise ValueError("Missing meta_path.")

  def read_meta(self) -> list[dict[str, Any]]:
    if self.dataset_path is not None:
      meta_uri = self.dataset_path.rstrip('/') + '/' + META_PATH
      if is_remote(self.dataset_path):
        return json.loads(FsspecFile(meta_uri).read().decode('utf-8'))
      else:
        with open(meta_uri) as f:
          return json.load(f)

    if self.archive is not None:
      meta_file = file_utils.ZipFile(self.archive, META_PATH)
      return json.loads(meta_file.read().decode('utf-8'))

    assert self.meta_path is not None
    with open(self.meta_path) as f:
      return json.load(f)

  def get_replay(self, slp_md5: str) -> str | file_utils.LocalFile:
    if self.dataset_path is not None:
      game_uri = self.dataset_path.rstrip('/') + '/' + GAMES_DIR + '/' + slp_md5
      if is_remote(self.dataset_path):
        return FsspecFile(game_uri)
      else:
        return game_uri

    if self.archive is not None:
      return file_utils.ZipFile(self.archive, GAMES_DIR + '/' + slp_md5)

    assert self.data_dir is not None
    return os.path.join(self.data_dir, slp_md5)

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
  config.validate()

  replays = []

  meta_rows = config.read_meta()

  allowed_characters = _charset(chars_from_string(config.allowed_characters))
  allowed_opponents = _charset(chars_from_string(config.allowed_opponents))
  name_filter = create_name_filter(config.allowed_names, config.banned_names)

  banned_counts = collections.Counter()

  for row in meta_rows:
    replay_meta = ReplayMeta.from_metadata(row)
    replay_path = config.get_replay(replay_meta.slp_md5)

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

  if banned_counts:
    print('Banned names:', banned_counts)

  return replays


def train_test_split(
    config: DatasetConfig,
) -> Tuple[List[ReplayInfo], List[ReplayInfo]]:
  # For dataset_path and archive modes, we get replays entirely from metadata
  # (no os.listdir verification, which isn't available for remote/archive).
  if config.dataset_path is not None or config.archive is not None:
    replays = replays_from_meta(config)
  else:
    if config.data_dir is None:
      raise ValueError("data_dir must be specified in DatasetConfig")

    filenames = sorted(os.listdir(config.data_dir))
    logging.info(f"Found {len(filenames)} files.")

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

      replays: list[ReplayInfo] = []

      for filename in filenames:
        replay_path = os.path.join(config.data_dir, filename)
        replays.append(ReplayInfo(replay_path, False))
        if config.swap:
          replays.append(ReplayInfo(replay_path, True))

  # TODO: stable partition
  if len(replays) < 2:
    raise ValueError("Not enough replays found.")

  rng = random.Random(config.seed)
  rng.shuffle(replays)

  # Ensure at least one train and one test replay.
  num_test = 1 + math.ceil(config.test_ratio * (len(replays) - 2))

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

Rank1 = tuple[int]

def game_len(game: Game[Rank1]) -> int:
  return game.stage.shape[0]

class TrajectoryManager:
  # TODO: manage recurrent state? can also do it in the learner

  def __init__(
      self,
      source: Iterator[ReplayInfo],
      unroll_length: int,
      encode_name: Callable[[str], int],
      overlap: int = 1,
      compressed: bool = True,
      game_filter: Optional[Callable[[Game[Rank1]], bool]] = None,
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

    self.needs_game = True

  def find_game(self):
    while True:
      info = next(self.source)
      game = info.read_game()
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
    self.needs_game = False

  def grab_chunk(self) -> Batch[Rank1]:
    """Grabs a chunk from a trajectory."""
    # TODO: write a unit test for this

    needs_reset = (
        self.needs_game or
        self.frame + self.unroll_length > game_len(self.game))

    if needs_reset:
      self.find_game()

    start = self.frame
    end = start + self.unroll_length
    slice = lambda a: a[start:end]
    # faster than tree.map_structure
    # states = utils.map_nt(slice, self.game)
    states = utils.cached_map_nt(Game)(slice, self.game)
    self.frame = end - self.overlap

    # Rewards could be deferred to the learner.
    rewards = self.reward[start:end - 1]
    name = np.full([self.unroll_length], self.name_code, np.int32)
    is_resetting = np.full([self.unroll_length], False)
    is_resetting[0] = needs_reset

    return Batch(
        game=states,
        name=name,
        is_resetting=is_resetting,
        reward=rewards,
        meta=ChunkMeta(start, end, self.info),
    )

def swap_players(game: Game[S]) -> Game[S]:
  return game._replace(p0=game.p1, p1=game.p0)

# TODO: this is redundant with ReplayInfo.read_game, but used in some places
def read_table(path: str, compressed: bool) -> Game[Rank1]:
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

Shape = tp.TypeVarTuple('Shape')

Rank2 = tuple[int, int]

class AbstractDataSource(abc.ABC):

  @abc.abstractmethod
  def __next__(self) -> tuple[Batch[Rank2], float]:
    """Returns the next batch and epoch number."""

  def shutdown(self):
    """Cleans up any resources used by the data source."""

  @property
  @abc.abstractmethod
  def batch_size(self) -> int:
    """Returns the batch size used by the data source."""

class DataSource(AbstractDataSource):
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
    self._batch_size = batch_size
    self.unroll_length = unroll_length
    self.chunk_size = unroll_length + extra_frames
    self.damage_ratio = damage_ratio
    self.compressed = compressed

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

  @property
  def batch_size(self) -> int:
    return self._batch_size

  def iter_replays(self) -> Iterator[ReplayInfo]:
    replay_iter = utils.cycle_iterable(self.replays)

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

  def is_allowed(self, game: Game[Rank1]) -> bool:
    # TODO: handle Zelda/Sheik transformation
    return (
        game.p0.character[0] in self.allowed_characters
        and
        game.p1.character[0] in self.allowed_opponents)

  def process_batch(self, batches: list[Batch[Rank1]]) -> Batch[Rank2]:
    return utils.cached_zip_map_nt(Batch)(np.stack, batches)  # type: ignore

  def __next__(self) -> Tuple[Batch[Rank2], float]:
    batch = self.process_batch(
        [m.grab_chunk() for m in self.managers])
    # TODO: the epoch isn't quite correct if we are balancing replays
    epoch = self.replay_counter / len(self.replays)
    assert batch.game.stage.shape[-1] == self.chunk_size
    assert batch.reward.shape[-1] == self.chunk_size - 1
    return batch, epoch

class TimeBatchedDataSource(AbstractDataSource):

  def __init__(
      self,
      unroll_chunks: int,
      unroll_length: int,
      extra_frames: int = 1,
      **kwargs,
  ):
    self.data_source = DataSource(
        unroll_length=unroll_chunks * unroll_length,
        extra_frames=extra_frames,
        **kwargs)
    self.unroll_chunks = unroll_chunks
    self.unroll_length = unroll_length
    self.extra_frames = extra_frames
    self._current_index = unroll_chunks

  def __next__(self) -> tuple[Batch[Rank2], float]:
    if self._current_index == self.unroll_chunks:
      self._current_batch_and_epoch = next(self.data_source)
      self._current_index = 0

    batch, epoch = self._current_batch_and_epoch

    start = self._current_index * self.unroll_length
    end = start + self.unroll_length + self.extra_frames
    slice = lambda a: a[:, start:end]

    self._current_index += 1

    return Batch(
        game=utils.cached_map_nt(Game)(slice, batch.game),
        name=slice(batch.name),
        is_resetting=slice(batch.is_resetting),
        reward=batch.reward[:, start:end - 1],
        meta=ChunkMeta(
            start=batch.meta.start + start,
            end=batch.meta.start + end,
            info=batch.meta.info,
        ),
    ), epoch

  def shutdown(self):
    self.data_source.shutdown()

  @property
  def batch_size(self) -> int:
    return self.data_source.batch_size


def produce_batches(data_source_kwargs: dict, batch_queue: mp.Queue):
  data_source = make_source(num_workers=0, **data_source_kwargs)
  while True:
    batch_queue.put(next(data_source))


class DataSourceMP(AbstractDataSource):
  def __init__(self, buffer=16, **kwargs):
    self._batch_size = kwargs['batch_size']

    # 'spawn' uses much less memory than 'fork'
    context = mp.get_context('spawn')

    self.batch_queue = context.Queue(buffer)
    self.process = context.Process(
        target=produce_batches, args=(kwargs, self.batch_queue),
        name='DataSourceMP')
    self.process.start()

    atexit.register(self.shutdown)

  @property
  def batch_size(self) -> int:
    return self._batch_size

  def shutdown(self):
    self.batch_queue.close()
    self.process.terminate()

  def __next__(self) -> tuple[Batch[Rank2], float]:
    return self.batch_queue.get()

  def __del__(self):
    self.shutdown()

class MultiDataSourceMP(AbstractDataSource):

  def __init__(
      self,
      replays: List[ReplayInfo],
      num_workers: int,
      batch_size: int,
      **kwargs,
  ):
    if num_workers > len(replays):
      num_workers = len(replays)
      logging.warning(
          f"num_workers reduced to {num_workers} since there are only "
          f"{len(replays)} replays.")

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

    self._batch_size = batch_size

  @property
  def batch_size(self) -> int:
    return self._batch_size

  def __next__(self) -> tuple[Batch[Rank2], float]:
    results = [next(source) for source in self.sources]
    batches, epochs = zip(*results)
    epoch = np.mean(epochs)
    return utils.cached_zip_map_nt(Batch)(np.concatenate, batches), epoch

  def shutdown(self):
    for source in self.sources:
      source.shutdown()

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
  unroll_chunks: int = 0

def make_source(
    num_workers: int,
    cached: bool = False,
    **kwargs):
  if num_workers == 0:
    if cached:
      return CachedDataSource(**kwargs)

    unroll_chunks: int = kwargs.pop('unroll_chunks')
    if unroll_chunks > 0:
      return TimeBatchedDataSource(unroll_chunks=unroll_chunks, **kwargs)

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
