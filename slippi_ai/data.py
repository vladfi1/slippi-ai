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
import shutil
from typing import (
    Any, Callable, Iterable, List, Optional, Set, Tuple, Iterator, NamedTuple,
)
import typing as tp
import zlib

import fsspec
import numpy as np
import pyarrow
import pyarrow.parquet as pq
import webdataset as wds

import melee

from slippi_ai import reward, utils, nametags, paths, observations
from slippi_ai.types import (
    S, Game, game_array_to_nt, array_from_nt,
    BoolArray, FloatArray, Int32Array, Rank1,
    # Re-exported for backward compatibility; canonical home is types.py.
    Action, NAME_DTYPE, StateAction, Frames,
)
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

  def swap_players(self) -> tp.Self:
    return self._replace(p0=self.p1, p1=self.p0)

class ReplayInfo(NamedTuple):
  path: file_utils.LocalFile | str
  swap: bool
  meta: ReplayMeta

  mirror: bool = False

  @classmethod
  def init(
      cls,
      path: file_utils.LocalFile | str,
      swap: bool,
      meta: ReplayMeta,
      mirror: bool = False,
  ) -> tp.Self:
    if swap:
      meta = meta.swap_players()
    return cls(path=path, swap=swap, meta=meta, mirror=mirror)

  @property
  def main_player(self) -> PlayerMeta:
    return self.meta.p0

  def read_raw(self) -> bytes:
    if isinstance(self.path, str):
      with open(self.path, 'rb') as f:
        contents = f.read()
    else:
      contents = self.path.read()
    return contents

  def read_pq(self) -> pyarrow.StructArray:
    contents = self.read_raw()

    if self.meta.zlib:
      contents = zlib.decompress(contents)
    reader = pyarrow.BufferReader(contents)
    table = pq.read_table(reader)

    return table['root'].combine_chunks()

  def read_game(self) -> Game[Rank1]:
    game_struct = self.read_pq()
    game = game_array_to_nt(game_struct)

    if self.swap:
      game = swap_players(game)

    if self.mirror:
      game = mirror_game(game)

    return game

class WdsReplayInfo(tp.NamedTuple):
  """A replay decoded from a WebDataset sample."""
  meta: ReplayMeta
  game: Game[Rank1]

  @property
  def main_player(self) -> PlayerMeta:
    return self.meta.p0

  def read_game(self) -> Game[Rank1]:
    return self.game


def _parse_wds_sample(sample: dict) -> WdsReplayInfo:
  """Decode a WebDataset sample dict into a WdsReplayInfo."""
  meta_dict = json.loads(sample['meta'].decode('utf-8'))

  content: bytes = sample['game']
  if meta_dict['zlib']:
    content = zlib.decompress(content)

  reader = pyarrow.BufferReader(content)
  table = pq.read_table(reader)
  game_struct = table['root'].combine_chunks()
  game = game_array_to_nt(game_struct)

  p0 = PlayerMeta(**meta_dict['p0'])
  p1 = PlayerMeta(**meta_dict['p1'])
  if meta_dict['swap']:
    p0, p1 = p1, p0
    game = swap_players(game)

  replay_meta = ReplayMeta(
      p0=p0, p1=p1,
      stage=meta_dict['stage'],
      slp_md5=meta_dict['slp_md5'],
      zlib=meta_dict['zlib'],  # not really used
  )
  return WdsReplayInfo(
    meta=replay_meta,
    game=game)


class ChunkMeta(NamedTuple):
  start: int
  end: int
  meta: ReplayMeta

class Batch(NamedTuple, tp.Generic[S]):
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
  filter_opponent_name: bool = False

  swap: bool = True  # yield swapped versions of each replay
  mirror: bool = False  # mirror left/right in each replay
  seed: int = 0

  wds_path: Optional[str] = None

  def validate(self):
    if self.dataset_path is not None:
      if self.data_dir is not None or self.meta_path is not None or self.archive is not None:
        logging.warning("dataset_path specified, ignoring data_dir, meta_path, and archive.")
      self.data_dir = self.dataset_path.rstrip('/') + '/' + GAMES_DIR
      self.meta_path = self.dataset_path.rstrip('/') + '/' + META_PATH
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
        if not os.path.isfile(game_uri):
          raise FileNotFoundError(f"Replay file not found: {game_uri}")
        return game_uri

    if self.archive is not None:
      return file_utils.ZipFile(self.archive, GAMES_DIR + '/' + slp_md5)

    assert self.data_dir is not None
    game_path = os.path.join(self.data_dir, slp_md5)
    if not os.path.isfile(game_path):
      raise FileNotFoundError(f"Replay file not found: {game_path}")
    return game_path

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
    if nametags.is_banned_name(name):
      return False

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

      replays.append(ReplayInfo.init(replay_path, False, replay_meta))

      continue

    for swap in [False, True]:
      players = [replay_meta.p0, replay_meta.p1]
      if swap:
        players = reversed(players)
      p0, p1 = players

      if (p0.character not in allowed_characters
          or p1.character not in allowed_opponents):
        continue

      if not name_filter(p0.name):
        banned_counts[p0.name] += 1
        continue

      if config.filter_opponent_name and not name_filter(p1.name):
        banned_counts[p1.name] += 1
        continue

      replays.append(ReplayInfo.init(replay_path, swap, replay_meta))

  if banned_counts:
    print('Banned names:', banned_counts)

  return replays


def train_test_split(
    config: DatasetConfig,
) -> Tuple[List[ReplayInfo], List[ReplayInfo]]:
  replays = replays_from_meta(config)

  if len(replays) == 0:
    raise ValueError("No replays found with the given configuration.")

  if len(replays) == 1:
    logging.warning("Only one replay found, using it for both train and test.")
    return replays, replays

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

def _replay_info_to_wds(info: ReplayInfo) -> dict:
  meta = info.meta
  return {
      'p0': {'character': meta.p0.character, 'name': meta.p0.name},
      'p1': {'character': meta.p1.character, 'name': meta.p1.name},
      'stage': meta.stage,
      'slp_md5': meta.slp_md5,
      'swap': info.swap,
      'zlib': meta.zlib,
  }

WDS_META = 'wds_meta.json'
WDS_SHARD_EXT = '.tar'
# WDS_SHARD_EXT = '.tar.xz'
SPLITS = ('train', 'test')

def _wds_shard_pattern(split: str):
  if split not in SPLITS:
    raise ValueError(f"Invalid split: {split}")
  return f'{split}-%06d' + WDS_SHARD_EXT

def wds_glob_pattern(split: str):
  if split not in SPLITS:
    raise ValueError(f"Invalid split: {split}")
  return f'{split}-*' + WDS_SHARD_EXT

@dataclasses.dataclass
class ShardWriterConfig:
  maxcount: int = 100000
  maxsize: int = int(1e9)

def write_wds_shards(
    config: DatasetConfig,
    output_dir: str,
    shard_writer_config: ShardWriterConfig = ShardWriterConfig(),
):
  config.validate()

  if config.mirror:
    raise ValueError("Mirror when reading instead of writing.")

  if config.meta_path is None:
    raise ValueError("meta_path must be specified in DatasetConfig to write WDS shards.")

  shutil.copy2(config.meta_path, os.path.join(output_dir, META_PATH))

  # TODO: don't duplicate swapped replays
  train_test = train_test_split(config)

  sizes = {}
  name_counts = collections.Counter()
  character_counts = collections.Counter()

  for split, replays in zip(SPLITS, train_test):
    sizes[split] = len(replays)

    shard_path = os.path.join(output_dir, _wds_shard_pattern(split))
    with wds.writer.ShardWriter(shard_path, **dataclasses.asdict(shard_writer_config)) as sink:
      import tqdm
      for replay_info in tqdm.tqdm(replays, unit='replay', desc=f'Writing {split} shards'):
        if config.swap:
          players = [replay_info.main_player]
        else:
          players = [replay_info.meta.p0, replay_info.meta.p1]

        for player in players:
          name_counts[player.name] += 1
          character_counts[player.character] += 1

        # game_pq = replay_info.read_pq()
        # parquet_bytes = parsing_utils.convert_game(game_pq)  # no compression

        md5 = replay_info.meta.slp_md5
        key = md5 + ('_swap' if replay_info.swap else '')

        sink.write({
            '__key__': key,
            'game': replay_info.read_raw(),  # already compressed if needed
            'meta': json.dumps(_replay_info_to_wds(replay_info), indent=2).encode('utf-8'),
        })

  wds_meta = dict(
      sizes=sizes,
      name_counts=name_counts,
      character_counts=character_counts,
  )
  with open(os.path.join(output_dir, WDS_META), 'w') as f:
    json.dump(wds_meta, f, indent=2)

def game_len(game: Game[Rank1]) -> int:
  return game.stage.shape[0]

class TrajectoryManager:
  # TODO: manage recurrent state? can also do it in the learner

  def __init__(
      self,
      source: Iterator[ReplayInfo] | Iterator[WdsReplayInfo],
      unroll_length: int,
      encode_name: Callable[[str], int],
      overlap: int = 1,
      game_filter: Optional[Callable[[Game[Rank1]], bool]] = None,
      observation_filter: Optional[observations.ObservationFilter] = None,
      reward_kwargs: dict = {},
  ):
    self.source = source
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

      break

    self.reward = reward.compute_rewards(game, **self.reward_kwargs)

    if self.observation_filter is not None:
      self.observation_filter.reset()
      game = self.observation_filter.filter_time(game)

    self.flat_game = utils.cached_flatten(Game)(game)
    self.game_len = game_len(game)
    self.frame = 0
    self.info = info
    self.flat_meta = utils.cached_flatten(ReplayMeta)(info.meta)
    self.name_code = self.encode_name(info.main_player.name)
    self.needs_game = False

  def grab_chunk(self) -> list:
    """Grabs a chunk from a trajectory."""
    # TODO: write a unit test for this

    needs_reset = (
        self.needs_game or
        self.frame + self.unroll_length > self.game_len)

    if needs_reset:
      self.find_game()

    start = self.frame
    end = start + self.unroll_length
    flat_game = [x[start:end] for x in self.flat_game]
    self.frame = end - self.overlap

    # Rewards could be deferred to the learner.
    rewards = self.reward[start:end - 1]
    name = np.full([self.unroll_length], self.name_code, np.int32)
    is_resetting = np.full([self.unroll_length], False)
    is_resetting[0] = needs_reset

    # Flat Batch
    flat_result = flat_game
    flat_result.append(name)
    flat_result.append(is_resetting)
    flat_result.append(rewards)
    # Flat ChunkMeta
    flat_result.append(start)
    flat_result.append(end)
    flat_result.extend(self.flat_meta)

    return flat_result

Rank2 = tuple[int, int]

def process_flat_batches(flat_batches: list) -> Batch[Rank2]:
  batched_flat = [np.stack(xs) for xs in zip(*flat_batches)]
  return utils.cached_unflatten(Batch, batched_flat)


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

def read_wds_meta(dataset_path: str) -> dict:
  with fsspec.open(os.path.join(dataset_path, WDS_META)) as f:
    return json.load(f)

@dataclasses.dataclass
class WebDataConfig:
  shuffle_buffer_size: int = 1000
  cache_dir: Optional[str] = None
  verbose: bool = True

# TODO: support mirroring
class WebDataSource(AbstractDataSource):

  def __init__(
      self,
      dataset_path: str,
      split: str,
      batch_size: int,
      unroll_length: int,
      extra_frames: int = 1,
      damage_ratio: float = 0.01,
      name_map: Optional[dict[str, int]] = None,
      observation_config: Optional[observations.ObservationConfig] = None,
      shuffle_buffer_size: int = 1000,
      cache_dir: Optional[str] = None,
      verbose: bool = True,
  ):
    self.dataset_path = dataset_path
    self.split = split
    self._batch_size = batch_size
    self.chunk_size = unroll_length + extra_frames
    self.shuffle_buffer_size = shuffle_buffer_size
    self.cache_dir = cache_dir
    self.name_map = name_map or {}
    self.encode_name = nametags.name_encoder(self.name_map)
    self.wds_meta = read_wds_meta(dataset_path)
    self.num_replays = self.wds_meta['sizes'][split]
    self.verbose = verbose

    def build_observation_filter():
      if observation_config is None:
        return None
      return observations.build_observation_filter(observation_config)

    self.replay_counter = 0
    replay_iter = self.iter_replays()
    self.managers = [
        TrajectoryManager(
            replay_iter,
            unroll_length=self.chunk_size,
            overlap=extra_frames,
            observation_filter=build_observation_filter(),
            reward_kwargs=dict(damage_ratio=damage_ratio),
            encode_name=self.encode_name,
        ) for _ in range(batch_size)
    ]

  def iter_replays(self) -> Iterator[WdsReplayInfo]:
    fs: fsspec.AbstractFileSystem
    fs, ds_path = fsspec.core.url_to_fs(self.dataset_path)
    shards = fs.glob(os.path.join(ds_path, wds_glob_pattern(self.split)))
    shards = tp.cast(list[str], shards)

    def to_url(shard: str) -> str:
      path = os.path.join(self.dataset_path, os.path.basename(shard))
      if 's3' in fs.protocol:
        return f"pipe:s3cmd get --quiet {path} -"
      return path

    shard_urls = [to_url(shard) for shard in shards]
    if self.verbose:
      print(shard_urls)

    cache_dir = self.cache_dir
    if cache_dir is not None:
      cache_dir = os.path.join(cache_dir, ds_path, self.split)
      os.makedirs(cache_dir, exist_ok=True)
    else:
      logging.warning("No cache_dir specified for WebDataSource.")

    dataset = wds.compat.WebDataset(
        shard_urls, shardshuffle=100,
        cache_dir=cache_dir,
        verbose=self.verbose,
    )

    pipeline = (
        dataset
        .repeat()
        .shuffle(self.shuffle_buffer_size)
        .map(_parse_wds_sample)
    )
    for item in pipeline:
      self.replay_counter += 1
      yield item

  @property
  def batch_size(self) -> int:
    return self._batch_size

  def __next__(self) -> Tuple[Batch[Rank2], float]:
    batch = process_flat_batches(
        [m.grab_chunk() for m in self.managers])
    epoch = self.replay_counter / self.num_replays
    assert batch.game.stage.shape[-1] == self.chunk_size
    assert batch.reward.shape[-1] == self.chunk_size - 1
    return batch, epoch

class DataSource(AbstractDataSource):
  def __init__(
      self,
      replays: List[ReplayInfo],
      batch_size: int = 64,
      unroll_length: int = 64,
      extra_frames: int = 1,
      damage_ratio: float = 0.01,
      balance_characters: bool = False,
      name_map: Optional[dict[str, int]] = None,
      observation_config: Optional[observations.ObservationConfig] = None,
  ):
    self.replays = replays
    self._batch_size = batch_size
    self.unroll_length = unroll_length
    self.chunk_size = unroll_length + extra_frames
    self.damage_ratio = damage_ratio

    self.balance_characters = balance_characters

    def build_observation_filter():
      if observation_config is None:
        return None
      return observations.build_observation_filter(observation_config)

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

  def __next__(self) -> Tuple[Batch[Rank2], float]:
    batch = process_flat_batches(
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
    self.data_source = make_source(
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
    slice = lambda x: x[:, start:end]

    self._current_index += 1

    return Batch(
        game=utils.cached_map_nt(Game)(slice, batch.game),
        name=slice(batch.name),
        is_resetting=slice(batch.is_resetting),
        reward=batch.reward[:, start:end - 1],
        meta=ChunkMeta(
            start=batch.meta.start + start,
            end=batch.meta.start + end,
            meta=batch.meta.meta,
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
      num_workers: int = 1,
      batch_size: int = 64,
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

    self.sources = []
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

class CachedDataSource(AbstractDataSource):
  """Guaranteed fast, useful for performance benchmarking."""

  def __init__(self, source: AbstractDataSource):
    self.source = source
    self.counter = 0
    self._get_batch = functools.cache(self.source.__next__)

  @property
  def batch_size(self) -> int:
    return self.source.batch_size

  def __next__(self) -> Tuple[Batch, float]:
    batch = self._get_batch()[0]
    self.counter += 1
    return batch, self.counter

@dataclasses.dataclass
class DataConfig:
  batch_size: int = 32
  unroll_length: int = 64
  damage_ratio: float = 0.01
  num_workers: int = 0
  buffer: int = 16  # DataSourceMP buffer size
  balance_characters: bool = False
  cached: bool = False
  unroll_chunks: int = 0

  wds: WebDataConfig = dataclasses.field(default_factory=WebDataConfig)

# TODO: this kwarg manipulation is a bit messy
# There is an asymmetry between the DataSource and WebDataSource code paths:
# DataSource assumes that the ReplayInfo objects have already been produced,
# while WebDataSource produces WdsReplayInfo objects on the fly from the WebDataset.
def make_source(
    num_workers: int = 0,
    cached: bool = False,
    **kwargs,  # should contain "replays"
) -> AbstractDataSource:
  is_ds = 'replays' in kwargs
  is_wds = 'dataset_path' in kwargs

  if is_ds and is_wds:
    raise ValueError("Cannot specify both replays and dataset_path.")

  if num_workers > 1 and is_wds:
    logging.warning("num_workers > 1 not supported for WebDataSource, setting to 1.")
    num_workers = 1

  if num_workers == 1:
    return DataSourceMP(**kwargs)
  elif num_workers > 1:
    return MultiDataSourceMP(num_workers=num_workers, **kwargs)

  if 'buffer' in kwargs:
    del kwargs['buffer']

  unroll_chunks: int = kwargs.pop('unroll_chunks', 0)

  if unroll_chunks > 0:
    return TimeBatchedDataSource(unroll_chunks=unroll_chunks, **kwargs)

  if is_ds:
    del kwargs['wds']
    source = DataSource(**kwargs)
  elif is_wds:
    del kwargs['balance_characters']  # not supported
    wds: dict = kwargs.pop('wds')  # already converted by dataclasses.asdict
    source = WebDataSource(**kwargs, **wds)
  else:
    raise ValueError("Must specify either replays or dataset_path.")

  if cached:
    source = CachedDataSource(source)

  return source

def toy_data_source(**kwargs) -> DataSource:
  dataset_config = DatasetConfig(
      data_dir=str(paths.TOY_DATA_DIR),
      meta_path=str(paths.TOY_META_PATH),
  )
  return DataSource(
      replays=replays_from_meta(dataset_config),
      **kwargs,
  )

class Sources(NamedTuple):
  train: AbstractDataSource
  test: AbstractDataSource
  name_map: dict[str, int]

def build_sources(
    dataset_config: DatasetConfig,
    train_data_config: DataConfig,
    test_data_config: Optional[DataConfig] = None,
    name_map: Optional[dict[str, int]] = None,
    max_names: Optional[int] = None,
    **kwargs,  # observation_config, extra_frames
) -> Sources:
  test_data_config = test_data_config or train_data_config

  if name_map is None and max_names is None:
    raise ValueError('Must specify max_names if name_map is not given.')

  if dataset_config.wds_path is not None:
    wds_meta = read_wds_meta(dataset_config.wds_path)

    # TODO: check opponents too
    allowed_characters = _charset(chars_from_string(dataset_config.allowed_characters))
    for char_str, _ in wds_meta['character_counts'].items():
      char_int = int(char_str)
      if char_int not in allowed_characters:
        raise ValueError(f"Character {melee.Character(char_int).name} present in dataset but not allowed by config.")

    train_size = wds_meta['sizes']['train']
    test_size = wds_meta['sizes']['test']
    logging.info(f'Training on {train_size} replays, testing on {test_size} replays from WebDataset at {dataset_config.wds_path}.')

    if name_map is None:
      assert max_names is not None
      name_map = nametags.name_map_from_counts(
          wds_meta['name_counts'], max_names=max_names)

    sources = {}

    for split, data_config in zip(SPLITS, [train_data_config, test_data_config]):
      sources[split] = make_source(
          dataset_path=dataset_config.wds_path,
          split=split,
          name_map=name_map,
          **dataclasses.asdict(data_config),
          **kwargs,
      )

    return Sources(**sources, name_map=name_map)

  train_and_test = train_test_split(dataset_config)
  train_replays, test_replays = train_and_test
  logging.info(f'Training on {len(train_replays)} replays, testing on {len(test_replays)}')


  if name_map is None:
    assert max_names is not None

    names = []
    for split_replays in train_and_test:
      for replay in split_replays:
        names.append(replay.meta.p0.name)
        if not dataset_config.swap:
          names.append(replay.meta.p1.name)
    name_map = nametags.name_map_from_entries(names, max_names)

  sources = {}

  for split, split_replays, data_config in zip(
      SPLITS, train_and_test, [train_data_config, test_data_config]):
    sources[split] = make_source(
        replays=split_replays,
        name_map=name_map,
        **dataclasses.asdict(data_config),
        **kwargs,
    )

  return Sources(**sources, name_map=name_map)
