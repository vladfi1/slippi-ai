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
import pickle
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

import melee

from slippi_ai import reward, utils, nametags, paths, observations, datasets
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

class Replay(tp.NamedTuple):
  """A replay with metadata."""
  meta: ReplayMeta
  game: Game[Rank1]

  @property
  def main_player(self) -> PlayerMeta:
    return self.meta.p0

  def read_game(self) -> Game[Rank1]:
    return self.game

  def mirror(self) -> tp.Self:
    return self._replace(game=mirror_game(self.game))

class ReplayInfo(NamedTuple):
  path: file_utils.LocalFile | str
  swap: bool
  meta: ReplayMeta  # already swapped if swap=True

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

  def to_replay(self) -> Replay:
    return Replay(meta=self.meta, game=self.read_game())


def _parse_wds_sample(sample: dict) -> Replay:
  """Decode a WebDataset sample dict into a Replay."""
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
  return Replay(
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

class BatchWithMeta(NamedTuple, tp.Generic[S]):
  batch: Batch[S]
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
  import webdataset as wds

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
      source: Iterator[ReplayInfo] | Iterator[Replay],
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
    # self.flat_meta = utils.cached_flatten(ReplayMeta)(info.meta)
    self.name_code = self.encode_name(info.main_player.name)
    self.needs_game = False

  def grab_chunk(self) -> tuple[list[np.ndarray], ChunkMeta]:
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

    flat_batch = flat_game
    flat_batch.extend([name, is_resetting, rewards])

    meta = ChunkMeta(start=start, end=end, meta=self.info.meta)

    return flat_batch, meta

Rank2 = tuple[int, int]


class BatchAccumulator:
  """Pre-allocates output buffers to avoid per-batch numpy allocations.

  Replaces np.stack([chunk[j] for chunk in chunks]) for each leaf j with
  in-place writes: buf[j][i] = chunk[j], reusing the same memory each batch.

  Note: returned buffers are reused across calls; do not retain references
  to batch arrays across multiple __next__ calls.
  """

  def __init__(self, batch_size: int):
    self._batch_size = batch_size
    self._needs_init = True

  def _init(self, unbatched: tp.Sequence[list[np.ndarray]]):
    assert len(unbatched) == self._batch_size
    assert self._needs_init
    self._needs_init = False

    prototype = unbatched[0]
    self._bufs = []
    for x in prototype:
      assert isinstance(x, np.ndarray)
      self._bufs.append(np.empty((self._batch_size, *x.shape), dtype=x.dtype))

  def collect(self, unbatched: tp.Sequence[list[np.ndarray]]) -> list[np.ndarray]:
    """Fill pre-allocated buffers from managers; return batched flat list."""
    if self._needs_init:
      self._init(unbatched)

    for i, xs in enumerate(unbatched):
      for buf, x in zip(self._bufs, xs):
        buf[i] = x

    return self._bufs


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
  def __next__(self) -> tuple[BatchWithMeta[Rank2], float]:
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
      num_workers: int = 0,
      buffer: int = 16,
      mirror: bool = False,
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

    self._accumulator = BatchAccumulator(batch_size)

    def build_observation_filter():
      if observation_config is None:
        return None
      return observations.build_observation_filter(observation_config)

    self.replay_counter = 0

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

    import webdataset as wds

    dataset = wds.compat.WebDataset(
        shard_urls,  # TODO: shuffle urls?
        shardshuffle=False,
        cache_dir=cache_dir,
        verbose=self.verbose,
    )

    pipeline = (
        dataset
        .repeat()
    )

    sample_ds = datasets.IteratorDataset(iter(pipeline))

    if num_workers > 0:
      replay_ds = sample_ds.map_mp(
          _parse_wds_sample,
          num_workers=num_workers, buffer=buffer)
    else:
      replay_ds = sample_ds.map(_parse_wds_sample)

    if mirror:
      replay_ds = replay_ds.map_iter(
          lambda replay: (replay, replay.mirror())
      )
      self.num_replays *= 2

    self.replay_ds = replay_ds.shuffle(self.shuffle_buffer_size)

    def iter_replays():
      for replay in replay_ds:
        self.replay_counter += 1
        yield replay

    replay_iter = iter_replays()

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

  def shutdown(self):
    self.replay_ds.stop()

  @property
  def batch_size(self) -> int:
    return self._batch_size

  def __next__(self) -> Tuple[BatchWithMeta[Rank2], float]:
    unbatched_flat, metas = zip(*(m.grab_chunk() for m in self.managers))

    batched_flat = self._accumulator.collect(unbatched_flat)
    batch = utils.cached_unflatten(Batch, batched_flat)

    meta = utils.cached_zip_map_nt(ChunkMeta)(np.stack, metas)
    batch_with_meta = BatchWithMeta(batch=batch, meta=meta)

    epoch = self.replay_counter / self.num_replays
    assert batch.game.stage.shape[-1] == self.chunk_size
    assert batch.reward.shape[-1] == self.chunk_size - 1
    return batch_with_meta, epoch

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
      num_workers: int = 0,
      buffer: int = 16,
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
    replay_iter = self.iter_replay_infos()

    replay_info_ds = datasets.IteratorDataset(replay_iter)

    if num_workers > 0:
      self.replay_ds = datasets.MPMap(
          replay_info_ds, map_fn=ReplayInfo.to_replay,
          num_workers=num_workers, buffer=buffer)
    else:
      self.replay_ds = datasets.MapDataset(replay_info_ds, map_fn=ReplayInfo.to_replay)

    self.managers = [
        TrajectoryManager(
            self.replay_ds,
            unroll_length=self.chunk_size,
            overlap=extra_frames,
            observation_filter=build_observation_filter(),
            reward_kwargs=dict(damage_ratio=damage_ratio),
            encode_name=self.encode_name,
        ) for _ in range(batch_size)
    ]
    self._accumulator = BatchAccumulator(batch_size)

  @property
  def batch_size(self) -> int:
    return self._batch_size

  def shutdown(self):
    self.replay_ds.stop()

  def iter_replay_infos(self) -> Iterator[ReplayInfo]:
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

  def __next__(self) -> Tuple[BatchWithMeta[Rank2], float]:
    unbatched_flat, metas = zip(*(m.grab_chunk() for m in self.managers))

    batched_flat = self._accumulator.collect(unbatched_flat)
    batch = utils.cached_unflatten(Batch, batched_flat)

    meta = utils.cached_zip_map_nt(ChunkMeta)(np.stack, metas)
    batch_with_meta = BatchWithMeta(batch=batch, meta=meta)

    # TODO: the epoch isn't quite correct if we are balancing replays
    epoch = self.replay_counter / len(self.replays)
    assert batch.game.stage.shape[-1] == self.chunk_size
    assert batch.reward.shape[-1] == self.chunk_size - 1
    return batch_with_meta, epoch


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

  def __next__(self) -> tuple[BatchWithMeta[Rank2], float]:
    if self._current_index == self.unroll_chunks:
      self._current_batch_and_epoch = next(self.data_source)
      self._current_index = 0

    batch_with_meta, epoch = self._current_batch_and_epoch
    batch = batch_with_meta.batch

    start = self._current_index * self.unroll_length
    end = start + self.unroll_length + self.extra_frames
    slice = lambda x: x[:, start:end]

    self._current_index += 1

    return BatchWithMeta(
        batch=Batch(
            game=utils.cached_map_nt(Game)(slice, batch.game),
            name=slice(batch.name),
            is_resetting=slice(batch.is_resetting),
            reward=batch.reward[:, start:end - 1],
        ),
        meta=ChunkMeta(
            start=batch_with_meta.meta.start + start,
            end=batch_with_meta.meta.start + end,
            meta=batch_with_meta.meta.meta,
        ),
    ), epoch

  def shutdown(self):
    self.data_source.shutdown()

  @property
  def batch_size(self) -> int:
    return self.data_source.batch_size


class CachedDataSource(AbstractDataSource):
  """Guaranteed fast, useful for performance benchmarking."""

  def __init__(self, source: AbstractDataSource):
    self.source = source
    self.counter = 0
    self._get_batch = functools.cache(self.source.__next__)

  @property
  def batch_size(self) -> int:
    return self.source.batch_size

  def __next__(self) -> Tuple[BatchWithMeta, float]:
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
  burnin: int = 5  # get rid of early-game correlations

  wds: WebDataConfig = dataclasses.field(default_factory=WebDataConfig)

def _make_source(
    cached: bool = False,
    burnin: int = 5,
    replays: Optional[List[ReplayInfo]] = None,
    wds_path: Optional[str] = None,
    **kwargs,
) -> AbstractDataSource:
  if replays is not None:
    del kwargs['wds']
    source = DataSource(replays=replays, **kwargs)
  elif wds_path is not None:
    del kwargs['balance_characters']  # not supported
    wds: dict = kwargs.pop('wds')  # already converted by dataclasses.asdict
    source = WebDataSource(dataset_path=wds_path, **kwargs, **wds)
  else:
    raise ValueError("Must specify either replays or wds_path.")

  if cached:
    source = CachedDataSource(source)
  else:
    for _ in range(burnin):
      next(source)

  return source

def _make_time_batched_source(
    unroll_chunks: int = 0,
    **kwargs,
) -> AbstractDataSource:
  if unroll_chunks > 0:
    return TimeBatchedDataSource(unroll_chunks=unroll_chunks, **kwargs)

  return _make_source(**kwargs)

make_source = _make_time_batched_source

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
          wds_path=dataset_config.wds_path,
          split=split,
          name_map=name_map,
          mirror=dataset_config.mirror and split == 'train',  # mirror only in train
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
