import atexit
import dataclasses
import itertools
import json
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

from slippi_ai import embed, reward, utils
from slippi_ai.types import Controller, Game, game_array_to_nt

from slippi_ai.embed import StateAction

class PlayerMeta(NamedTuple):
  character: int
  name: str

  @classmethod
  def from_metadata(cls, player_meta: dict) -> 'PlayerMeta':
    netplay = player_meta['netplay']
    if netplay is None:
      name = player_meta['name_tag']
    else:
      name = netplay['name']
    return cls(
        character=player_meta['character'],
        name=name)

class ReplayMeta(NamedTuple):
  p0: PlayerMeta
  p1: PlayerMeta
  stage: int
  slp_md5: str

  @classmethod
  def from_metadata(cls, metadata: dict) -> 'ReplayMeta':
    return cls(
        p0=PlayerMeta.from_metadata(metadata['players'][0]),
        p1=PlayerMeta.from_metadata(metadata['players'][1]),
        stage=metadata['stage'],
        slp_md5=metadata['slp_md5'])

class ReplayInfo(NamedTuple):
  path: str
  swap: bool
  # We use empty tuple instead of None to play nicely with Tensorflow.
  meta: Union[ReplayMeta, Tuple[()]] = ()

class ChunkMeta(NamedTuple):
  start: int
  end: int
  info: ReplayInfo

class Chunk(NamedTuple):
  states: Game
  meta: ChunkMeta

class Frames(NamedTuple):
  state_action: StateAction
  # This will have length one less than the states and actions.
  reward: np.float32

class Batch(NamedTuple):
  frames: Frames
  needs_reset: bool
  meta: ChunkMeta


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
      replay_meta = ReplayMeta.from_metadata(row)

      c0 = replay_meta.p0.character
      c1 = replay_meta.p1.character
      replay_path = os.path.join(config.data_dir, replay_meta.slp_md5)

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


def game_len(game: Game):
  return len(game.stage)

class TrajectoryManager:
  # TODO: manage recurrent state? can also do it in the learner

  def __init__(
      self,
      source: Iterator[ReplayInfo],
      unroll_length: int,
      overlap: int = 1,
      compressed: bool = True,
      game_filter: Optional[Callable[[Game], bool]] = None,
  ):
    self.source = source
    self.compressed = compressed
    self.unroll_length = unroll_length
    self.overlap = overlap
    self.game_filter = game_filter or (lambda _: True)

    self.game: Game = None
    self.frame: int = None
    self.info: ReplayInfo = None

  def load_game(self, info: ReplayInfo) -> Game:
    game = read_table(info.path, compressed=self.compressed)
    if info.swap:
      game = swap_players(game)
    return game

  def find_game(self):
    while True:
      info = next(self.source)
      game = self.load_game(info)
      if game_len(game) < self.unroll_length:
        continue
      if not self.game_filter(game):
        continue
      break
    self.game = game
    self.frame = 0
    self.info = info

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

    return Chunk(states, ChunkMeta(start, end, self.info))

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
      embed_controller: embed.Embedding[Controller, Any],
      embed_game: embed.Embedding[Game, Any] = embed.default_embed_game,
      compressed: bool = True,
      batch_size: int = 64,
      unroll_length: int = 64,
      extra_frames: int = 1,
      # None means all allowed.
      allowed_characters: Optional[list[melee.Character]] = None,
      allowed_opponents: Optional[list[melee.Character]] = None,
  ):
    self.replays = replays
    self.batch_size = batch_size
    self.unroll_length = unroll_length
    self.chunk_size = unroll_length + extra_frames
    self.compressed = compressed
    self.embed_controller = embed_controller
    self.embed_game = embed_game

    replays = self.iter_replays()
    self.managers = [
        TrajectoryManager(
            replays,
            unroll_length=self.chunk_size,
            overlap=extra_frames,
            compressed=compressed,
            game_filter=self.is_allowed)
        for _ in range(batch_size)]

    self.allowed_characters = _charset(allowed_characters)
    self.allowed_opponents = _charset(allowed_opponents)

  def iter_replays(self) -> Iterator[ReplayInfo]:
    self.replay_counter = 0
    for replay in itertools.cycle(self.replays):
      self.replay_counter += 1
      yield replay

  def is_allowed(self, game: Game) -> bool:
    # TODO: handle Zelda/Sheik transformation
    return (
        game.p0.character[0] in self.allowed_characters
        and
        game.p1.character[0] in self.allowed_opponents)

  def process_game(self, game: Game) -> Frames:
    # These could be deferred to the learner.
    rewards = reward.compute_rewards(game)
    controllers = self.embed_controller.from_state(game.p0.controller)

    states = self.embed_game.from_state(game)

    state_action = StateAction(states, controllers)
    return Frames(state_action=state_action, reward=rewards)

  def process_batch(self, chunks: list[Chunk]) -> Batch:
    batches: List[Batch] = []

    for chunk in chunks:
      batches.append(Batch(
          frames=self.process_game(chunk.states),
          needs_reset=chunk.meta.start == 0,
          meta=chunk.meta))

    return utils.batch_nest_nt(batches)

  def __next__(self) -> Tuple[Batch, float]:
    batch: Batch = self.process_batch(
        [m.grab_chunk() for m in self.managers])
    epoch = self.replay_counter / len(self.replays)
    assert batch.frames.state_action.state.stage.shape[-1] == self.chunk_size
    return batch, epoch

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

  def __del__(self):
    self.process.terminate()

CONFIG = dict(
    batch_size=32,
    unroll_length=64,
    compressed=True,
    in_parallel=True,
    # `extra_frames` is determined by policy.delay
)

def make_source(
    in_parallel: bool,
    **kwargs):
  constructor = DataSourceMP if in_parallel else DataSource
  return constructor(**kwargs)
