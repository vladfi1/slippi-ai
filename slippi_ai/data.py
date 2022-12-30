import atexit
import dataclasses
import itertools
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

# These keys correspond to corrupted replays. I only found them because
# they cause crashes slippi-ai; there may be others which are bad but
# don't trigger any errors.
# TODO: remove these from the dataset
BAD_KEYS = {
    '260950213a47132a2b88310734883c8a',  # causes out-of-bounds rewards
    '4a9e2e174679bcb5ed72dbe5d858753d',  # player changes chars mid-game!
}

class Batch(NamedTuple):
  game: StateActionReward
  needs_reset: bool

class ReplayInfo(NamedTuple):
  path: str
  swap: bool

def _get_keys(
    df: pd.DataFrame,
    allowed_p0: Iterable[int],
    allowed_p1: Iterable[int],
) -> Iterable[str]:
  df = df[df['p0.character'].isin(allowed_p0)]
  df = df[df['p1.character'].isin(allowed_p1)]
  return df['key']


def _charset(chars: Optional[Iterable[melee.Character]]) -> Set[int]:
  if chars is None:
    chars = list(melee.Character)
  return set(c.value for c in chars)


@dataclasses.dataclass
class DatasetConfig:
    data_dir: Optional[str] = None
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

  if config.meta_path is not None:
    df = pd.read_parquet(config.meta_path)
    # check that we have the right metadata
    assert sorted(df['key']) == filenames

    allowed_characters = _charset(config.allowed_characters)
    allowed_opponents = _charset(config.allowed_opponents)

    unswapped = _get_keys(df, allowed_characters, allowed_opponents)
    swapped = _get_keys(df, allowed_opponents, allowed_characters)
  else:
    assert config.allowed_characters is None
    assert config.allowed_opponents is None

    swapped = filenames
    unswapped = filenames

  # filter bad keys
  swapped = [key for key in swapped if key not in BAD_KEYS]
  unswapped = [key for key in unswapped if key not in BAD_KEYS]

  replays = []
  for key in unswapped:
    path = os.path.join(config.data_dir, key)
    replays.append(ReplayInfo(path, False))
  for key in swapped:
    path = os.path.join(config.data_dir, key)
    replays.append(ReplayInfo(path, True))

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
    """Grabs a chunk from a trajectory.

    Subsequent trajectories overlap by a single frame.
    """
    # TODO: write a unit test for this
    needs_reset = self.game is None or self.frame + n > game_len(self.game)

    if needs_reset:
      self.find_game(n)

    new_frame = self.frame + n
    slice = lambda a: a[self.frame:new_frame]
    chunk = tree.map_structure(slice, self.game)
    self.frame = new_frame

    return Batch(chunk, needs_reset)

def swap_players(game: Game) -> Game:
  return game._replace(p0=game.p1, p1=game.p0)

def detect_repeated_actions(controllers: Nest[np.ndarray]) -> Sequence[bool]:
  """Labels actions as repeated or not.

  Args:
    controllers: A nest of numpy arrays with shape [T].
  Returns:
    A boolean numpy array `repeats` with shape [T-1].
    repeats[i] is True iff controllers[i+1] equals controllers[i]
  """
  is_same = lambda a: a[:-1] == a[1:]
  repeats = tree.map_structure(is_same, controllers)
  repeats = np.stack(tree.flatten(repeats), -1)
  repeats = np.all(repeats, -1)
  return repeats

def indices_and_counts(
    repeats: Sequence[bool],
    max_repeat=15,
  ) -> Tuple[Sequence[int], Sequence[int]]:
  """Finds the indices and counts of repeated actions.

  `repeats` is meant to be produced by `detect_repeated_actions`
  If `controllers` is [a, a, a, c, b, b], then
  repeats = [T, T, F, F, T]
  indices = [2, 3, 5]
  counts = [2, 0, 1]

  Args:
    repeats: A boolean array with shape [T-1].
    max_repeat: Maximum number of consecutive repeated actions before a repeat
      is considered a non-repeat.
  Returns:
    A tuple (indices, counts).
  """
  indices = []
  counts = []

  count = 0

  for i, is_repeat in enumerate(repeats):
    if not is_repeat or count == max_repeat:
      indices.append(i)  # index of the last repeated action
      counts.append(count)
      count = 0
    else:
      count += 1

  indices.append(len(repeats))
  counts.append(count)

  return np.array(indices), np.array(counts)

def compress_repeated_actions(
    game: Game,
    rewards: Sequence[float],
    embed_controller: embed.Embedding[Controller, Any],
    max_repeat: int,
    embed_game=embed.default_embed_game,
  ) -> StateActionReward:
  controllers = embed_controller.from_state(game.p0.controller)

  repeats = detect_repeated_actions(controllers)
  indices, counts = indices_and_counts(repeats, max_repeat)

  compressed_states = tree.map_structure(lambda a: a[indices], game)
  compressed_states = embed_game.from_state(compressed_states)
  actions = tree.map_structure(lambda a: a[indices], controllers)
  compressed_rewards = np.concatenate([
      np.zeros([1], dtype=np.float32),
      np.add.reduceat(rewards, indices[:-1]),
  ])
  compressed_game = StateActionReward(
      state=compressed_states,
      action=embed.ActionWithRepeat(
          action=actions,
          repeat=counts),
      reward=compressed_rewards)

  shapes = [x.shape for x in tree.flatten(compressed_game)]
  for s in shapes:
    assert s == shapes[0]

  return compressed_game

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
      batch_size=64,
      unroll_length=64,
      max_action_repeat=15,
      # Lists of melee.Character. None means all allowed.
      allowed_characters=None,
      allowed_opponents=None,
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
    return compress_repeated_actions(
        game, rewards, self.embed_controller, self.max_action_repeat)

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
    next_batch = utils.batch_nest(
        [m.grab_chunk(self.unroll_length) for m in self.managers])
    epoch = self.replay_counter / len(self.replays)
    return next_batch, epoch
  
  def close(self):
    pass

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
  
  def close(self):
    self.batch_queue.close()
    self.process.terminate()

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
