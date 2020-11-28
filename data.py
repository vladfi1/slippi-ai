import atexit
import itertools
import multiprocessing as mp
import os
import pickle
import random
import zlib

import numpy as np
import tree

import melee

import reward
import stats
import utils

def train_test_split(data_dir, subset=None, test_ratio=.1):
  if subset:
    print("Using subset:", subset)
    filenames = stats.get_subset(subset)
    filenames = [name + '.pkl' for name in filenames]
  else:
    print("Using all replays in", data_dir)
    filenames = sorted(os.listdir(data_dir))

  print(f"Found {len(filenames)} replays.")

  # reproducible train/test split
  rng = random.Random()
  rng.shuffle(filenames)
  test_files = rng.sample(filenames, int(test_ratio * len(filenames)))
  test_set = set(test_files)
  train_files = [f for f in filenames if f not in test_set]
  train_paths = [os.path.join(data_dir, f) for f in train_files]
  test_paths = [os.path.join(data_dir, f) for f in test_files]
  return train_paths, test_paths

def game_len(game):
  return len(game[1])

class TrajectoryManager:
  # TODO: manage recurrent state? can also do it in the learner

  def __init__(self, source):
    self.source = source
    self.game = None

  def find_game(self, n):
    while True:
      game = next(self.source)
      if game_len(game) >= n: break
    self.game = game
    self.frame = 0

  def grab_chunk(self, n):
    # TODO: write a unit test for this
    needs_reset = self.game is None or game_len(self.game) - self.frame < n
    if needs_reset:
      self.find_game(n)

    new_frame = self.frame + n
    slice = lambda a: a[self.frame:new_frame]
    chunk = tree.map_structure(slice, self.game)
    self.frame = new_frame

    return chunk, needs_reset

def swap_players(game):
  old_players = game['player']
  new_players = {1: old_players[2], 2: old_players[1]}
  new_game = game.copy()
  new_game['player'] = new_players
  return new_game

def detect_repeated_actions(controllers):
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

def indices_and_counts(repeats, max_repeat=15):
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

def compress_repeated_actions(game, rewards, embed_controller, max_repeat):
  controllers = game['player'][1]['controller_state']
  controllers = embed_controller.map(lambda e, a: e.preprocess(a), controllers)

  repeats = detect_repeated_actions(controllers)
  indices, counts = indices_and_counts(repeats, max_repeat)

  compressed_game = tree.map_structure(lambda a: a[indices], game)
  sum_rewards = np.array([sum(rewards[idx-count:idx+1]) for idx, count in zip(indices, counts)])
  return compressed_game, counts, sum_rewards

def _charset(chars):
  if chars is None:
    chars = list(melee.Character)
  return set(c.value for c in chars)

class DataSource:
  def __init__(
      self,
      filenames,
      compressed=True,
      batch_size=64,
      unroll_length=64,
      max_action_repeat=15,
      # preprocesses (discretizes) actions before repeat detection
      embed_controller=None,
      # Lists of melee.Character. None means all allowed.
      allowed_characters=None,
      allowed_opponents=None,
      ):
    self.filenames = filenames
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

  def produce_trajectories(self):
    raw_games = self.produce_raw_games()
    allowed_games = filter(self.is_allowed, raw_games)
    processed_games = map(self.process_game, allowed_games)
    yield from processed_games

  def process_game(self, game):
    rewards = reward.compute_rewards(game)
    return compress_repeated_actions(
        game, rewards, self.embed_controller, self.max_action_repeat)

  def produce_raw_games(self):
    """Raw games without post-processing."""
    for path in itertools.cycle(self.filenames):
      with open(path, 'rb') as f:
        obj_bytes = f.read()
      if self.compressed:
        obj_bytes = zlib.decompress(obj_bytes)
      game = pickle.loads(obj_bytes)
      yield game
      yield swap_players(game)

  def is_allowed(self, game):
    return (
        game['player'][1]['character'][0] in self.allowed_characters
        and
        game['player'][2]['character'][0] in self.allowed_opponents)

  def __next__(self):
    return utils.batch_nest(
        [m.grab_chunk(self.unroll_length) for m in self.managers])

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

  def __next__(self):
    return self.batch_queue.get()

_name_to_character = {c.name.lower(): c for c in melee.Character}

def _chars_from_string(chars: str):
  if chars == 'all':
    return list(melee.Character)
  chars = chars.split(',')
  return [_name_to_character[c] for c in chars]

CONFIG = dict(
    batch_size=32,
    unroll_length=64,
    compressed=True,
    max_action_repeat=15,
    in_parallel=True,
    # comma-separated lists of characters, or "all"
    allowed_characters='fox',
    allowed_opponents='fox',
)

def make_source(
    allowed_characters: str,
    allowed_opponents: str,
    in_parallel: bool,
    **kwargs):
  constructor = DataSourceMP if in_parallel else DataSource
  return constructor(
      allowed_characters=_chars_from_string(allowed_characters),
      allowed_opponents=_chars_from_string(allowed_opponents),
      **kwargs)
