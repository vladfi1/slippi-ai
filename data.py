import itertools
import pickle
import zlib

import numpy as np
import tensorflow as tf

import utils

class TrajectoryManager:
  # TODO: manage recurrent state? can also do it in the learner

  def __init__(self, source, num_frames: int):
    self.source = source
    self.num_frames = num_frames
    self.game = None

  def grab_chunk(self, n):
    # TODO: write a unit test for this
    is_first = False
    if self.game is None:
      self.game = next(self.source)
      self.frame = 0
      is_first = True

    left = len(self.game['stage']) - self.frame

    if n < left:
      new_frame = self.frame + n
      slice = lambda a: a[self.frame:new_frame]
      chunk = tf.nest.map_structure(slice, self.game)
      self.frame = new_frame
      size = n
    else:
      slice = lambda a: a[self.frame:]
      chunk = tf.nest.map_structure(slice, self.game)
      self.game = None
      size = left

    restarting = np.zeros([size], dtype=bool)
    if is_first:
      restarting[0] = True

    return size, (chunk, restarting)

  def next(self):
    chunks = []
    frames_left = self.num_frames
    while frames_left > 0:
      size, chunk = self.grab_chunk(frames_left)
      chunks.append(chunk)
      frames_left -= size
    return tf.nest.map_structure(lambda *xs: np.concatenate(xs), *chunks)

def swap_players(game):
  old_players = game['player']
  new_players = {1: old_players[2], 2: old_players[1]}
  new_game = game.copy()
  new_game['player'] = new_players
  return new_game

class DataSource:
  def __init__(
      self, embed_game, filenames, compressed,
      batch_size=64,
      unroll_length=64):
    self.batch_size = batch_size
    self.unroll_length = unroll_length
    self.embed_game = embed_game
    self.filenames = filenames
    self.compressed = compressed
    trajectories = self.produce_trajectories()
    self.managers = [
        TrajectoryManager(trajectories, unroll_length)
        for _ in range(batch_size)]

  def produce_trajectories(self):
    for path in itertools.cycle(self.filenames):
      with open(path, 'rb') as f:
        obj_bytes = f.read()
      if self.compressed:
        obj_bytes = zlib.decompress(obj_bytes)
      game = pickle.loads(obj_bytes)
      yield game
      yield swap_players(game)

  def __next__(self):
    return utils.batch_nest([m.next() for m in self.managers])
