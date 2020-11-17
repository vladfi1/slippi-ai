import itertools
import pickle
import zlib

import numpy as np
import tree

import utils

def game_len(game):
  return len(game['stage'])

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

class DataSource:
  def __init__(
      self, filenames, compressed,
      batch_size=64,
      unroll_length=64):
    self.batch_size = batch_size
    self.unroll_length = unroll_length

    self.filenames = filenames
    self.compressed = compressed
    trajectories = self.produce_trajectories()
    self.managers = [
        TrajectoryManager(trajectories)
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
    return utils.batch_nest(
        [m.grab_chunk(self.unroll_length) for m in self.managers])
