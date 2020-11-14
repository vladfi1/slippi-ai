"""Learner test script."""

import itertools
import os
import pickle

from absl import app
from absl import flags

import numpy as np
import sonnet as snt
import tensorflow as tf

import melee
import embed
import stats
import make_dataset
import paths
import utils

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 2, 'Learner batch size.')

def to_time_major(t):
  permutation = list(range(len(t.shape)))
  permutation[0] = 1
  permutation[1] = 0
  return tf.transpose(t, permutation)

class TrajectoryManager:
  # TODO: manage recurrent state? can also do it in the learner

  def __init__(self, source, num_frames: int):
    self.source = source
    self.num_frames = num_frames
    self.game = None

  def grab_chunk(self, n):
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

class DataSource:
  def __init__(self, embed_game, filenames, batch_size=64, unroll_length=64):
    self.embed_game = embed_game
    self.filenames = filenames
    trajectories = self.produce_trajectories()
    self.managers = [
        TrajectoryManager(trajectories, unroll_length)
        for _ in range(batch_size)]

  def produce_trajectories(self):
    for path in itertools.cycle(self.filenames):
      with open(path, 'rb') as f:
        game = pickle.load(f)
      yield game
      # TODO: also yield with players swapped?

  def __next__(self):
    return utils.batch_nest([m.next() for m in self.managers])

class Learner:

  def __init__(self, embed_game):
    self.network = snt.nets.MLP([256, 128, embed.embed_controller.size])
    self.optimizer = snt.optimizers.Adam(1e-4)
    self.embed_game = embed_game

  def step(self, batch):
    gamestate, restarting = tf.nest.map_structure(to_time_major, batch)

    flat_gamestate = self.embed_game(gamestate)
    prev_gamestate = flat_gamestate[:-1]

    p1_controller = gamestate['player'][1]['controller_state']
    next_action = tf.nest.map_structure(lambda t: t[1:], p1_controller)

    with tf.GradientTape() as tape:
      outputs = self.network(prev_gamestate)
      next_action_distances = embed.embed_controller.distance(
          outputs, next_action)
      mean_distances = tf.nest.map_structure(
          tf.reduce_mean, next_action_distances)
      loss = tf.add_n(tf.nest.flatten(mean_distances))

    params = self.network.trainable_variables
    grads = tape.gradient(loss, params)
    self.optimizer.apply(grads, params)
    return loss

def main(_):
  embed_game = embed.make_game_embedding()
  learner = Learner(embed_game)

  filenames = [
      paths.FOX_DITTO_PATH + name + '.pkl'
      for name in make_dataset.get_fox_ditto_names()]
  data_source = DataSource(embed_game, filenames, batch_size=FLAGS.batch_size)

  for _ in range(1000):
    loss = learner.step(next(data_source))
    print(loss.numpy())

if __name__ == '__main__':
  app.run(main)
