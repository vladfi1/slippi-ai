"""Learner test script."""

import os

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
import data

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 2, 'Learner batch size.')
flags.DEFINE_boolean('compressed', False, 'Compress with zlib.')

def to_time_major(t):
  permutation = list(range(len(t.shape)))
  permutation[0] = 1
  permutation[1] = 0
  return tf.transpose(t, permutation)

class Learner:

  def __init__(self, embed_game):
    self.network = snt.nets.MLP([256, 128, embed.embed_controller.size])
    self.optimizer = snt.optimizers.Adam(1e-4)
    self.embed_game = embed_game

  @tf.function
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

  data_dir = paths.FOX_DITTO_PATH
  if FLAGS.compressed:
    data_dir += 'Compressed'

  filenames = [
      os.path.join(data_dir, name + '.pkl')
      for name in make_dataset.get_fox_ditto_names()]
  data_source = data.DataSource(
      embed_game, filenames,
      batch_size=FLAGS.batch_size,
      compressed=FLAGS.compressed)

  data_profiler = utils.Profiler()
  step_profiler = utils.Profiler()

  for _ in range(1000):
    for _ in range(10):
      with data_profiler:
        batch = next(data_source)
      with step_profiler:
        loss = learner.step(batch)
    print(loss.numpy())
    print('data', data_profiler.mean_time())
    print('step', step_profiler.mean_time())

if __name__ == '__main__':
  app.run(main)
