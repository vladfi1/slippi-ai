"""Learner test script."""

import os
import random
import time

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
flags.DEFINE_string('dataset', paths.FOX_DITTO_PATH, 'Path to pickled dataset.')
flags.DEFINE_boolean('compressed', False, 'Compress with zlib.')

flags.DEFINE_integer('batch_size', 2, 'Learner batch size.')
flags.DEFINE_integer('unroll_length', 64, 'Learner unroll length.')

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
  def step(self, batch, train=True):
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

    if train:
      params = self.network.trainable_variables
      grads = tape.gradient(loss, params)
      self.optimizer.apply(grads, params)
    return loss

def main(_):
  embed_game = embed.make_game_embedding()
  learner = Learner(embed_game)

  data_dir = FLAGS.dataset
  filenames = sorted(os.listdir(data_dir))

  # reproducible train/test split
  indices = range(len(filenames))
  rng = random.Random()
  test_indices = rng.sample(indices, int(.1 * len(filenames)))
  is_test = [False] * len(filenames)
  for i in test_indices:
    is_test[i] = True
  train_files = []
  test_files = []
  for i, b in enumerate(is_test):
    files = test_files if b else train_files
    files.append(os.path.join(data_dir, filenames[i]))
  print(f'Training on {len(train_files)} replays, testing with {len(test_files)}')

  train_data = data.DataSource(
      embed_game, train_files,
      batch_size=FLAGS.batch_size,
      unroll_length=FLAGS.unroll_length,
      compressed=FLAGS.compressed)
  test_data = data.DataSource(
      embed_game, test_files,
      batch_size=FLAGS.batch_size,
      unroll_length=FLAGS.unroll_length,
      compressed=FLAGS.compressed)

  data_profiler = utils.Profiler()
  step_profiler = utils.Profiler()

  steps = 0
  frames_per_batch = FLAGS.batch_size * FLAGS.unroll_length

  start_time = time.perf_counter()
  for _ in range(1000):
    # train for a while
    for _ in range(20):
      with data_profiler:
        batch = next(train_data)
      with step_profiler:
        train_loss = learner.step(batch)
      steps += 1

    # now test
    batch = next(test_data)
    test_loss = learner.step(batch, train=False)

    elapsed_time = time.perf_counter() - start_time
    sps = steps / elapsed_time
    mps = sps * frames_per_batch / (60 * 60)

    print(f'batches={steps} sps={sps:.2f} mps={mps:.2f}')
    print(f'losses: train={train_loss.numpy():.2f} test={test_loss.numpy():.2f}')
    print(f'timing:'
          f' data={data_profiler.mean_time():.3f}'
          f' step={step_profiler.mean_time():.3f}')
    print()

if __name__ == '__main__':
  app.run(main)
