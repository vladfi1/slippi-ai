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

from config import ConfigParser
import data
import embed
from learner import Learner
import networks
import paths
import stats
import utils

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', paths.COMPRESSED_PATH, 'Path to pickled dataset.')
flags.DEFINE_string('subset', None, 'Subset to train on. Defaults to all files.')

DEFAULT_CONFIG = dict(
    data=dict(
        batch_size=32,
        unroll_length=64,
        compressed=True,
    ),
    learner=Learner.DEFAULT_CONFIG,
    network=networks.DEFAULT_CONFIG,
)

config_parser = ConfigParser('config', DEFAULT_CONFIG)

def main(_):
  config = config_parser.parse()
  embed_game = embed.make_game_embedding()
  network = networks.construct_network(**config['network'])
  learner = Learner(
      embed_game=embed_game,
      network=network,
      **config['learner'])

  data_dir = FLAGS.dataset
  if FLAGS.subset:
    filenames = stats.SUBSETS[FLAGS.subset]()
    filenames = [name + '.pkl' for name in filenames]
  else:
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

  data_config = config['data']
  train_data = data.DataSource(embed_game, train_files, **data_config)
  test_data = data.DataSource(embed_game, test_files, **data_config)

  data_profiler = utils.Profiler()
  step_profiler = utils.Profiler()

  steps = 0
  frames_per_batch = data_config['batch_size'] * data_config['unroll_length']

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
    print(f'losses: train={train_loss.numpy():.4f} test={test_loss.numpy():.4f}')
    print(f'timing:'
          f' data={data_profiler.mean_time():.3f}'
          f' step={step_profiler.mean_time():.3f}')
    print()

if __name__ == '__main__':
  app.run(main)
