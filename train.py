"""Learner test script."""

import os
import random
import time

from sacred import Experiment
from sacred.observers import MongoObserver

import numpy as np
import sonnet as snt
import tensorflow as tf

import melee

import data
import embed
from learner import Learner
import networks
import paths
import stats
import utils

ex = Experiment('imitation')
ex.observers.append(MongoObserver())

@ex.config
def config():
  dataset = paths.COMPRESSED_PATH  # Path to pickled dataset.
  subset = None  # Subset to train on. Defaults to all files.
  data = dict(
      batch_size=32,
      unroll_length=64,
      compressed=True,
  )
  learner = Learner.DEFAULT_CONFIG
  network = networks.DEFAULT_CONFIG

@ex.automain
def main(dataset, subset, _config):
  embed_game = embed.make_game_embedding()
  network = networks.construct_network(**_config['network'])
  learner = Learner(
      embed_game=embed_game,
      network=network,
      **_config['learner'])

  data_dir = dataset
  if subset:
    filenames = stats.SUBSETS[subset]()
    filenames = [name + '.pkl' for name in filenames]
  else:
    filenames = sorted(os.listdir(data_dir))

  # reproducible train/test split
  rng = random.Random()
  test_files = rng.sample(filenames, int(.1 * len(filenames)))
  test_set = set(test_files)
  train_files = [f for f in filenames if f not in test_set]
  print(f'Training on {len(train_files)} replays, testing on {len(test_files)}')
  train_paths = [os.path.join(data_dir, f) for f in train_files]
  test_paths = [os.path.join(data_dir, f) for f in test_files]

  data_config = _config['data']
  train_data = data.DataSource(embed_game, train_paths, **data_config)
  test_data = data.DataSource(embed_game, test_paths, **data_config)

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

    train_loss = train_loss.numpy()
    ex.log_scalar('train.loss', train_loss)

    # now test
    batch = next(test_data)
    test_loss = learner.step(batch, train=False)
    test_loss = test_loss.numpy()
    ex.log_scalar('test.loss', test_loss)

    elapsed_time = time.perf_counter() - start_time
    sps = steps / elapsed_time
    mps = sps * frames_per_batch / (60 * 60)
    ex.log_scalar('sps', sps)
    ex.log_scalar('mps', mps)

    print(f'batches={steps} sps={sps:.2f} mps={mps:.2f}')
    print(f'losses: train={train_loss:.4f} test={test_loss:.4f}')
    print(f'timing:'
          f' data={data_profiler.mean_time():.3f}'
          f' step={step_profiler.mean_time():.3f}')
    print()

if __name__ == '__main__':
  app.run(main)
