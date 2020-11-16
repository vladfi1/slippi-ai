"""Learner test script."""

import datetime
import os
import random
import secrets
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

LOG_INTERVAL = 10
SAVE_INTERVAL = 30

ex = Experiment('imitation')
ex.observers.append(MongoObserver())

def get_experiment_directory():
  # create directory for tf checkpoints and other experiment artifacts
  today = datetime.date.today()
  expt_tag = f'{today.year}-{today.month}-{today.day}_{secrets.token_hex(8)}'
  expt_dir = f'experiments/{expt_tag}'
  os.makedirs(expt_dir, exist_ok=True)
  return expt_dir

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
  expt_dir = get_experiment_directory()

@ex.automain
def main(dataset, subset, expt_dir, _config, _log):
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

  train_hidden_state = network.initial_state(data_config['batch_size'])
  test_hidden_state = network.initial_state(data_config['batch_size'])

  # initialize variables
  train_loss, train_hidden_state = learner.compiled_step(
      next(train_data), train_hidden_state)
  _log.info('loss initial: %f', train_loss.numpy())

  ckpt = tf.train.Checkpoint(
      step=tf.Variable(0, trainable=False),
      optimizer=learner.optimizer,
      network=network,
      controller_head=learner.controller_head)
  manager = tf.train.CheckpointManager(
      ckpt, f'{expt_dir}/tf_ckpts', max_to_keep=3)
  manager.restore_or_initialize()
  save = utils.Periodically(manager.save, SAVE_INTERVAL)
  train_loss, train_hidden_state = learner.compiled_step(
      next(train_data), train_hidden_state, train=False)
  _log.info('loss post-restore: %f', train_loss.numpy())

  data_profiler = utils.Profiler()
  step_profiler = utils.Profiler()

  total_steps = 0
  frames_per_batch = data_config['batch_size'] * data_config['unroll_length']

  for _ in range(1000):
    steps = 0
    start_time = time.perf_counter()

    # train for a while
    while True:
      elapsed_time = time.perf_counter() - start_time
      if elapsed_time > LOG_INTERVAL: break

      with data_profiler:
        batch = next(train_data)
      with step_profiler:
        train_loss, train_hidden_state = learner.compiled_step(
            batch, train_hidden_state)
      steps += 1

    ckpt.step.assign_add(steps)
    total_steps = ckpt.step.numpy()

    train_loss = train_loss.numpy()
    ex.log_scalar('train.loss', train_loss, total_steps)

    # now test
    batch = next(test_data)
    test_loss, test_hidden_state = learner.compiled_step(
        batch, test_hidden_state, train=False)
    test_loss = test_loss.numpy()
    ex.log_scalar('test.loss', test_loss, total_steps)

    sps = steps / elapsed_time
    mps = sps * frames_per_batch / (60 * 60)
    ex.log_scalar('sps', sps, total_steps)
    ex.log_scalar('mps', mps, total_steps)

    print(f'batches={total_steps} sps={sps:.2f} mps={mps:.2f}')
    print(f'losses: train={train_loss:.4f} test={test_loss:.4f}')
    print(f'timing:'
          f' data={data_profiler.mean_time():.3f}'
          f' step={step_profiler.mean_time():.3f}')
    print()

    save_path = save()
    if save_path:
      _log.info('Saved network to %s', save_path)

if __name__ == '__main__':
  app.run(main)
