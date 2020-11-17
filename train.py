"""Learner test script."""

import datetime
import functools
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
from learner import Learner
import networks
import paths
from policy import Policy
import stats
import utils

LOG_INTERVAL = 10
SAVE_INTERVAL = 300

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

class TrainManager:

  def __init__(self, learner, data_source, step_kwargs={}):
    self.learner = learner
    self.data_source = data_source
    self.hidden_state = learner.policy.initial_state(data_source.batch_size)
    self.step_kwargs = step_kwargs

    self.data_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()

  def step(self):
    with self.data_profiler:
      batch = next(self.data_source)
    with self.step_profiler:
      loss, self.hidden_state = self.learner.compiled_step(
          batch, self.hidden_state, **self.step_kwargs)
    return loss

@ex.automain
def main(dataset, subset, expt_dir, _config, _log):
  network = networks.construct_network(**_config['network'])
  policy = Policy(network)
  learner = Learner(
      policy=policy,
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
  train_data = data.DataSource(train_paths, **data_config)
  test_data = data.DataSource(test_paths, **data_config)
  test_batch = next(test_data)

  train_manager = TrainManager(learner, train_data, dict(train=True))
  test_manager = TrainManager(learner, test_data, dict(train=False))

  # initialize variables
  train_loss = train_manager.step()
  _log.info('loss initial: %f', train_loss.numpy())

  ckpt = tf.train.Checkpoint(
      step=tf.Variable(0, trainable=False),
      policy=policy,
      optimizer=learner.optimizer,
  )
  manager = tf.train.CheckpointManager(
      ckpt, os.path.join(expt_dir, 'tf_ckpts'), max_to_keep=3)
  manager.restore_or_initialize()
  save = utils.Periodically(manager.save, SAVE_INTERVAL)
  train_loss = train_manager.step()
  _log.info('loss post-restore: %f', train_loss.numpy())

  # signatures without batch dims
  gamestate_signature = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[2:], t.dtype),
      test_batch[0])
  hidden_state_signature = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[1:], t.dtype),
      test_manager.hidden_state)

  loss_signature = [
      utils.nested_add_batch_dims(gamestate_signature, 2),
      utils.nested_add_batch_dims(hidden_state_signature, 1),
  ]

  saved_module = snt.Module()
  # with_flat_signature is a workaround for tf.function not supporting dicts
  # with non-string keys in the input_signature. The solution is to change
  # embed_players in embed.py to be an ArrayEmbedding, not a StructEmbedding.
  saved_module.loss = utils.with_flat_signature(policy.loss, loss_signature)
  saved_module.all_variables = policy.variables

  saved_model_path = os.path.join(expt_dir, 'saved_model')
  save_model = utils.Periodically(functools.partial(
      tf.saved_model.save, saved_module, saved_model_path), SAVE_INTERVAL)

  total_steps = 0
  frames_per_batch = data_config['batch_size'] * data_config['unroll_length']

  for _ in range(1000):
    steps = 0
    start_time = time.perf_counter()

    # train for a while
    while True:
      elapsed_time = time.perf_counter() - start_time
      if elapsed_time > LOG_INTERVAL: break
      train_loss = train_manager.step()
      steps += 1

    ckpt.step.assign_add(steps)
    total_steps = ckpt.step.numpy()

    train_loss = train_loss.numpy()
    ex.log_scalar('train.loss', train_loss, total_steps)

    # now test
    test_loss = test_manager.step()
    test_loss = test_loss.numpy()
    ex.log_scalar('test.loss', test_loss, total_steps)

    sps = steps / elapsed_time
    mps = sps * frames_per_batch / (60 * 60)
    ex.log_scalar('sps', sps, total_steps)
    ex.log_scalar('mps', mps, total_steps)

    print(f'batches={total_steps} sps={sps:.2f} mps={mps:.2f}')
    print(f'losses: train={train_loss:.4f} test={test_loss:.4f}')
    print(f'timing:'
          f' data={train_manager.data_profiler.mean_time():.3f}'
          f' step={train_manager.step_profiler.mean_time():.3f}')
    print()

    save_path = save()
    if save_path:
      _log.info('Saved network to %s', save_path)
    save_model()
