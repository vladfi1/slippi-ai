"""Train (and test) a network via imitation learning."""

import functools
import os
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
import train_lib
import utils

LOG_INTERVAL = 10
SAVE_INTERVAL = 300

ex = Experiment('imitation')
ex.observers.append(MongoObserver())

@ex.config
def config():
  dataset = dict(
      data_dir=paths.COMPRESSED_PATH,  # Path to pickled dataset.
      subset=None,  # Subset to train on. Defaults to all files.
      test_ratio=.1,  # Fraction of dataset for testing.
  )
  data = dict(
      batch_size=32,
      unroll_length=64,
      compressed=True,
  )
  learner = Learner.DEFAULT_CONFIG
  network = networks.DEFAULT_CONFIG
  expt_dir = train_lib.get_experiment_directory()

@ex.automain
def main(dataset, expt_dir, _config, _log):
  network = networks.construct_network(**_config['network'])
  policy = Policy(network)
  learner = Learner(
      policy=policy,
      **_config['learner'])

  train_paths, test_paths = data.train_test_split(**dataset)
  print(f'Training on {len(train_paths)} replays, testing on {len(test_paths)}')

  data_config = _config['data']
  train_data = data.DataSource(train_paths, **data_config)
  test_data = data.DataSource(test_paths, **data_config)
  test_batch = train_lib.sanitize_batch(next(test_data))

  import numpy as np
  assert test_batch[0]['player'][1]['jumps_left'].dtype == np.uint8

  train_manager = train_lib.TrainManager(learner, train_data, dict(train=True))
  test_manager = train_lib.TrainManager(learner, test_data, dict(train=False))

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
  saved_module.initial_state = tf.function(
      policy.initial_state, input_signature=[tf.TensorSpec((), tf.int64)])
  saved_module.all_variables = policy.variables

  saved_model_path = os.path.join(expt_dir, 'saved_model')
  save_model = utils.Periodically(functools.partial(
      tf.saved_model.save, saved_module, saved_model_path), SAVE_INTERVAL)

  total_steps = 0
  frames_per_batch = train_data.batch_size * train_data.unroll_length

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
