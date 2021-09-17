"""Train (and test) a network via imitation learning."""

import functools
import os
import pickle
import time
import io, tarfile

import sacred

import numpy as np
import sonnet as snt
import tensorflow as tf

import melee

import controller_heads
import data
import embed
from learner import Learner
import networks
import paths
import policies
import s3_lib
import stats
import train_lib
import utils

ex = sacred.Experiment('imitation')

mongo_uri = os.environ.get('MONGO_URI')
if mongo_uri:
  from sacred.observers import MongoObserver
  db_name = os.environ.get('MONGO_DB_NAME', 'sacred')
  ex.observers.append(MongoObserver(url=mongo_uri, db_name=db_name))

@ex.config
def config():
  num_epochs = 1000  # an "epoch" is just "epoch_time" seconds
  epoch_time = 10  # seconds between testing/logging
  save_interval = 300  # seconds between saving to disk
  save_model = True  # serialize model with tf.saved_model

  dataset = dict(
      data_dir=paths.COMPRESSED_PATH,  # Path to pickled dataset.
      subset=None,  # Subset to train on. Defaults to all files.
      test_ratio=.1,  # Fraction of dataset for testing.
  )
  data = data.CONFIG
  learner = Learner.DEFAULT_CONFIG
  network = networks.DEFAULT_CONFIG
  controller_head = controller_heads.DEFAULT_CONFIG

  expt_dir = train_lib.get_experiment_directory()
  tag = train_lib.get_experiment_tag()

@ex.automain
def main(dataset, expt_dir, num_epochs, epoch_time, save_interval, _config, _log, _run):
  embed_controller = embed.embed_controller_discrete  # TODO: configure

  controller_head_config = dict(
      _config['controller_head'],
      embed_controller=embed.get_controller_embedding_with_action_repeat(
          embed_controller,
          _config['data']['max_action_repeat']))

  policy = policies.Policy(
      networks.construct_network(**_config['network']),
      controller_heads.construct(**controller_head_config))
  
  learning_rate = tf.Variable(
      _config['learner']['learning_rate'], name='learning_rate')
  learner = Learner(
      learning_rate=learning_rate,
      policy=policy,
      compile=_config['learner']['compile'],
  )

  for comp in ['network', 'controller_head']:
    print(f'\nUsing {comp}: {_config[comp]["name"]}')

  train_paths, test_paths = data.train_test_split(**dataset)
  print(f'Training on {len(train_paths)} replays, testing on {len(test_paths)}')

  data_config = dict(_config['data'], embed_controller=embed_controller)
  train_data = data.make_source(filenames=train_paths, **data_config)
  test_data = data.make_source(filenames=test_paths, **data_config)
  test_batch = train_lib.sanitize_batch(next(test_data))

  assert test_batch[0][0]['player'][1]['jumps_left'].dtype == np.uint8

  train_manager = train_lib.TrainManager(learner, train_data, dict(train=True))
  test_manager = train_lib.TrainManager(learner, test_data, dict(train=False))

  # initialize variables
  train_stats = train_manager.step()
  _log.info('loss initial: %f', train_stats['loss'].numpy())

  # saving and restoring
  tf_state = dict(
    policy=policy.variables,
    optimizer=learner.optimizer.variables,
  )

  def get_state():
    return tf.nest.map_structure(lambda v: v.numpy(), tf_state)

  def set_state(state):
    tf.nest.map_structure(
      lambda var, val: var.assign(val),
      tf_state, state)

  pickle_path = os.path.join(expt_dir, 'latest.pkl')
  tag = _config["tag"]
  
  save_to_s3 = False
  if 'S3_CREDS' in os.environ:
    s3_store = s3_lib.get_store()
    s3_keys = s3_lib.get_keys(tag)
    save_to_s3 = True

  def save():
    # Local Save
    _log.info('saving state to %s', pickle_path)
    with open(pickle_path, 'wb') as f:
      pickle.dump(get_state(), f)

    if save_to_s3:
      _log.info('saving state to S3: %s', s3_keys.params)
      pickled_state = pickle.dumps(get_state())
      s3_store.put(s3_keys.params, pickled_state)

  save = utils.Periodically(save, save_interval)

  ckpt = tf.train.Checkpoint(
      step=tf.Variable(0, trainable=False),
      policy=policy,
      optimizer=learner.optimizer,
  )
  
  if save_to_s3:
    try:
      obj = s3_store.get(s3_keys.params)
      _log.info('restoring from %s', s3_keys.params)
      set_state(pickle.loads(obj))
    except KeyError:
      _log.info('no params found at %s', s3_keys.params)
  else:
    manager = tf.train.CheckpointManager(
        ckpt, os.path.join(expt_dir, 'tf_ckpts'), max_to_keep=3)
    manager.restore_or_initialize()
    save = utils.Periodically(manager.save, save_interval)

    if os.path.exists(pickle_path):
      _log.info('restoring from %s', pickle_path)
      with open(pickle_path, 'rb') as f:
        pickled_state = pickle.load(f)
      set_state(pickled_state)

  train_loss = train_manager.step()['loss']
  _log.info('loss post-restore: %f', train_loss.numpy())

  if _config['save_model']:
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

    sample_signature = [
        utils.nested_add_batch_dims(gamestate_signature, 1),
        utils.nested_add_batch_dims(hidden_state_signature, 1),
    ]
    saved_module.sample = utils.with_flat_signature(policy.sample, sample_signature)

    saved_model_path = os.path.join(expt_dir, 'saved_model')

    def save_model():
      tf.saved_model.save(saved_module, saved_model_path)
      saved_model_bytes = io.BytesIO()
      with tarfile.open(fileobj=saved_model_bytes, mode='x') as tar:
        tar.add(saved_model_path, arcname='.')

      if save_to_s3:
        _log.info('saving model to S3: %s', s3_keys.saved_model)
        s3_store.put(s3_keys.saved_model, saved_model_bytes.getvalue())

    save_model = utils.Periodically(save_model, save_interval)

  total_steps = 0
  frames_per_batch = train_data.batch_size * train_data.unroll_length

  for _ in range(num_epochs):
    start_time = time.perf_counter()

    # train for epoch_time seconds
    steps = 0
    while True:
      train_stats = train_manager.step()
      steps += 1

      elapsed_time = time.perf_counter() - start_time
      if elapsed_time > epoch_time: break

    ckpt.step.assign_add(steps)
    total_steps = ckpt.step.numpy()

    # now test
    test_stats = test_manager.step()

    train_loss = train_stats['loss'].numpy()
    test_loss = test_stats['loss'].numpy()

    all_stats = dict(
        train=train_stats,
        test=test_stats,
        learning_rate=learning_rate.numpy(),
    )
    train_lib.log_stats(ex, all_stats, total_steps)

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
    if _config['save_model']:
      save_model()
