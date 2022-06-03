"""Train (and test) a network via imitation learning."""

import os
import pickle
import time

import sacred

import tensorflow as tf

from slippi_ai import (
    controller_heads,
    embed,
    networks,
    s3_lib,
    train_lib,
    utils,
)
from slippi_ai.learner import Learner
from slippi_ai import data as data_lib

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

  dataset = data_lib.DATASET_CONFIG
  data = data_lib.CONFIG
  learner = Learner.DEFAULT_CONFIG
  network = networks.DEFAULT_CONFIG
  controller_head = controller_heads.DEFAULT_CONFIG

  expt_dir = train_lib.get_experiment_directory()
  tag = train_lib.get_experiment_tag()
  save_to_s3 = False
  restore_tag = None

@ex.automain
def main(expt_dir, _config, _log):
  embed_controller = embed.embed_controller_discrete  # TODO: configure

  policy = train_lib.build_policy(
      controller_head_config=_config['controller_head'],
      max_action_repeat=_config['data']['max_action_repeat'],
      network_config=_config['network'],
      embed_controller=embed_controller,
  )

  learner_kwargs = _config['learner'].copy()
  learning_rate = tf.Variable(
      learner_kwargs['learning_rate'], name='learning_rate')
  learner_kwargs.update(learning_rate=learning_rate)
  learner = Learner(
      policy=policy,
      **learner_kwargs,
  )

  for comp in ['network', 'controller_head']:
    print(f'\nUsing {comp}: {_config[comp]["name"]}')

  dataset_config = _config['dataset'].copy()

  char_filters = {
      k: data_lib.chars_from_string(dataset_config[k])
      for k in ['allowed_characters', 'allowed_opponents']
  }
  dataset_config.update(char_filters)

  train_replays, test_replays = data_lib.train_test_split(**dataset_config)
  print(f'Training on {len(train_replays)} replays, testing on {len(test_replays)}')

  data_config = dict(
      _config['data'],
      embed_controller=embed_controller,
      **char_filters,
  )
  train_data = data_lib.make_source(replays=train_replays, **data_config)
  test_data = data_lib.make_source(replays=test_replays, **data_config)

  train_manager = train_lib.TrainManager(learner, train_data, dict(train=True))
  test_manager = train_lib.TrainManager(learner, test_data, dict(train=False))

  # initialize variables
  train_stats = train_manager.step()
  _log.info('loss initial: %f', train_stats['loss'].numpy())

  step = tf.Variable(0, trainable=False, name="step")

  # saving and restoring
  tf_state = dict(
      step=step,
      policy=policy.variables,
      optimizer=learner.optimizer.variables,
      # TODO: add in learning_rate?
  )

  def get_tf_state():
    return tf.nest.map_structure(lambda v: v.numpy(), tf_state)

  def set_tf_state(state):
    tf.nest.map_structure(
      lambda var, val: var.assign(val),
      tf_state, state)

  pickle_path = os.path.join(expt_dir, 'latest.pkl')
  tag = _config["tag"]

  save_to_s3 = _config['save_to_s3']
  if save_to_s3 or _config['restore_tag']:
    if 'S3_CREDS' not in os.environ:
      raise ValueError('must set the S3_CREDS environment variable')

    s3_store = s3_lib.get_store()
    s3_keys = s3_lib.get_keys(tag)

  def save():
    # Local Save
    tf_state = get_tf_state()

    # easier to always bundle the config with the state
    combined_state = dict(
        state=tf_state,
        config=_config,
    )
    pickled_state = pickle.dumps(combined_state)

    _log.info('saving state to %s', pickle_path)
    with open(pickle_path, 'wb') as f:
      f.write(pickled_state)

    if save_to_s3:
      _log.info('saving state to S3: %s', s3_keys.combined)
      s3_store.put(s3_keys.combined, pickled_state)

  save = utils.Periodically(save, _config['save_interval'])

  if _config['restore_tag']:
    restore_s3_keys = s3_lib.get_keys(_config['restore_tag'])
  elif save_to_s3:
    restore_s3_keys = s3_keys
  else:
    restore_s3_keys = None

  # attempt to restore parameters
  restored = False
  if restore_s3_keys is not None:
    try:
      restore_key = restore_s3_keys.combined
      obj = s3_store.get(restore_key)
      _log.info('restoring from %s', restore_key)
      combined_state = pickle.loads(obj)
      set_tf_state(combined_state['state'])
      restored = True
      # TODO: do some config compatibility validation
    except KeyError:
      # TODO: re-raise if user specified restore_tag
      _log.info('no params found at %s', restore_key)
  elif os.path.exists(pickle_path):
    _log.info('restoring from %s', pickle_path)
    with open(pickle_path, 'rb') as f:
      combined_state = pickle.load(f)
    set_tf_state(combined_state['state'])
    restored = True
  else:
    _log.info('not restoring any params')

  if restored:
    train_loss = train_manager.step()['loss']
    _log.info('loss post-restore: %f', train_loss.numpy())

  FRAMES_PER_MINUTE = 60 * 60

  for _ in range(_config['num_epochs']):
    start_time = time.perf_counter()

    # train for epoch_time seconds
    steps = 0
    num_frames = 0
    while True:
      train_stats = train_manager.step()
      steps += 1
      num_frames += train_stats['num_frames']

      elapsed_time = time.perf_counter() - start_time
      if elapsed_time > _config['epoch_time']: break

    step.assign_add(steps)
    total_steps = step.numpy()

    # now test
    test_stats = test_manager.step()

    train_loss = train_stats['loss'].numpy()
    test_loss = test_stats['loss'].numpy()
    epoch = train_stats['epoch']

    all_stats = dict(
        train=train_stats,
        test=test_stats,
        learning_rate=learning_rate.numpy(),
    )
    train_lib.log_stats(ex, all_stats, total_steps)

    sps = steps / elapsed_time
    mps = num_frames / FRAMES_PER_MINUTE / elapsed_time
    ex.log_scalar('sps', sps, total_steps)
    ex.log_scalar('mps', mps, total_steps)

    print(f'steps={total_steps} sps={sps:.2f} mps={mps:.2f} epoch={epoch:.3f}')
    print(f'losses: train={train_loss:.4f} test={test_loss:.4f}')
    print(f'timing:'
          f' data={train_manager.data_profiler.mean_time():.3f}'
          f' step={train_manager.step_profiler.mean_time():.3f}')
    print()

    save_path = save()
    if save_path:
      _log.info('Saved network to %s', save_path)
