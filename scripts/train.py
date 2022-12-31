"""Train (and test) a network via imitation learning."""

import dataclasses
import os
import pickle
import time

import sacred
import tensorflow as tf

from slippi_ai import (
    controller_heads,
    embed,
    networks,
    policies,
    saving,
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

@dataclasses.dataclass
class RuntimeConfig:
  max_runtime: int = 1 * 60 * 60  # maximum runtime in seconds
  log_interval: int = 10  # seconds between logging
  save_interval: int = 300  # seconds between saving to disk

  eval_every_n: int = 100  # number of training steps between evaluations
  num_eval_steps: int = 10  # number of batches per evaluation

@ex.config
def config():
  runtime = dataclasses.asdict(RuntimeConfig())

  dataset = data_lib.DATASET_CONFIG
  data = data_lib.CONFIG
  learner = Learner.DEFAULT_CONFIG
  network = networks.DEFAULT_CONFIG
  controller_head = controller_heads.DEFAULT_CONFIG
  policy = policies.DEFAULT_CONFIG

  expt_dir = train_lib.get_experiment_directory()
  tag = train_lib.get_experiment_tag()
  save_to_s3 = False
  restore_tag = None

def _get_loss(stats: dict):
  return stats['total_loss'].numpy().mean()

@ex.automain
def main(expt_dir, _config, _log):
  runtime = RuntimeConfig(**_config['runtime'])

  embed_controller = embed.embed_controller_discrete  # TODO: configure

  policy = saving.build_policy(
      controller_head_config=_config['controller_head'],
      max_action_repeat=_config['data']['max_action_repeat'],
      network_config=_config['network'],
      embed_controller=embed_controller,
      **_config['policy'],
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
  _log.info('loss initial: %f', _get_loss(train_stats))

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
        config=dict(_config, version=saving.VERSION),
    )
    pickled_state = pickle.dumps(combined_state)

    _log.info('saving state to %s', pickle_path)
    with open(pickle_path, 'wb') as f:
      f.write(pickled_state)

    if save_to_s3:
      _log.info('saving state to S3: %s', s3_keys.combined)
      s3_store.put(s3_keys.combined, pickled_state)

  save = utils.Periodically(save, runtime.save_interval)

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
    train_loss = _get_loss(train_manager.step())
    _log.info('loss post-restore: %f', train_loss.numpy())

  FRAMES_PER_MINUTE = 60 * 60

  step_tracker = utils.Tracker(step.numpy())
  epoch_tracker = utils.Tracker(train_stats['epoch'])
  log_tracker = utils.Tracker(time.time())

  @utils.periodically(runtime.log_interval)
  def maybe_log(train_stats: dict):
    """Do a test step, then log both train and test stats."""
    test_stats = test_manager.step()

    elapsed_time = log_tracker.update(time.time())
    total_steps = step.numpy()
    steps = step_tracker.update(total_steps)
    # assume num_frames is constant per step
    num_frames = steps * train_stats['num_frames']

    epoch = train_stats['epoch']
    delta_epoch = epoch_tracker.update(epoch)

    sps = steps / elapsed_time
    mps = num_frames / FRAMES_PER_MINUTE / elapsed_time
    eph = delta_epoch / elapsed_time * 60 * 60
    data_time = train_manager.data_profiler.mean_time()
    step_time = train_manager.step_profiler.mean_time()

    timings = dict(
        sps=sps,
        mps=mps,
        eph=eph,
        data=data_time,
        step=step_time,
    )

    all_stats = dict(
        train=train_stats,
        test=test_stats,
        timings=timings,
    )
    train_lib.log_stats(ex, all_stats, total_steps)

    train_loss = _get_loss(train_stats)
    test_loss = _get_loss(test_stats)

    print(f'step={total_steps} epoch={epoch:.3f}')
    print(f'sps={sps:.2f} mps={mps:.2f} eph={eph:.2e}')
    print(f'losses: train={train_loss:.4f} test={test_loss:.4f}')
    print(f'timing:'
          f' data={data_time:.3f}'
          f' step={step_time:.3f}')
    print()

  def maybe_eval():
    total_steps = step.numpy()
    if total_steps % runtime.eval_every_n != 0:
      return

    eval_stats = [test_manager.step() for _ in range(runtime.num_eval_steps)]
    eval_stats = tf.nest.map_structure(utils.to_numpy, eval_stats)
    eval_stats = tf.nest.map_structure(utils.stack, *eval_stats)

    to_log = dict(eval=eval_stats)
    train_lib.log_stats(ex, to_log, total_steps)

  start_time = time.time()

  while time.time() - start_time < runtime.max_runtime:
    train_stats = train_manager.step()
    step.assign_add(1)
    maybe_log(train_stats)
    maybe_eval()

    save_path = save()
    if save_path:
      _log.info('Saved network to %s', save_path)
