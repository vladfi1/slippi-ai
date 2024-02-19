"""Train (and test) a network via imitation learning."""

import dataclasses
import os
import pickle
import time
import typing as tp

from absl import app
from absl import logging

import fancyflags as ff
import numpy as np
import tensorflow as tf

import wandb

from slippi_ai import (
    controller_heads,
    embed,
    flag_utils,
    networks,
    policies,
    saving,
    s3_lib,
    train_lib,
    utils,
)
from slippi_ai import learner as learner_lib
from slippi_ai import data as data_lib
from slippi_ai.file_cache import FileCache

T = tp.TypeVar('T')

def _field(default_factory: tp.Callable[[], T]) -> T:
  return dataclasses.field(default_factory=default_factory)

@dataclasses.dataclass
class RuntimeConfig:
  max_runtime: int = 1 * 60 * 60  # maximum runtime in seconds
  log_interval: int = 10  # seconds between logging
  save_interval: int = 300  # seconds between saving to disk

  eval_every_n: int = 100  # number of training steps between evaluations
  num_eval_steps: int = 10  # number of batches per evaluation

@dataclasses.dataclass
class FileCacheConfig:
  use: bool = False
  path: tp.Optional[str] = None
  wipe: bool = False
  version: str = 'test'

@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = _field(RuntimeConfig)

  file_cache: FileCacheConfig = _field(FileCacheConfig)
  dataset: data_lib.DatasetConfig = _field(data_lib.DatasetConfig)
  data: data_lib.DataConfig = _field(data_lib.DataConfig)

  learner: learner_lib.LearnerConfig = _field(learner_lib.LearnerConfig)

  # TODO: turn these into dataclasses too
  network: dict = _field(lambda: networks.DEFAULT_CONFIG)
  controller_head: dict = _field(lambda: controller_heads.DEFAULT_CONFIG)

  policy: policies.PolicyConfig = _field(policies.PolicyConfig)

  expt_dir: tp.Optional[str] = None
  tag: tp.Optional[str] = None

  # TODO: group these into their own subconfig
  save_to_s3: bool = False
  restore_tag: tp.Optional[str] = None
  restore_pickle: tp.Optional[str] = None

  is_test: bool = False  # for db management
  version: int = saving.VERSION

def _get_loss(stats: dict):
  return stats['total_loss'].numpy().mean()

def train(config: Config):
  runtime = config.runtime

  embed_controller = embed.embed_controller_discrete  # TODO: configure

  policy = saving.build_policy(
      controller_head_config=config.controller_head,
      network_config=config.network,
      embed_controller=embed_controller,
      **dataclasses.asdict(config.policy),
  )

  learner_kwargs = dataclasses.asdict(config.learner)
  learning_rate = tf.Variable(
      learner_kwargs['learning_rate'], name='learning_rate', trainable=False)
  learner_kwargs.update(learning_rate=learning_rate)
  learner = learner_lib.Learner(
      policy=policy,
      **learner_kwargs,
  )

  logging.info("Network configuration")
  for comp in ['network', 'controller_head']:
    logging.info(f'Using {comp}: {getattr(config, comp)["name"]}')

  ### Dataset Creation ###
  dataset_config = config.dataset

  file_cache_config = config.file_cache
  if file_cache_config.use:
    file_cache = FileCache(
        root=file_cache_config.path,
        wipe=file_cache_config.wipe,
    )

    file_cache.pull_dataset(file_cache_config.version)

    dataset_config.data_dir = file_cache.games_dir
    dataset_config.meta_path = file_cache.meta_path

  # Parse csv chars into list of enum values.
  char_filters = {}
  for key in ['allowed_characters', 'allowed_opponents']:
    chars_string = getattr(dataset_config, key)
    char_filters[key] = data_lib.chars_from_string(chars_string)

  train_replays, test_replays = data_lib.train_test_split(dataset_config)
  logging.info(f'Training on {len(train_replays)} replays, testing on {len(test_replays)}')

  # Create data sources for train and test.
  data_config = dict(
      dataclasses.asdict(config.data),
      embed_controller=embed_controller,
      extra_frames=1 + policy.delay,
      **char_filters,
  )
  train_data = data_lib.make_source(replays=train_replays, **data_config)
  test_data = data_lib.make_source(replays=test_replays, **data_config)

  train_manager = train_lib.TrainManager(learner, train_data, dict(train=True))
  test_manager = train_lib.TrainManager(learner, test_data, dict(train=False))

  # initialize variables
  train_stats = train_manager.step()
  logging.info('loss initial: %f', _get_loss(train_stats))

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

  tag = config.tag or train_lib.get_experiment_tag()
  expt_dir = config.expt_dir
  if expt_dir is None:
    expt_dir = f'experiments/{tag}'
    os.makedirs(expt_dir, exist_ok=True)
  pickle_path = os.path.join(expt_dir, 'latest.pkl')

  save_to_s3 = config.save_to_s3
  if save_to_s3 or config.restore_tag:
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
        config=dataclasses.asdict(config),
    )
    pickled_state = pickle.dumps(combined_state)

    logging.info('saving state to %s', pickle_path)
    with open(pickle_path, 'wb') as f:
      f.write(pickled_state)

    if save_to_s3:
      logging.info('saving state to S3: %s', s3_keys.combined)
      s3_store.put(s3_keys.combined, pickled_state)

  save = utils.Periodically(save, runtime.save_interval)

  if config.restore_tag:
    restore_s3_keys = s3_lib.get_keys(config.restore_tag)
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
      logging.info('restoring from %s', restore_key)
      combined_state = pickle.loads(obj)
      set_tf_state(combined_state['state'])
      restored = True
      # TODO: do some config compatibility validation
    except KeyError:
      # TODO: re-raise if user specified restore_tag
      logging.info('no params found at %s', restore_key)
  elif config.restore_pickle:
    logging.info('restoring from %s', config.restore_pickle)
    with open(config.restore_pickle, 'rb') as f:
      combined_state = pickle.load(f)
    set_tf_state(combined_state['state'])
    restored = True
  elif os.path.exists(pickle_path):
    logging.info('restoring from %s', pickle_path)
    with open(pickle_path, 'rb') as f:
      combined_state = pickle.load(f)
    set_tf_state(combined_state['state'])
    restored = True
  else:
    logging.info('not restoring any params')

  if restored:
    train_loss = _get_loss(train_manager.step())
    logging.info('loss post-restore: %f', train_loss)

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
    train_lib.log_stats(all_stats, total_steps)

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
    train_lib.log_stats(to_log, total_steps)

  start_time = time.time()

  while time.time() - start_time < runtime.max_runtime:
    train_stats = train_manager.step()
    step.assign_add(1)
    maybe_log(train_stats)
    maybe_eval()

    save_path = save()
    if save_path:
      logging.info('Saved network to %s', save_path)

CONFIG = ff.DEFINE_dict(
    'config', **flag_utils.get_flags_from_dataclass(Config))

# passed to wandb.init
WANDB = ff.DEFINE_dict(
    'wandb',
    project=ff.String('slippi-ai'),
    mode=ff.Enum('disabled', ['online', 'offline', 'disabled']),
    group=ff.String('imitation'),
    name=ff.String(None),
    notes=ff.String(None),
)

def main(_):
  config = flag_utils.dataclass_from_dict(Config, CONFIG.value)

  wandb_kwargs = dict(WANDB.value)
  if config.tag:
    wandb_kwargs['name'] = config.tag
  wandb.init(
      config=CONFIG.value,
      **wandb_kwargs,
  )
  train(config)

if __name__ == '__main__':
  app.run(main)
