"""Train (and test) a network via imitation learning."""

import collections
import dataclasses
import datetime
import json
import os
import pickle
import secrets
import time
import typing as tp

from absl import logging

import numpy as np
import tensorflow as tf
import tree

import wandb

from slippi_ai import (
    controller_heads,
    flag_utils,
    nametags,
    networks,
    policies,
    saving,
    tf_utils,
    train_lib,
    utils,
)
from slippi_ai import learner as learner_lib
from slippi_ai import data as data_lib
from slippi_ai import value_function as vf_lib
from slippi_ai import embed as embed_lib


def get_experiment_tag():
  today = datetime.date.today()
  return f'{today.year}-{today.month}-{today.day}_{secrets.token_hex(8)}'


class TrainManager:

  def __init__(
      self,
      learner: learner_lib.Learner,
      data_source: data_lib.DataSource,
      step_kwargs={},
  ):
    self.learner = learner
    self.data_source = data_source
    self.hidden_state = learner.initial_state(data_source.batch_size)
    self.step_kwargs = step_kwargs
    self.data_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()

  def step(self, compiled: bool = True) -> tuple[dict, data_lib.Batch]:
    with self.data_profiler:
      batch, epoch = next(self.data_source)
    with self.step_profiler:
      stats, self.hidden_state = self.learner.step(
          batch, self.hidden_state, compile=compiled, **self.step_kwargs)
    stats.update(
        epoch=epoch,
    )
    return stats, batch

def mean(value):
  if isinstance(value, tf.Tensor):
    value = value.numpy()
  if isinstance(value, np.ndarray):
    value = value.mean().item()
  return value

def log_stats(
    stats: tree.Structure,
    step: tp.Optional[int] = None,
    take_mean: bool = True,
):
  if take_mean:
    stats = tree.map_structure(mean, stats)
  wandb.log(data=stats, step=step)

_field = utils.field

@dataclasses.dataclass
class RuntimeConfig:
  max_runtime: int = 1 * 60 * 60  # maximum runtime in seconds
  log_interval: int = 10  # seconds between logging
  save_interval: int = 300  # seconds between saving to disk

  eval_every_n: int = 100  # number of training steps between evaluations
  num_eval_steps: int = 10  # number of batches per evaluation

@dataclasses.dataclass
class ValueFunctionConfig:
  train_separate_network: bool = True
  separate_network_config: bool = True
  network: dict = _field(lambda: networks.DEFAULT_CONFIG)

@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = _field(RuntimeConfig)

  dataset: data_lib.DatasetConfig = _field(data_lib.DatasetConfig)
  data: data_lib.DataConfig = _field(data_lib.DataConfig)

  learner: learner_lib.LearnerConfig = _field(learner_lib.LearnerConfig)

  # TODO: turn these into dataclasses too
  network: dict = _field(lambda: networks.DEFAULT_CONFIG)
  controller_head: dict = _field(lambda: controller_heads.DEFAULT_CONFIG)

  embed: embed_lib.EmbedConfig = _field(embed_lib.EmbedConfig)

  policy: policies.PolicyConfig = _field(policies.PolicyConfig)
  value_function: ValueFunctionConfig = _field(ValueFunctionConfig)

  max_names: int = 16

  expt_root: str = 'experiments'
  expt_dir: tp.Optional[str] = None
  tag: tp.Optional[str] = None

  # TODO: group these into their own subconfig
  restore_pickle: tp.Optional[str] = None

  is_test: bool = False  # for db management
  version: int = saving.VERSION

def _get_loss(stats: dict):
  return stats['total_loss'].numpy().mean()

def create_name_map(
    replays: list[data_lib.ReplayInfo],
    max_names: int,
) -> dict[str, int]:
  name_map = {}
  name_counts = collections.Counter()

  normalized_names = [
      nametags.normalize_name(replay.main_player.name)
      for replay in replays]
  name_counts.update(normalized_names)

  for i, (name, _) in enumerate(name_counts.most_common(max_names)):
    name_map[name] = i

  # Bake in name groups from nametags.py
  for first, *rest in nametags.NAME_GROUPS:
    if first not in name_map:
      continue
    for name in rest:
      name_map[name] = name_map[first]

  return name_map

def train(config: Config):
  tag = config.tag or train_lib.get_experiment_tag()
  # Might want to use wandb.run.dir instead, but it doesn't seem
  # to be set properly even when we try to override it.
  expt_dir = config.expt_dir
  if expt_dir is None:
    expt_dir = os.path.join(config.expt_root, tag)
    os.makedirs(expt_dir, exist_ok=True)
  config.expt_dir = expt_dir  # for wandb logging
  logging.info('experiment directory: %s', expt_dir)

  runtime = config.runtime

  pickle_path = os.path.join(expt_dir, 'latest.pkl')



  # attempt to restore parameters
  restored = False
  if config.restore_pickle:
    logging.info('restoring from %s', config.restore_pickle)
    with open(config.restore_pickle, 'rb') as f:
      combined_state = pickle.load(f)
    restored = True
  elif os.path.exists(pickle_path):
    logging.info('restoring from %s', pickle_path)
    with open(pickle_path, 'rb') as f:
      combined_state = pickle.load(f)
    restored = True
  else:
    logging.info('not restoring any params')

  if restored:
    restore_config = flag_utils.dataclass_from_dict(
        Config, saving.upgrade_config(combined_state['config']))

    # We can update the delay as it doesn't affect the network architecture.
    if restore_config.policy.delay != config.policy.delay:
      logging.warning(f'Changing delay from {restore_config.policy.delay} to {config.policy.delay}.')

    # These we can't change after the fact.
    for key in ['network', 'controller_head', 'embed']:
      current = getattr(config, key)
      previous = getattr(restore_config, key)
      if current != previous:
        logging.warning(f'Requested {key} config doesn\'t match, overriding from checkpoint.')
        setattr(config, key, previous)

  policy = saving.policy_from_config(dataclasses.asdict(config))

  value_function = None
  if config.value_function.train_separate_network:
    value_net_config = config.network
    if config.value_function.separate_network_config:
      value_net_config = config.value_function.network
    value_function = vf_lib.ValueFunction(
        network_config=value_net_config,
        embed_state_action=policy.embed_state_action,
    )

  learner_kwargs = dataclasses.asdict(config.learner)
  learning_rate = tf.Variable(
      learner_kwargs['learning_rate'], name='learning_rate', trainable=False)
  learner_kwargs.update(learning_rate=learning_rate)
  learner = learner_lib.Learner(
      policy=policy,
      value_function=value_function,
      **learner_kwargs,
  )

  logging.info("Network configuration")
  for comp in ['network', 'controller_head']:
    logging.info(f'Using {comp}: {getattr(config, comp)["name"]}')

  ### Dataset Creation ###
  dataset_config = config.dataset

  # Parse csv chars into list of enum values.
  char_filters = {}
  for key in ['allowed_characters', 'allowed_opponents']:
    chars_string = getattr(dataset_config, key)
    char_filters[key] = data_lib.chars_from_string(chars_string)

  train_replays, test_replays = data_lib.train_test_split(dataset_config)
  logging.info(f'Training on {len(train_replays)} replays, testing on {len(test_replays)}')

  if restored:
    name_map: dict[str, int] = combined_state['name_map']
  else:
    name_map = create_name_map(train_replays, config.max_names)

  # Record name map
  print(name_map)
  name_map_path = os.path.join(expt_dir, 'name_map.json')
  with open(name_map_path, 'w') as f:
    json.dump(name_map, f)
  wandb.save(name_map_path, policy='now')

  num_codes = nametags.max_name_code(name_map) + 1
  encode_name = nametags.name_encoder(name_map)
  encode_name_uint8 = lambda name: np.uint8(encode_name(name))
  batch_encode_name = np.vectorize(encode_name_uint8)

  # Create data sources for train and test.
  data_config = dict(
      dataclasses.asdict(config.data),
      extra_frames=1 + policy.delay,
      name_map=name_map,
      **char_filters,
  )
  train_data = data_lib.make_source(replays=train_replays, **data_config)
  test_data = data_lib.make_source(replays=test_replays, **data_config)
  del train_replays, test_replays  # free up memory

  train_manager = train_lib.TrainManager(learner, train_data, dict(train=True))
  test_manager = train_lib.TrainManager(learner, test_data, dict(train=False))

  # initialize variables
  train_stats, _ = train_manager.step()
  logging.info('loss initial: %f', _get_loss(train_stats))

  with tf.device('/cpu:0'):
    step = tf.Variable(0, trainable=False, name="step", dtype=tf.int64)

  # saving and restoring
  tf_state = dict(
      step=step,
      policy=policy.variables,
      value_function=value_function.variables if value_function else [],
      optimizers=dict(
          policy=learner.policy_optimizer.variables,
          value=learner.value_optimizer.variables,
      ),
  )

  def get_tf_state():
    return tf.nest.map_structure(lambda v: v.numpy(), tf_state)

  def set_tf_state(state):
    tf.nest.map_structure(
      lambda var, val: var.assign(val),
      tf_state, state)

  # Initialize the best eval loss to infinity
  best_eval_loss = float('inf')

  def save(eval_loss=None):
    # Local Save
    tf_state = get_tf_state()

    # easier to always bundle the config with the state
    combined_state = dict(
        state=tf_state,
        config=dataclasses.asdict(config),
        name_map=name_map,
        best_eval_loss=eval_loss if eval_loss is not None else best_eval_loss,
    )
    pickled_state = pickle.dumps(combined_state)

    logging.info('saving state to %s', pickle_path)
    with open(pickle_path, 'wb') as f:
      f.write(pickled_state)


  if restored:
    set_tf_state(combined_state['state'])
    best_eval_loss = combined_state.get('best_eval_loss', float('inf'))
    train_loss = _get_loss(train_manager.step()[0])
    logging.info('loss post-restore: %f', train_loss)

  FRAMES_PER_MINUTE = 60 * 60
  FRAMES_PER_STEP = config.data.batch_size * config.data.unroll_length

  step_tracker = utils.Tracker(step.numpy())
  epoch_tracker = utils.Tracker(train_stats['epoch'])
  log_tracker = utils.Tracker(time.time())

  @utils.periodically(runtime.log_interval)
  def maybe_log(train_stats: dict):
    """Do a test step, then log both train and test stats."""
    test_stats, _ = test_manager.step()

    elapsed_time = log_tracker.update(time.time())
    total_steps = step.numpy()
    steps = step_tracker.update(total_steps)
    num_frames = steps * FRAMES_PER_STEP

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
        num_frames=num_frames,
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
    nonlocal best_eval_loss  # Allow modification of the best_eval_loss variable
    total_steps = int(step.numpy())
    if total_steps % runtime.eval_every_n != 0:
      return

    eval_stats = []
    batches = []

    for _ in range(runtime.num_eval_steps):
      stats, batch = test_manager.step()
      # Convert to numpy to free up GPU memory.
      eval_stats.append(utils.map_nt(tf_utils.to_numpy, stats))
      batches.append(batch)

    eval_stats = tf.nest.map_structure(utils.stack, *eval_stats)

    total_frames = total_steps * FRAMES_PER_STEP
    train_epoch = epoch_tracker.last
    counters = dict(
        total_frames=total_frames,
        train_epoch=train_epoch,
    )

    to_log = dict(eval=eval_stats, **counters)
    train_lib.log_stats(to_log, total_steps)

    # Calculate the mean eval loss
    eval_loss = eval_stats['policy']['loss'].mean()
    logging.info('eval loss: %f', eval_loss)

    # Save if the eval loss is the best so far
    if eval_loss < best_eval_loss:
      logging.info('New best eval loss: %f (previous: %f)', eval_loss, best_eval_loss)
      best_eval_loss = eval_loss
      save(eval_loss=best_eval_loss)

    # Log losses aggregated by name.

    # Stats have shape [num_eval_steps, unroll_length, batch_size]
    time_mean = lambda x: np.mean(x, axis=1)
    loss = time_mean(eval_stats['policy']['loss'])
    assert loss.shape == (runtime.num_eval_steps, config.data.batch_size)

    # Metadata only has a batch dimension.
    metas: list[data_lib.ChunkMeta] = [batch.meta for batch in batches]
    meta: data_lib.ChunkMeta = tf.nest.map_structure(utils.stack, *metas)

    # Name of the player we're imitating.
    name = np.where(
        meta.info.swap, meta.info.meta.p1.name, meta.info.meta.p0.name)
    encoded_name = batch_encode_name(name)
    assert encoded_name.dtype == np.uint8
    assert encoded_name.shape == loss.shape

    loss_sums_and_counts = []
    for i in range(num_codes):
      mask = encoded_name == i
      loss_sums_and_counts.append((np.sum(loss * mask), np.sum(mask)))

    losses, counts = zip(*loss_sums_and_counts)
    to_log = dict(
        losses=np.array(losses, dtype=np.float32),
        counts=np.array(counts, dtype=np.uint32),
    )

    to_log = dict(eval_names=to_log, **counters)
    train_lib.log_stats(to_log, total_steps, take_mean=False)

  start_time = time.time()

  while time.time() - start_time < runtime.max_runtime:
    train_stats, _ = train_manager.step()
    step.assign_add(1)
    maybe_log(train_stats)
    maybe_eval()
