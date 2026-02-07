"""Train (and test) a network via imitation learning."""

import collections
import contextlib
import dataclasses
import datetime
import json
import os
import pickle
import queue
import secrets
import threading
import time
import typing as tp

from absl import logging

import numpy as np
import tensorflow as tf
import tree

import wandb

import melee

from slippi_ai import (
    flag_utils,
    nametags,
    utils,
)
from slippi_ai import observations as obs_lib
from slippi_ai import data as data_lib
from slippi_ai.policies import Platform
from slippi_ai.tf import learner as learner_lib, networks, policies, saving, tf_utils, train_lib
from slippi_ai.tf import value_function as vf_lib
from slippi_ai.tf import embed as embed_lib
from slippi_ai.tf import controller_heads

def get_experiment_tag():
  today = datetime.date.today()
  return f'{today.year}-{today.month}-{today.day}_{secrets.token_hex(8)}'


class TrainManager:

  def __init__(
      self,
      learner: learner_lib.Learner,
      data_source: data_lib.AbstractDataSource,
      step_kwargs={},
      prefetch: int = 16,
  ):
    self.learner = learner
    self.data_source = data_source
    self.hidden_state = learner.initial_state(data_source.batch_size)
    self.step_kwargs = step_kwargs
    self.data_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()
    self.last_epoch = 0.

    self.frames_queue = queue.Queue(maxsize=prefetch)
    self.stop_requested = threading.Event()

    self.data_thread = threading.Thread(target=self.produce_frames)
    self.data_thread.start()

  def produce_frames(self):
    while not self.stop_requested.is_set():
      batch, epoch = next(self.data_source)
      frames = batch.frames

      if np.any(frames.is_resetting[:, 1:]):
        raise ValueError("Unexpected mid-episode reset.")

      frames = frames._replace(
          state_action=self.learner.policy.network.encode(frames.state_action))
      frames = utils.map_nt(tf.convert_to_tensor, frames)
      data = (batch, epoch, frames)

      # Try to put data into the queue, but check for stop_requested
      while not self.stop_requested.is_set():
        try:
          self.frames_queue.put(data, timeout=1)
          break
        except queue.Full:
          continue

  def stop(self):
    self.stop_requested.set()
    self.data_thread.join()

  def step(self, compiled: tp.Optional[bool] = None) -> tuple[dict, data_lib.Batch]:
    with self.data_profiler:
      frames_queue_size = self.frames_queue.qsize()
      batch, epoch, frames = self.frames_queue.get()
    with self.step_profiler:
      stats, self.hidden_state = self.learner.step(
          frames, self.hidden_state, compile=compiled, **self.step_kwargs)

    self.last_epoch = epoch
    stats.update(
        epoch=epoch,
        frames_queue_size=frames_queue_size,
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
    stats = utils.map_nt(mean, stats)
  wandb.log(data=stats, step=step)

_field = utils.field

@dataclasses.dataclass
class RuntimeConfig:
  max_runtime: int = 1 * 60 * 60  # maximum runtime in seconds
  log_interval: int = 10  # seconds between logging
  save_interval: int = 300  # seconds between saving to disk

  num_evals_per_epoch: float = 1  # number evaluations per training epoch
  eval_at_start: bool = False  # do an eval at the start of training
  num_eval_epochs: float = 1  # number of test-set epochs per evaluation

  max_eval_steps: tp.Optional[int] = None  # used in tests

@dataclasses.dataclass
class ValueFunctionConfig:
  train_separate_network: bool = True
  separate_network_config: bool = True
  network: dict = _field(networks.default_config)

@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = _field(RuntimeConfig)

  dataset: data_lib.DatasetConfig = _field(data_lib.DatasetConfig)
  data: data_lib.DataConfig = _field(data_lib.DataConfig)
  observation: obs_lib.ObservationConfig = _field(obs_lib.ObservationConfig)

  learner: learner_lib.LearnerConfig = _field(learner_lib.LearnerConfig)

  # TODO: turn these into dataclasses too
  network: dict = _field(networks.default_config)
  controller_head: dict = _field(controller_heads.default_config)

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
  platform: str = Platform.TF.value

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

def value_function_from_config(
    config: Config
) -> tp.Optional[vf_lib.ValueFunction]:
  vf_config = config.value_function
  if vf_config.train_separate_network:
    network_config = config.network
    if vf_config.separate_network_config:
      network_config = vf_config.network

    return vf_lib.ValueFunction(
        network_config=network_config,
        num_names=config.max_names,
        embed_config=config.embed,
    )
  else:
    return None

def train(config: Config):
  with contextlib.ExitStack() as exit_stack:
    _train(config, exit_stack)

def _train(config: Config, exit_stack: contextlib.ExitStack):
  tag = config.tag or train_lib.get_experiment_tag()
  # Might want to use wandb.run.dir instead, but it doesn't seem
  # to be set properly even when we try to override it.
  expt_dir = config.expt_dir
  if expt_dir is None:
    expt_dir = os.path.join(config.expt_root, tag)
    os.makedirs(expt_dir, exist_ok=True)
  config.expt_dir = expt_dir  # for wandb logging
  logging.info('experiment directory: %s', expt_dir)

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

  # Initialize the best eval loss to infinity
  best_eval_loss = float('inf')

  if restored:
    best_eval_loss = combined_state.get('best_eval_loss', float('inf'))

    restore_config = flag_utils.dataclass_from_dict(
        Config, saving.upgrade_config(combined_state['config']))

    # We can update the delay as it doesn't affect the network architecture.
    if restore_config.policy.delay != config.policy.delay:
      logging.warning(f'Changing delay from {restore_config.policy.delay} to {config.policy.delay}.')
      best_eval_loss = float('inf')  # Old losses don't apply to new delay.

    # These we can't change after the fact.
    for key in ['network', 'controller_head', 'embed', 'value_function']:
      current = getattr(config, key)
      previous = getattr(restore_config, key)
      if current != previous:
        logging.warning(f'Requested {key} config doesn\'t match, overriding from checkpoint.')
        setattr(config, key, previous)

    if (config.dataset.allowed_characters != restore_config.dataset.allowed_characters or
        config.dataset.allowed_opponents != restore_config.dataset.allowed_opponents):
      logging.warning('Dataset character/opponent filters changed, resetting best eval loss.')
      best_eval_loss = float('inf')
      config.runtime.eval_at_start = True

  policy = saving.policy_from_config(dataclasses.asdict(config))
  value_function = value_function_from_config(config)

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

  character_quantities = collections.Counter()
  for replay in train_replays:
    character_quantities[melee.Character(replay.main_player.character)] += 1
  dataset_metrics = {
      'characters': dict(character_quantities),
  }

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
      observation_config=config.observation,
      **char_filters,
  )
  train_data = data_lib.make_source(replays=train_replays, **data_config)

  test_batch_size = 2 * config.data.batch_size
  test_data_config = dict(
      data_config,
      # Use more workers for test data to keep up with eval speed.
      num_workers=2 * config.data.num_workers,
      batch_size=test_batch_size,
  )
  test_data = data_lib.make_source(replays=test_replays, **test_data_config)
  del train_replays, test_replays  # free up memory

  train_manager = train_lib.TrainManager(learner, train_data, dict(train=True))
  test_manager = train_lib.TrainManager(learner, test_data, dict(train=False))

  # TrainManager should probably be a proper context manager.
  exit_stack.callback(train_manager.stop)
  exit_stack.callback(test_manager.stop)

  runtime = config.runtime

  # initialize variables
  if config.learner.minibatch_size > 0:
    # TODO: figure out why this is needed when minibatching is on
    test_manager.step()
  train_stats, _ = train_manager.step()
  logging.info('Initialized policy with %d variables', len(policy.variables))
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

  def save(eval_loss=None):
    # Local Save
    tf_state = get_tf_state()

    # easier to always bundle the config with the state
    combined_state = dict(
        state=tf_state,
        config=dataclasses.asdict(config),
        name_map=name_map,
        best_eval_loss=eval_loss if eval_loss is not None else best_eval_loss,
        dataset_metrics=dataset_metrics,
    )
    pickled_state = pickle.dumps(combined_state)

    logging.info('saving state to %s', pickle_path)
    with open(pickle_path, 'wb') as f:
      f.write(pickled_state)


  if restored:
    set_tf_state(combined_state['state'])
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

  allowed_characters = data_lib.chars_from_string(
      config.dataset.allowed_characters)
  if allowed_characters is None:
    allowed_characters = list(melee.Character)

  last_train_epoch_evaluated = 0.
  needs_initial_eval = runtime.eval_at_start

  def maybe_eval():
    nonlocal best_eval_loss
    nonlocal last_train_epoch_evaluated
    nonlocal needs_initial_eval

    # Check whether we need to run an evaluation
    train_epoch = train_manager.last_epoch
    if (
      (train_epoch - last_train_epoch_evaluated) * runtime.num_evals_per_epoch < 1
      and not needs_initial_eval):
      return
    last_train_epoch_evaluated = train_epoch
    needs_initial_eval = False

    per_step_eval_stats: list[dict] = []
    metas: list[data_lib.ChunkMeta] = []

    def time_mean(x):
      # Stats are either scalars or (time, batch)-shaped.
      x = tf_utils.to_numpy(x)

      if isinstance(x, np.ndarray):
        if len(x.shape) == 0:
          return x.item()

        return np.mean(x, axis=0)

      return x

    logging.info('Starting evaluation at train epoch %.3f', train_epoch)
    start_time = time.perf_counter()
    test_epoch = test_manager.last_epoch
    while (
      (test_manager.last_epoch - test_epoch) < runtime.num_eval_epochs
      and (runtime.max_eval_steps is None or len(per_step_eval_stats) < runtime.max_eval_steps)):
      stats, batch = test_manager.step()

      # Convert to numpy and take time-mean to free up memory.
      stats = utils.map_single_structure(time_mean, stats)

      per_step_eval_stats.append(stats)
      metas.append(batch.meta)

    eval_time = time.perf_counter() - start_time

    # [eval_steps, batch_size], mean taken over time
    eval_stats = utils.batch_nest_nt(per_step_eval_stats)

    data_time = test_manager.data_profiler.mean_time()
    step_time = test_manager.step_profiler.mean_time()

    sps = len(per_step_eval_stats) / eval_time
    frames_per_step = test_batch_size * config.data.unroll_length
    mps = sps * frames_per_step / FRAMES_PER_MINUTE

    total_steps = int(step.numpy())
    total_frames = total_steps * FRAMES_PER_STEP
    train_epoch = epoch_tracker.last
    counters = dict(
        total_frames=total_frames,
        train_epoch=train_epoch,
        data_time=data_time,
        step_time=step_time,
        sps=sps,
        mps=mps,
    )

    to_log = dict(
        counters,
        eval=utils.map_nt(mean, eval_stats),
    )

    # Calculate the mean eval loss
    eval_loss = eval_stats['policy']['loss'].mean()
    logging.info('eval loss: %.4f data: %.3f step: %.3f mps: %.1f', eval_loss, data_time, step_time, mps)

    # Save if the eval loss is the best so far
    if eval_loss < best_eval_loss:
      logging.info('New best eval loss: %f (previous: %f)', eval_loss, best_eval_loss)
      best_eval_loss = eval_loss
      save(eval_loss=best_eval_loss)

    # Stats have shape [num_eval_steps, batch_size]
    loss = eval_stats['policy']['loss']
    assert loss.shape == (len(per_step_eval_stats), test_batch_size)

    meta = utils.batch_nest_nt(metas)

    # Name of the player we're imitating.
    name = meta.info.main_player.name
    encoded_name: np.ndarray = batch_encode_name(name)
    assert encoded_name.dtype == np.uint8
    assert encoded_name.shape == loss.shape

    loss_sums_and_counts = []
    for i in range(num_codes):
      mask = encoded_name == i
      loss_sums_and_counts.append((np.sum(loss * mask), np.sum(mask)))

    losses, counts = zip(*loss_sums_and_counts)
    to_log['eval_names'] = dict(
        losses=np.array(losses, dtype=np.float32),
        counts=np.array(counts, dtype=np.uint32),
    )

    # Log losses aggregated by character
    if len(allowed_characters) > 1:
      characters = meta.info.main_player.character
      per_character_loss_sums = {}
      per_character_loss_counts = {}
      for character in allowed_characters:
        mask = character.value == characters
        name = character.name.lower()
        per_character_loss_sums[name] = np.sum(loss * mask)
        per_character_loss_counts[name] = np.sum(mask)

      to_log['eval_characters'] = dict(
          losses=per_character_loss_sums,
          counts=per_character_loss_counts,
      )

    log_stats(to_log, total_steps, take_mean=False)

  start_time = time.time()

  while time.time() - start_time < runtime.max_runtime:
    maybe_eval()

    train_stats, _ = train_manager.step()
    step.assign_add(1)
    maybe_log(train_stats)
