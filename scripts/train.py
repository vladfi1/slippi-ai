"""Train (and test) a network via imitation learning."""

import contextlib
import collections
import dataclasses
import json
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

import melee

from slippi_ai import (
    controller_heads,
    dolphin,
    evaluators,
    flag_utils,
    nametags,
    networks,
    policies,
    saving,
    s3_lib,
    tf_utils,
    train_lib,
    utils,
)
from slippi_ai import learner as learner_lib
from slippi_ai import data as data_lib
from slippi_ai.file_cache import FileCache
from slippi_ai import value_function as vf_lib

T = tp.TypeVar('T')

def _field(default_factory: tp.Callable[[], T]) -> T:
  return dataclasses.field(default_factory=default_factory)

@dataclasses.dataclass
class RuntimeConfig:
  max_runtime: int = 1 * 60 * 60  # maximum runtime in seconds
  log_interval: int = 10  # seconds between logging
  save_interval: int = 300  # seconds between saving to disk

  validate_every_n: int = 1000  # number of training steps between validations
  num_valid_steps: int = 100  # number of batches per validation

@dataclasses.dataclass
class EvaluatorConfig:
  run: bool = False
  eval_every_n: int = 1000
  rollout_length: int = 1 * 60 * 60  # one minute in-game time
  agent_name: str = 'Master Player'  # TODO: evaluate multiple names
  num_envs: int = 1
  async_envs: bool = True
  ray_envs: bool = False
  num_env_steps: int = 0
  inner_batch_size: int = 1
  async_inference: bool = False
  use_gpu: bool = True
  num_agent_steps: int = 0

  @property
  def env_kwargs(self) -> dict:
    kwargs = dict(num_steps=self.num_env_steps)
    if self.async_envs:
      kwargs.update(inner_batch_size=self.inner_batch_size)
    return kwargs

@dataclasses.dataclass
class DolphinConfig:
  """Configure dolphin for evaluation."""
  path: str = None  # Path to folder containing the dolphin executable
  iso: str = None  # Path to melee 1.02 iso.
  stage: melee.Stage = melee.Stage.RANDOM_STAGE  # Which stage to play on.
  online_delay: int = 0  # Simulate online delay.
  blocking_input: bool = True  # Have game wait for AIs to send inputs.
  slippi_port: int = 51441  # Local ip port to communicate with dolphin.
  render: bool = True  # Render frames. Only disable if using vladfi1\'s slippi fork.
  save_replays: bool = False  # Save slippi replays to the usual location.
  headless: bool = True  # Headless configuration: exi + ffw, no graphics or audio.

@dataclasses.dataclass
class FileCacheConfig:
  use: bool = False
  path: tp.Optional[str] = None
  wipe: bool = False
  version: str = 'test'

@dataclasses.dataclass
class ValueFunctionConfig:
  train_separate_network: bool = True
  separate_network_config: bool = True
  network: dict = _field(lambda: networks.DEFAULT_CONFIG)

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
  value_function: ValueFunctionConfig = _field(ValueFunctionConfig)

  max_names: int = 16

  dolphin: DolphinConfig = _field(DolphinConfig)
  evaluator: EvaluatorConfig = _field(EvaluatorConfig)

  expt_root: str = 'experiments'
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

  # TODO: configure controller embedding
  policy = saving.policy_from_config(dataclasses.asdict(config))
  embed_controller = policy.controller_embedding

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

  # Set up name mapping.
  max_names = config.max_names
  name_map = {}
  name_counts = collections.Counter()

  normalized_names = [
      nametags.normalize_name(replay.main_player.name)
      for replay in train_replays]
  name_counts.update(normalized_names)

  for name, _ in name_counts.most_common(max_names):
    name_map[name] = len(name_map)
  missing_name_code = len(name_map)
  num_codes = missing_name_code + 1

  # Bake in name groups from nametags.py
  for first, *rest in nametags.name_groups:
    if first not in name_map:
      continue
    for name in rest:
      name_map[name] = name_map[first]

  # Record name map
  print(name_map)
  name_map_path = os.path.join(expt_dir, 'name_map.json')
  with open(name_map_path, 'w') as f:
    json.dump(name_map, f)
  wandb.save(name_map_path, policy='now')

  def encode_name(name: str) -> np.uint8:
    return np.uint8(name_map.get(name, missing_name_code))
  batch_encode_name = np.vectorize(encode_name)

  # Create data sources for train and test.
  data_config = dict(
      dataclasses.asdict(config.data),
      embed_controller=embed_controller,
      extra_frames=1 + policy.delay,
      name_map=name_map,
      **char_filters,
  )
  train_data = data_lib.make_source(replays=train_replays, **data_config)
  test_data = data_lib.make_source(replays=test_replays, **data_config)

  train_manager = train_lib.TrainManager(learner, train_data, dict(train=True))
  test_manager = train_lib.TrainManager(learner, test_data, dict(train=False))

  # initialize variables
  train_stats, _ = train_manager.step()
  logging.info('loss initial: %f', _get_loss(train_stats))

  step = tf.Variable(0, trainable=False, name="step")

  # saving and restoring
  tf_state = dict(
      step=step,
      policy=policy.variables,
      value_function=value_function.variables if value_function else [],
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

  save_to_s3 = config.save_to_s3
  if save_to_s3 or config.restore_tag:
    if 'S3_CREDS' not in os.environ:
      raise ValueError('must set the S3_CREDS environment variable')

    s3_store = s3_lib.get_store()
    s3_keys = s3_lib.get_keys(tag)

  def get_combined_state():
    return dict(
        state=get_tf_state(),
        config=dataclasses.asdict(config),
        name_map=name_map,
    )

  def save():
    pickled_state = pickle.dumps(get_combined_state())

    logging.info('saving state to %s', pickle_path)
    with open(pickle_path, 'wb') as f:
      f.write(pickled_state)

    if save_to_s3:
      logging.info('saving state to S3: %s', s3_keys.combined)
      s3_store.put(s3_keys.combined, pickled_state)

  maybe_save = utils.Periodically(save, runtime.save_interval)

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
    train_loss = _get_loss(train_manager.step()[0])
    logging.info('loss post-restore: %f', train_loss)

  FRAMES_PER_MINUTE = 60 * 60

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

  def maybe_validate():
    total_steps = step.numpy()
    if total_steps % runtime.validate_every_n != 0:
      return

    valid_results = [test_manager.step() for _ in range(runtime.num_valid_steps)]

    valid_stats, batches = zip(*valid_results)
    valid_stats = tf.nest.map_structure(tf_utils.to_numpy, valid_stats)
    valid_stats = tf.nest.map_structure(utils.stack, *valid_stats)

    to_log = dict(validation=valid_stats)
    train_lib.log_stats(to_log, total_steps)

    # Log losses aggregated by name.

    # Stats have shape [num_valid_steps, unroll_length, batch_size]
    time_mean = lambda x: np.mean(x, axis=1)
    loss = time_mean(valid_stats['policy']['loss'])
    assert loss.shape == (runtime.num_valid_steps, config.data.batch_size)

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

    to_log = dict(validation_names=to_log)
    train_lib.log_stats(to_log, total_steps, take_mean=False)

  # Set up RL evaluation
  if config.evaluator.run:
    players = {
        1: dolphin.AI(),
        2: dolphin.CPU(),
    }
    dolphin_kwargs = dict(
        players=players,
        **dataclasses.asdict(config.dolphin),
    )
    agent_kwargs = dict(
        state=get_combined_state(),
        name=config.evaluator.agent_name,
        batch_steps=config.evaluator.num_agent_steps,
    )
    evaluator = evaluators.Evaluator(
        agent_kwargs={1: agent_kwargs},
        dolphin_kwargs=dolphin_kwargs,
        num_envs=config.evaluator.num_envs,
        async_envs=config.evaluator.async_envs,
        ray_envs=config.evaluator.ray_envs,
        env_kwargs=config.evaluator.env_kwargs,
        async_inference=config.evaluator.async_inference,
        use_gpu=config.evaluator.use_gpu,
    )
    del players, dolphin_kwargs, agent_kwargs
    evaluator_profiler = utils.Profiler()

    # TODO: implement parallel evaluation
    def maybe_evaluate():
      total_steps = step.numpy()
      if total_steps % config.evaluator.eval_every_n != 0:
        return

      policy_vars = [t.numpy() for t in policy.variables]

      with evaluator_profiler:
        metrics, _ = evaluator.rollout(
            num_steps=config.evaluator.rollout_length,
            policy_vars={1: policy_vars})

      total_reward = metrics[1].reward
      num_frames = config.evaluator.rollout_length * config.evaluator.num_envs
      mean_reward = total_reward / num_frames
      reward_per_minute = mean_reward * 60 * 60

      logging.info('Reward per minute: %f', reward_per_minute)
      logging.info('Evaluation timing: %.3f', evaluator_profiler.mean_time())

      to_log = dict(evaluation=dict(reward_per_minute=reward_per_minute))
      train_lib.log_stats(to_log, total_steps)

  start_time = time.time()

  with contextlib.ExitStack() as stack:
    if config.evaluator.run:
      stack.enter_context(evaluator.run())

    while time.time() - start_time < runtime.max_runtime:
      train_stats, _ = train_manager.step()
      step.assign_add(1)
      maybe_log(train_stats)
      maybe_validate()

      if config.evaluator.run:
        maybe_evaluate()

      save_path = maybe_save()
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
    dir=ff.String(None, 'directory to save logs'),
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
