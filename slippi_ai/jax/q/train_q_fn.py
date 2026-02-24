
"""Train a q-function (without sample policy or q-policy)."""

import contextlib
import dataclasses
import json
import os
import pickle
import time
import typing as tp

from absl import logging

import numpy as np
import jax
from flax import nnx
import wandb

from slippi_ai import (
    flag_utils,
    observations as obs_lib,
    utils,
    data as data_lib,
)
from slippi_ai.policies import Platform
from slippi_ai.jax import (
    embed as embed_lib,
    saving,
    train_lib,
    jax_utils,
    networks,
    train_lib,
)
from slippi_ai.jax.q import (
    q_fn_learner as learner_lib,
    q_function as q_lib,
)

_field = utils.field

@dataclasses.dataclass
class RuntimeConfig:
  max_runtime: int = 1 * 60 * 60  # maximum runtime in seconds
  log_interval: int = 10  # seconds between logging
  save_interval: int = 300  # seconds between saving to disk

  num_evals_per_epoch: float = 1  # number evaluations per training epoch
  num_eval_epochs: float = 1  # number of epochs per evaluation
  max_eval_steps: tp.Optional[int] = None  # useful for tests

@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = _field(RuntimeConfig)

  dataset: data_lib.DatasetConfig = _field(data_lib.DatasetConfig)
  data: data_lib.DataConfig = _field(data_lib.DataConfig)
  observation: obs_lib.ObservationConfig = _field(obs_lib.ObservationConfig)

  # Loads obs config and name map to be compatible with a given policy.
  # This is needed later in the combined setting so that the data works with
  # both the q-function and the policy.
  compatible_policy: tp.Optional[str] = None

  max_names: int = 16

  learner: learner_lib.LearnerConfig = _field(learner_lib.LearnerConfig)
  delay: tp.Optional[int] = None  # if None, use the policy's delay

  # Used only to define the action embedding and delay.
  embed: embed_lib.EmbedConfig = _field(embed_lib.EmbedConfig)
  network: dict = _field(networks.default_config)

  expt_root: str = 'experiments/q_only'
  expt_dir: tp.Optional[str] = None
  tag: tp.Optional[str] = None

  restore_path: tp.Optional[str] = None

  seed: int = 0
  version: int = 0
  platform: str = Platform.JAX.value


class TrainManager:

  def __init__(
      self,
      learner: learner_lib.Learner,
      data_source: data_lib.AbstractDataSource,
      step_kwargs={},
      rngs: tp.Optional[nnx.Rngs] = None,
      data_sharding: tp.Optional[jax.sharding.NamedSharding] = None,
      epoch_offset: float = 0,
  ):
    self.learner = learner
    self.data_source = data_source
    self.rngs = rngs or nnx.Rngs(0)
    self.step_kwargs = step_kwargs
    self.data_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()
    self.data_sharding = data_sharding
    self.epoch_offset = epoch_offset
    self.last_epoch = 0.

    hidden_state = learner.initial_state(data_source.batch_size, self.rngs)
    if data_sharding is not None:
      hidden_state = jax_utils.shard_pytree(hidden_state, data_sharding)
    self.hidden_state = hidden_state

  def step(self) -> tuple[dict, data_lib.Batch]:
    stats = {}

    with self.data_profiler:
      batch, epoch = next(self.data_source)

    self.last_epoch = epoch

    with self.step_profiler:
      learner_stats, self.hidden_state = self.learner.step(
          batch, self.hidden_state, **self.step_kwargs)
      stats.update(learner_stats)

    return stats, batch

def print_losses(name: str, stats: dict):
  v_uev = stats[learner_lib.Q_FUNCTION]['v']['uev']
  q_uev = stats[learner_lib.Q_FUNCTION]['q']['uev']
  v_loss = stats[learner_lib.Q_FUNCTION]['v']['loss']
  q_loss = stats[learner_lib.Q_FUNCTION]['q']['loss']

  v_uev, q_uev, v_loss, q_loss = map(train_lib.mean, (v_uev, q_uev, v_loss, q_loss))

  print(f'{name}: v_uev={v_uev:.4f} q_uev={q_uev:.4f} v_loss={v_loss:.4f} q_loss={q_loss:.4f}')

def train(config: Config):
  with contextlib.ExitStack() as exit_stack:
    _train(config, exit_stack)

def _train(config: Config, exit_stack: contextlib.ExitStack):
  tag = config.tag or train_lib.get_experiment_tag()
  expt_dir = config.expt_dir
  if expt_dir is None:
    expt_dir = os.path.join(config.expt_root, tag)
    os.makedirs(expt_dir, exist_ok=True)
  config.expt_dir = expt_dir
  logging.info('experiment directory: %s', expt_dir)

  runtime = config.runtime

  step = 0
  train_time = 0.0
  best_eval_loss = float('inf')
  total_frames = 0

  name_map: tp.Optional[dict[str, int]] = None

  pickle_path = os.path.join(expt_dir, 'latest.pkl')

  restored = False
  if config.restore_path:
    logging.info('restoring from %s', config.restore_path)
    restored_state = saving.load_state_from_disk(config.restore_path)
    restored = True
  elif os.path.exists(pickle_path):
    logging.info('restoring from %s', pickle_path)
    restored_state = saving.load_state_from_disk(pickle_path)
    restored = True
  else:
    logging.info('not restoring any params')

  if restored:
    assert isinstance(restored_state, dict)
    counters: dict = restored_state['counters']

    step = counters['step']
    best_eval_loss = counters['best_eval_loss']
    train_time = counters['train_time']
    total_frames: int = counters['total_frames']

    restore_config = flag_utils.dataclass_from_dict(
        Config, restored_state['config'])

    for key in ['network', 'embed', 'observation', 'delay', 'max_names']:
      current = getattr(config, key)
      previous = getattr(restore_config, key)
      if current != previous:
        logging.warning(
            f'Requested {key} config doesn\'t match, overriding from checkpoint.')
        setattr(config, key, previous)

    assert config.delay is not None
    name_map = restored_state['name_map']
  elif config.compatible_policy is not None:
    logging.info('loading configs from %s', config.compatible_policy)
    imitation_state = saving.load_state_from_disk(config.compatible_policy)
    imitation_config = flag_utils.dataclass_from_dict(
        train_lib.Config, imitation_state['config'])

    config.observation = imitation_config.observation
    config.max_names = imitation_config.max_names
    name_map = imitation_state['name_map']
    if config.delay is None:
      logging.info('setting delay from compatible policy: %d', imitation_config.policy.delay)
      config.delay = imitation_config.policy.delay
  else:
    logging.warning('No compatible policy or checkpoint specified.')
    if config.delay is None:
      raise ValueError('Must specify delay.')

  rngs = nnx.Rngs(config.seed)

  q_function = q_lib.QFunction(
      rngs=rngs,
      network_config=config.network,
      embed_action=config.embed.controller.make_embedding(),
      embed_config=config.embed,
      num_names=config.max_names,
  )

  mesh = jax_utils.get_mesh()
  data_sharding = jax_utils.data_sharding(mesh)

  num_devices = jax_utils.num_devices()
  if num_devices == 1:
    logging.warning(
        'Multi-device training requested but only 1 device available.')
  else:
    logging.info('Multi-device training enabled with %d devices', num_devices)
  if config.data.batch_size % num_devices != 0:
    raise ValueError(
        f'Batch size {config.data.batch_size} must be divisible by '
        f'num_devices {num_devices}')

  learner = learner_lib.Learner(
      config=config.learner,
      q_function=q_function,
      delay=config.delay,
      mesh=mesh,
      data_sharding=data_sharding,
  )

  logging.info("Network configuration")
  for comp in ['network']:
    logging.info(f'Using {comp}: {getattr(config, comp)["name"]}')

  ### Dataset Creation ###
  dataset_config = config.dataset

  train_replays, test_replays = data_lib.train_test_split(dataset_config)
  logging.info(f'Training on {len(train_replays)} replays, testing on {len(test_replays)}')

  if name_map is None:
    name_map = train_lib.create_name_map(train_replays, config.max_names)

  name_map_path = os.path.join(expt_dir, 'name_map.json')
  print(name_map)
  with open(name_map_path, 'w') as f:
    json.dump(name_map, f)
  wandb.save(name_map_path, policy='now')

  data_config = dict(
      dataclasses.asdict(config.data),
      extra_frames=1 + config.delay,
      name_map=name_map,
      observation_config=config.observation,
  )
  train_data = data_lib.make_source(replays=train_replays, **data_config)
  test_data = data_lib.make_source(replays=test_replays, **data_config)
  del train_replays, test_replays

  train_manager = TrainManager(
      learner, train_data, dict(train=True),
      rngs=rngs, data_sharding=data_sharding)
  test_manager = TrainManager(
      learner, test_data, dict(train=False),
      rngs=rngs, data_sharding=data_sharding)

  print_losses('initial', train_manager.step()[0])

  if restored:
    assert isinstance(restored_state, dict)
    jax_utils.set_module_state(
        learner,
        jax_utils.shard_pytree(restored_state['state'], data_sharding))
    print_losses('post-restore', train_manager.step()[0])
    del restored_state

  def save(eval_loss=None):
    nonlocal best_eval_loss
    jax_state = jax_utils.get_module_state(learner)

    counters = dict(
        step=step,
        total_frames=total_frames,
        train_time=train_time,
        best_eval_loss=eval_loss if eval_loss is not None else best_eval_loss,
    )

    combined = dict(
        state=jax_state,
        step=step,
        config=dataclasses.asdict(config),
        name_map=name_map,
        counters=counters,
    )
    pickled_state = pickle.dumps(combined)

    logging.info('saving state to %s', pickle_path)
    with open(pickle_path, 'wb') as f:
      f.write(pickled_state)

  FRAMES_PER_MINUTE = 60 * 60
  FRAMES_PER_STEP = config.data.batch_size * config.data.unroll_length

  step_tracker = utils.Tracker(step)
  epoch_tracker = utils.Tracker(train_manager.last_epoch)
  log_tracker = utils.Tracker(time.time())

  @utils.periodically(runtime.log_interval)
  def maybe_log(train_stats: dict):
    test_stats, _ = test_manager.step()

    train_stats, test_stats = utils.map_single_structure(
        train_lib.mean, (train_stats, test_stats))

    elapsed_time = log_tracker.update(time.time())
    total_steps = step
    steps = step_tracker.update(total_steps)
    num_frames = steps * FRAMES_PER_STEP

    epoch = train_manager.last_epoch
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

    print(f'step={total_steps} epoch={epoch:.3f}')
    print(f'sps={sps:.2f} mps={mps:.2f} eph={eph:.2e}')
    print_losses('train', train_stats)
    print_losses('test', test_stats)
    print(f'timing:'
          f' data={data_time:.3f}'
          f' step={step_time:.3f}')
    print()

  last_train_epoch_evaluated = 0.

  def maybe_eval(force: bool = False):
    nonlocal best_eval_loss, last_train_epoch_evaluated

    train_epoch = train_manager.last_epoch
    if not force and (train_epoch - last_train_epoch_evaluated) * runtime.num_evals_per_epoch < 1:
      return
    last_train_epoch_evaluated = train_epoch

    per_step_eval_stats: list[dict] = []

    def time_mean(x: jax.Array) -> np.ndarray:
      return np.mean(x, axis=1)

    start_time = time.perf_counter()
    initial_test_epoch = test_manager.last_epoch
    test_stats_jax = None
    num_eval_steps = 0
    while test_manager.last_epoch - initial_test_epoch < runtime.num_eval_epochs:
      if test_stats_jax is not None:
        test_stats_np = utils.map_single_structure(time_mean, test_stats_jax)
        per_step_eval_stats.append(test_stats_np)
      test_stats_jax, _ = test_manager.step()

      num_eval_steps += 1
      if (config.runtime.max_eval_steps is not None and
          num_eval_steps >= config.runtime.max_eval_steps):
        break

    assert test_stats_jax is not None
    test_stats_np = utils.map_single_structure(time_mean, test_stats_jax)
    per_step_eval_stats.append(test_stats_np)

    eval_stats = utils.batch_nest_nt(per_step_eval_stats)
    eval_time = time.perf_counter() - start_time

    data_time = test_manager.data_profiler.mean_time()
    step_time = test_manager.step_profiler.mean_time()

    sps = len(per_step_eval_stats) / eval_time
    frames_per_step = test_data.batch_size * config.data.unroll_length
    mps = sps * frames_per_step / FRAMES_PER_MINUTE

    train_epoch = epoch_tracker.last
    counters = dict(
        total_frames=total_frames,
        train_epoch=train_epoch,
        train_time=train_time,
    )

    timings = dict(
        sps=sps,
        mps=mps,
        data=data_time,
        step=step_time,
        total=eval_time,
        num_batches=len(per_step_eval_stats),
    )

    mean_stats = utils.map_single_structure(train_lib.mean, eval_stats)

    to_log = dict(
        counters,
        eval=mean_stats,
        eval_timings=timings,
    )

    train_lib.log_stats(to_log, step, take_mean=False)

    eval_loss = mean_stats[learner_lib.Q_FUNCTION]['q']['loss']

    if eval_loss < best_eval_loss:
      logging.info('New best eval loss: %f (previous: %f)', eval_loss, best_eval_loss)
      best_eval_loss = eval_loss
      save(eval_loss=best_eval_loss)

    print(f'EVAL step={step} epoch={train_epoch:.3f} loss={eval_loss:.4f}')
    print_losses('eval', mean_stats)
    print(f'sps={sps:.2f} mps={mps:.2f}'
          f' data={data_time:.3f} step={step_time:.3f}'
          f' total={eval_time:.1f}'
          f' num_batches={len(per_step_eval_stats)}')
    print()

  start_time = time.time()
  train_profiler = utils.Profiler(burnin=0)

  while time.time() - start_time < runtime.max_runtime:
    with train_profiler:
      train_stats, _ = train_manager.step()

    step += 1
    total_frames += FRAMES_PER_STEP
    train_time += train_profiler.last_time

    maybe_log(train_stats)
    maybe_eval()

  maybe_eval(force=True)
