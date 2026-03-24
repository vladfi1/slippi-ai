"""Train a value function standalone - JAX version."""

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
    vf_learner as learner_lib,
    value_function as vf_lib,
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
  eval_at_start: bool = False


@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = _field(RuntimeConfig)

  dataset: data_lib.DatasetConfig = _field(data_lib.DatasetConfig)
  data: data_lib.DataConfig = _field(data_lib.DataConfig)
  observation: obs_lib.ObservationConfig = _field(obs_lib.ObservationConfig)

  # Loads obs config and name map to be compatible with a given policy.
  compatible_policy: tp.Optional[str] = None

  max_names: int = 16

  learner: learner_lib.VFLearnerConfig = _field(learner_lib.VFLearnerConfig)

  embed: embed_lib.EmbedConfig = _field(embed_lib.EmbedConfig)
  network: dict = _field(networks.default_network_config)

  expt_root: str = 'experiments/value_function'
  expt_dir: tp.Optional[str] = None
  tag: tp.Optional[str] = None

  restore_path: tp.Optional[str] = None

  seed: int = 0
  version: int = 0
  platform: str = Platform.JAX.value


class TrainManager:

  def __init__(
      self,
      learner: learner_lib.VFLearner,
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
      hidden_state = jax_utils.device_put(hidden_state, data_sharding)
    self.hidden_state = hidden_state

  def step(self) -> tuple[dict, data_lib.BatchWithMeta]:
    stats = {}

    with self.data_profiler:
      batch_with_meta, epoch = next(self.data_source)

    self.last_epoch = epoch + self.epoch_offset

    with self.step_profiler:
      learner_stats, self.hidden_state = self.learner.step(
          batch_with_meta.batch, self.hidden_state, **self.step_kwargs)
      stats.update(learner_stats)

    return stats, batch_with_meta


def print_losses(name: str, stats: dict):
  loss = train_lib.mean(stats['value']['loss'])
  uev = train_lib.mean(stats['value']['uev'])
  print(f'{name}: loss={loss:.4f} uev={uev:.4f}')


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
  train_epoch = 0.0

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
    train_epoch = counters.get('train_epoch', 0.0)

    restore_config = flag_utils.dataclass_from_dict(
        Config, restored_state['config'])

    for key in ['network', 'embed', 'observation', 'max_names']:
      current = getattr(config, key)
      previous = getattr(restore_config, key)
      if current != previous:
        logging.warning(
            f'Requested {key} config doesn\'t match, overriding from checkpoint.')
        setattr(config, key, previous)

    name_map = restored_state['name_map']
  elif config.compatible_policy is not None:
    logging.info('loading configs from %s', config.compatible_policy)
    imitation_state = saving.load_state_from_disk(config.compatible_policy)
    imitation_config = flag_utils.dataclass_from_dict(
        train_lib.Config, imitation_state['config'])

    config.observation = imitation_config.observation
    config.max_names = imitation_config.max_names
    name_map = imitation_state['name_map']
  else:
    logging.warning('No compatible policy or checkpoint specified.')

  # Set wandb config after potential overrides from checkpoint or compatible policy.
  wandb.config.update(dataclasses.asdict(config))

  rngs = nnx.Rngs(config.seed)

  value_function = vf_lib.ValueFunction(
      rngs=rngs,
      network_config=config.network,
      num_names=config.max_names,
      embed_config=config.embed,
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

  learner = learner_lib.VFLearner(
      value_function=value_function,
      config=config.learner,
      mesh=mesh,
      data_sharding=data_sharding,
  )

  logging.info("Network configuration: %s", config.network['name'])

  ### Dataset Creation ###
  train_data_config = config.data
  test_data_config = dataclasses.replace(
      train_data_config,
      num_workers=2 * train_data_config.num_workers,
  )

  train_data, test_data, name_map = data_lib.build_sources(
      dataset_config=config.dataset,
      train_data_config=train_data_config,
      test_data_config=test_data_config,
      name_map=name_map,
      max_names=config.max_names,
      extra_frames=1,
      observation_config=config.observation,
  )
  exit_stack.callback(train_data.shutdown)
  exit_stack.callback(test_data.shutdown)

  train_manager = TrainManager(
      learner, train_data, dict(train=True),
      rngs=rngs, data_sharding=data_sharding, epoch_offset=train_epoch)
  test_manager = TrainManager(
      learner, test_data, dict(train=False),
      rngs=rngs, data_sharding=data_sharding)

  print_losses('initial', test_manager.step()[0])

  if restored:
    assert isinstance(restored_state, dict)
    replicated_state = jax_utils.device_put(
        restored_state['state'], jax_utils.replicate_sharding(mesh))
    jax_utils.set_module_state(learner, replicated_state)
    print_losses('post-restore', test_manager.step()[0])
    del restored_state, replicated_state

  def save(eval_loss=None):
    nonlocal best_eval_loss
    jax_state = jax_utils.get_module_state(learner)

    counters = dict(
        step=step,
        total_frames=total_frames,
        train_time=train_time,
        best_eval_loss=eval_loss if eval_loss is not None else best_eval_loss,
        train_epoch=train_manager.last_epoch,
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

    def time_mean(x: jax.Array, axis: int = 1) -> np.ndarray:
      assert x.shape[axis] == config.data.unroll_length
      return np.mean(np.asarray(x), axis=axis)

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
    frames_per_step = test_data.batch_size * test_data_config.unroll_length
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

    eval_loss = mean_stats['value']['loss']

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

  maybe_eval(force=config.runtime.eval_at_start)

  while time.time() - start_time < runtime.max_runtime:
    with train_profiler:
      train_stats, _ = train_manager.step()

    step += 1
    total_frames += FRAMES_PER_STEP
    train_time += train_profiler.last_time

    maybe_log(train_stats)
    maybe_eval()

  maybe_eval(force=True)
