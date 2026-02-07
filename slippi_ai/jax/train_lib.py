"""Train (and test) a network via imitation learning - JAX version."""

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

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import wandb

import melee

from slippi_ai import (
    flag_utils,
    nametags,
    utils,
)
from slippi_ai import data as data_lib
from slippi_ai import observations as obs_lib
from slippi_ai.policies import Platform
from slippi_ai.jax import networks, controller_heads, jax_utils
from slippi_ai.jax import learner as learner_lib
from slippi_ai.jax import embed as embed_lib
from slippi_ai.jax import policies as policies_lib
from slippi_ai.jax import value_function as vf_lib


def get_experiment_tag():
  today = datetime.date.today()
  return f'{today.year}-{today.month}-{today.day}_{secrets.token_hex(8)}'


class TrainManager:

  def __init__(
      self,
      learner: learner_lib.Learner,
      data_source: data_lib.AbstractDataSource,
      step_kwargs={},
      prefetch: int = 0,
      rngs: tp.Optional[nnx.Rngs] = None,
      data_sharding: tp.Optional[jax.sharding.NamedSharding] = None,
      # TODO: pass in epoch offset when resuming from checkpoint
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

    # Initialize hidden state (and shard if multi-device)
    hidden_state = learner.initial_state(data_source.batch_size, self.rngs)
    if data_sharding is not None:
      hidden_state = jax_utils.shard_pytree(hidden_state, data_sharding)
    self.hidden_state = hidden_state

    self._encode_state_action = learner.policy.network.encode

    self.prefetch = prefetch
    if prefetch > 0:
      self.frames_queue = queue.Queue(maxsize=prefetch)
      self.stop_requested = threading.Event()
      self.data_thread = threading.Thread(target=self.fetch_batches)
      self.data_thread.start()

  def fetch_batch(self) -> tuple[data_lib.Batch, float, data_lib.Frames]:
    batch, epoch = next(self.data_source)
    epoch += self.epoch_offset
    frames = batch.frames

    if np.any(frames.is_resetting[:, 1:]):
      raise ValueError("Unexpected mid-episode reset.")

    # Encode frames using the policy's network
    # TODO: when prefetching, calling network.encode can result strange
    # errors, such as:
    # 'SimpleEmbedNetwork' object has no attribute '_embed_state_action'
    # I believe this is due to a race condition with flax's in-place Module
    # updates, which can briefly cause members of the modules to disappear.
    # This should be mitigated by the use of nnx.cached_partial in the Learner.
    frames = frames._replace(
        state_action=self._encode_state_action(frames.state_action))

    # Convert to JAX arrays (and shard if multi-device)
    if self.data_sharding is not None:
      frames = jax_utils.shard_pytree(frames, self.data_sharding)
    else:
      frames = utils.map_nt(jnp.asarray, frames)

    return (batch, epoch, frames)

  def fetch_batches(self):
    # TODO: we might not need this anymore due to jax runahead
    while not self.stop_requested.is_set():
      data = self.fetch_batch()

      # Try to put data into the queue, but check for stop_requested
      while not self.stop_requested.is_set():
        try:
          self.frames_queue.put(data, timeout=1)
          break
        except queue.Full:
          continue

  def stop(self):
    if self.prefetch > 0:
      self.stop_requested.set()
      self.data_thread.join()

  def step(self) -> tuple[dict, data_lib.Batch]:
    stats = {}

    with self.data_profiler:
      if self.prefetch > 0:
        stats.update(frames_queue_size=self.frames_queue.qsize())
        batch, epoch, frames = self.frames_queue.get()
      else:
        batch, epoch, frames = self.fetch_batch()

    self.last_epoch = epoch

    with self.step_profiler:
      learner_stats, self.hidden_state = self.learner.step(
          frames, self.hidden_state, **self.step_kwargs)
      stats.update(learner_stats)

    return stats, batch


def mean(value):
  if isinstance(value, jax.Array):
    value = np.asarray(value)
  if isinstance(value, np.ndarray):
    value = value.mean().item()
  return value


def log_stats(
    stats: dict,
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

  num_evals_per_epoch: float = 1  # number evaluations per training epoch
  num_eval_epochs: float = 1  # number of epochs per evaluation

  compile: bool = True  # whether to JIT compile the training step
  multi_device: bool = True  # whether to use multi-device data parallelism
  prefetch: int = 0  # number of batches to prefetch in data loader

@dataclasses.dataclass
class ValueFunctionConfig:
  separate_network_config: bool = True
  network: dict = _field(networks.default_network_config)


@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = _field(RuntimeConfig)

  dataset: data_lib.DatasetConfig = _field(data_lib.DatasetConfig)
  data: data_lib.DataConfig = _field(data_lib.DataConfig)
  observation: obs_lib.ObservationConfig = _field(obs_lib.ObservationConfig)

  learner: learner_lib.LearnerConfig = _field(learner_lib.LearnerConfig)

  network: dict = _field(networks.default_network_config)
  controller_head: dict = _field(controller_heads.default_config)

  embed: embed_lib.EmbedConfig = _field(embed_lib.EmbedConfig)

  policy: policies_lib.PolicyConfig = _field(policies_lib.PolicyConfig)
  value_function: ValueFunctionConfig = _field(ValueFunctionConfig)

  max_names: int = 16  # Move to embed or dataset?

  expt_root: str = 'experiments/jax'
  expt_dir: tp.Optional[str] = None
  tag: tp.Optional[str] = None

  restore_path: tp.Optional[str] = None

  seed: int = 0
  version: int = 1
  platform: str = Platform.JAX.value


def _get_loss(stats: dict):
  loss = stats['policy']['loss']
  if isinstance(loss, jax.Array):
    loss = np.asarray(loss)
  if isinstance(loss, np.ndarray):
    return loss.mean().item()
  return loss


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


def policy_from_config(
    config: Config,
    rngs: nnx.Rngs,
) -> policies_lib.Policy:
  """Build a Policy from configuration."""
  embed_config = config.embed
  embed_controller = embed_config.controller.make_embedding()

  network = networks.build_embed_network(
      rngs=rngs,
      embed_config=embed_config,
      num_names=config.max_names,
      network_config=config.network,
  )

  controller_head = controller_heads.construct(
      rngs=rngs,
      input_size=network.output_size,
      embed_controller=embed_controller,
      **config.controller_head,
  )

  policy = policies_lib.Policy(
      network=network,
      controller_head=controller_head,
      delay=config.policy.delay,
  )

  return policy


def value_function_from_config(
    config: Config,
    rngs: nnx.Rngs,
) -> tp.Optional[vf_lib.ValueFunction]:
  vf_config = config.value_function
  network_config = config.network
  if vf_config.separate_network_config:
    network_config = vf_config.network

  return vf_lib.ValueFunction(
      rngs=rngs,
      network_config=network_config,
      num_names=config.max_names,
      embed_config=config.embed,
  )


def train(config: Config):
  with contextlib.ExitStack() as exit_stack:
    _train(config, exit_stack)


def _train(config: Config, exit_stack: contextlib.ExitStack):
  # Giving XLA all available memory avoids fragmentation issues
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

  tag = config.tag or get_experiment_tag()
  expt_dir = config.expt_dir
  if expt_dir is None:
    expt_dir = os.path.join(config.expt_root, tag)
    os.makedirs(expt_dir, exist_ok=True)
  config.expt_dir = expt_dir
  logging.info('experiment directory: %s', expt_dir)

  pickle_path = os.path.join(expt_dir, 'latest.pkl')

  # attempt to restore parameters
  restored = False
  combined_state: tp.Optional[dict] = None
  if config.restore_path:
    logging.info('restoring from %s', config.restore_path)
    with open(config.restore_path, 'rb') as f:
      combined_state = pickle.load(f)
    restored = True
  elif os.path.exists(pickle_path):
    logging.info('restoring from %s', pickle_path)
    with open(pickle_path, 'rb') as f:
      combined_state = pickle.load(f)
    restored = True
  else:
    logging.info('not restoring any params')

  # Training state. TODO: use orbax?
  step = 0
  train_time = 0.0
  best_eval_loss = float('inf')
  total_frames = 0

  if restored:
    assert isinstance(combined_state, dict)
    counters: dict = combined_state['counters']

    step = counters['step']
    best_eval_loss = counters['best_eval_loss']
    train_time = counters['train_time']
    total_frames: int = counters['total_frames']

    restore_config = flag_utils.dataclass_from_dict(
        Config, combined_state['config'])

    # We can update the delay as it doesn't affect the network architecture.
    if restore_config.policy.delay != config.policy.delay:
      logging.warning(
          f'Changing delay from {restore_config.policy.delay} to {config.policy.delay}.')
      best_eval_loss = float('inf')  # Old losses don't apply to new delay.

    # These we can't change after the fact.
    for key in ['network', 'controller_head', 'embed']:
      current = getattr(config, key)
      previous = getattr(restore_config, key)
      if current != previous:
        if config.restore_path is None:
          # In this case we are implicitly restoring from the same experiment,
          # and it would be surprising to use the old config.
          # TODO: improve this check, maybe ask the user for confirmation?
          raise ValueError(
              f'Requested {key} config doesn\'t match existing config.')

        logging.warning(
            f'Requested {key} config doesn\'t match, overriding from checkpoint.')
        setattr(config, key, previous)

  logging.info("Network configuration")
  for comp in ['network', 'controller_head']:
    logging.info(f'Using {comp}: {getattr(config, comp)["name"]}')

  # Multi-device setup
  runtime = config.runtime
  mesh: tp.Optional[jax.sharding.Mesh] = None
  data_sharding = None
  num_devices = jax_utils.num_devices()
  if runtime.multi_device:
    if num_devices == 1:
      logging.warning(
          'Multi-device training requested but only 1 device available.')
    else:
      logging.info('Multi-device training enabled with %d devices', num_devices)
    if config.data.batch_size % num_devices != 0:
      raise ValueError(
          f'Batch size {config.data.batch_size} must be divisible by '
          f'num_devices {num_devices}')
    mesh = jax_utils.get_mesh()
    data_sharding = jax_utils.data_sharding(mesh)
  else:
    if num_devices > 1:
      logging.warning(
          'Multiple devices detected but multi-device training not enabled.')

    if config.learner.use_shard_map:
      logging.warning(
          'Learner.use_shard_map enabled without multi-device training; disabling.')
      config.learner.use_shard_map = False

    logging.info('Single-device training')


  # Initialize RNG
  rngs = nnx.Rngs(config.seed)

  # Build policy and value function
  policy = policy_from_config(config, rngs)
  value_function = value_function_from_config(config, rngs)

  learner_kwargs = dataclasses.asdict(config.learner)
  learner = learner_lib.Learner(
      policy=policy,
      value_function=value_function,
      mesh=mesh,
      **learner_kwargs,
  )

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
    assert isinstance(combined_state, dict)  # appease type checker
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
      extra_frames=1 + policy._delay,
      name_map=name_map,
      observation_config=config.observation,
      **char_filters,
  )
  train_data = data_lib.make_source(replays=train_replays, **data_config)

  test_data_config = dict(
      data_config,
      # Use more workers for test data to keep up with eval speed.
      num_workers=2 * config.data.num_workers,
      # batch_size=2 * config.data.batch_size,
  )
  test_data = data_lib.make_source(replays=test_replays, **test_data_config)
  del train_replays, test_replays  # free up memory

  train_manager = TrainManager(
      learner, train_data, dict(train=True, compile=runtime.compile),
      rngs=rngs, data_sharding=data_sharding, prefetch=runtime.prefetch)
  test_manager = TrainManager(
      learner, test_data, dict(train=False, compile=runtime.compile),
      rngs=rngs, data_sharding=data_sharding, prefetch=runtime.prefetch)

  # TrainManager should probably be a proper context manager.
  exit_stack.callback(train_manager.stop)
  exit_stack.callback(test_manager.stop)

  train_stats, _ = train_manager.step()
  logging.info('loss initial: %f', _get_loss(train_stats))

  gpu_memory = jax_utils.get_process_gpu_memory_gb()
  if gpu_memory is not None:
    logging.info('initial GPU memory usage: %.2f GB', gpu_memory)
  else:
    logging.info('GPU memory usage not available (pynvml not installed)')

  # TODO: use orbax instead?
  def save(eval_loss=None):
    nonlocal best_eval_loss
    # Local Save
    state = jax_utils.get_module_state(learner)

    counters = dict(
        step=step,
        total_frames=total_frames,
        train_time=train_time,
        best_eval_loss=eval_loss if eval_loss is not None else best_eval_loss,
    )

    # easier to always bundle the config with the state
    combined = dict(
        state=state,
        step=step,
        config=dataclasses.asdict(config),
        name_map=name_map,
        dataset_metrics=dataset_metrics,
        counters=counters,
    )
    pickled_state = pickle.dumps(combined)

    logging.info('saving state to %s', pickle_path)
    with open(pickle_path, 'wb') as f:
      f.write(pickled_state)

  if restored:
    assert isinstance(combined_state, dict)  # appease type checker
    jax_utils.set_module_state(learner, combined_state['state'])
    train_loss = _get_loss(train_manager.step()[0])
    logging.info('loss post-restore: %f', train_loss)

  FRAMES_PER_MINUTE = 60 * 60
  FRAMES_PER_STEP = config.data.batch_size * config.data.unroll_length

  step_tracker = utils.Tracker(step)
  epoch_tracker = utils.Tracker(train_manager.last_epoch)
  log_tracker = utils.Tracker(time.time())

  @utils.periodically(runtime.log_interval)
  def maybe_log(train_stats: dict):
    """Do a test step, then log both train and test stats."""
    test_stats, _ = test_manager.step()

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
        num_frames=num_frames,
        total_frames=total_frames,
        train_time=train_time,
    )
    log_stats(all_stats, total_steps)

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

  def maybe_eval():
    nonlocal best_eval_loss, last_train_epoch_evaluated

    train_epoch = train_manager.last_epoch
    if (train_epoch - last_train_epoch_evaluated) * runtime.num_evals_per_epoch < 1:
      return
    last_train_epoch_evaluated = train_epoch

    per_step_eval_stats: list[dict] = []
    metas: list[data_lib.ChunkMeta] = []

    def time_mean(x: jax.Array) -> np.ndarray:
      return np.mean(x, axis=1)

    start_time = time.perf_counter()
    initial_test_epoch = test_manager.last_epoch
    test_stats_jax = None
    while test_manager.last_epoch - initial_test_epoch < runtime.num_eval_epochs:
      # Get _previous_ step's stats to allow jax runahead
      if test_stats_jax is not None:
        test_stats_np = utils.map_single_structure(time_mean, test_stats_jax)
        per_step_eval_stats.append(test_stats_np)

      test_stats_jax, batch = test_manager.step()

      metas.append(batch.meta)

    assert test_stats_jax is not None
    test_stats_np = utils.map_single_structure(time_mean, test_stats_jax)
    per_step_eval_stats.append(test_stats_np)

    eval_time = time.perf_counter() - start_time

    # [eval_steps, batch_size], mean taken over time
    eval_stats = utils.batch_nest_nt(per_step_eval_stats)

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

    to_log = dict(
        counters,
        eval=utils.map_nt(mean, eval_stats),
        eval_timings=timings,
    )

    # Calculate the mean eval loss
    eval_loss = eval_stats['policy']['loss'].mean()

    print(f'EVAL step={step} epoch={train_epoch:.3f} loss={eval_loss:.4f}')
    print(f'sps={sps:.2f} mps={mps:.2f}'
          f' data={data_time:.3f} step={step_time:.3f}'
          f' total={eval_time:.1f}'
          f' num_batches={len(per_step_eval_stats)}')
    print()

    # Save if the eval loss is the best so far
    if eval_loss < best_eval_loss:
      logging.info('New best eval loss: %f (previous: %f)', eval_loss, best_eval_loss)
      best_eval_loss = eval_loss
      save(eval_loss=best_eval_loss)

    # Stats have shape [num_eval_steps, batch_size]
    loss = eval_stats['policy']['loss']
    assert loss.shape == (len(per_step_eval_stats), test_data.batch_size)

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

      losses, counts = zip(*loss_sums_and_counts)
      to_log['eval_characters'] = dict(
          losses=per_character_loss_sums,
          counts=per_character_loss_counts,
      )

    log_stats(to_log, step, take_mean=False)

  start_time = time.time()

  train_profiler = utils.Profiler(burnin=0)

  while time.time() - start_time < runtime.max_runtime:
    with train_profiler:
      train_stats, _ = train_manager.step()

    # Update counters
    step += 1
    total_frames += FRAMES_PER_STEP
    train_time += train_profiler.last_time

    maybe_log(train_stats)
    maybe_eval()
