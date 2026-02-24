
"""Train (and test) a network via imitation learning."""

import contextlib
import dataclasses
import functools
import json
import os
import pickle
import time
import typing as tp
import queue, threading

from absl import logging

import numpy as np
import jax
from flax import nnx
import wandb

from slippi_ai import (
    evaluators,
    flag_utils,
    nametags,
    utils,
    data as data_lib,
    dolphin as dolphin_lib,
    observations as obs_lib,
)
from slippi_ai.policies import Platform
from slippi_ai.jax import (
    embed as embed_lib,
    policies,
    q_learner as learner_lib,
    saving,
    train_lib,
    jax_utils,
    q_function as q_lib,
    networks,
    controller_heads,
)

_field = utils.field

@dataclasses.dataclass
class RuntimeConfig:
  max_runtime: int = 1 * 60 * 60  # maximum runtime in seconds
  log_interval: int = 10  # seconds between logging
  save_interval: int = 300  # seconds between saving to disk

  num_evals_per_epoch: float = 1  # number evaluations per training epoch
  num_eval_epochs: float = 1  # number of epochs per evaluation

@dataclasses.dataclass
class AgentConfig:
  batch_steps: int = 0
  compile: bool = True
  jit_compile: bool = True
  name: str = nametags.DEFAULT_NAME
  async_inference: bool = False

@dataclasses.dataclass
class RLEvaluatorConfig:
  use: bool = False
  # Seconds between evaluations. Note that the evaluator runs at around
  # half real-time, so this should be ~20x the rollout length if you want
  # to spend 10% of the time evaluating.
  # TODO: try running in parallel with training (so evaluator must be on CPU)
  interval_seconds: float = 15 * 60
  runtime_seconds: float = 60

  dolphin: dolphin_lib.DolphinConfig = _field(dolphin_lib.DolphinConfig)

  # env
  rollout_length: int = 600  # rollout chunk size
  num_envs: int = 1
  async_envs: bool = True
  num_env_steps: int = 0
  inner_batch_size: int = 1
  use_fake_envs: bool = False
  reset_every_n_evals: int = 1

  agent: AgentConfig = _field(AgentConfig)
  opponent: tp.Optional[str] = None
  opponent_name: str = nametags.DEFAULT_NAME
  gpu_inference: bool = True

@dataclasses.dataclass
class QFunctionConfig:
  network: dict = _field(networks.default_config)

@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = _field(RuntimeConfig)
  rl_evaluator: RLEvaluatorConfig = _field(RLEvaluatorConfig)

  dataset: data_lib.DatasetConfig = _field(data_lib.DatasetConfig)
  data: data_lib.DataConfig = _field(data_lib.DataConfig)
  observation: obs_lib.ObservationConfig = _field(obs_lib.ObservationConfig)

  max_names: int = 16

  learner: learner_lib.LearnerConfig = _field(learner_lib.LearnerConfig)

  # These apply to both sample and q policies.
  # TODO: can we support distinct configurations here?
  policy: policies.PolicyConfig = _field(policies.PolicyConfig)
  network: dict = _field(networks.default_config)
  controller_head: dict = _field(controller_heads.default_config)

  embed: embed_lib.EmbedConfig = _field(embed_lib.EmbedConfig)

  q_function: QFunctionConfig = _field(QFunctionConfig)

  expt_root: str = 'experiments/q_learning'
  expt_dir: tp.Optional[str] = None
  tag: tp.Optional[str] = None

  # TODO: group these into their own subconfig
  restore_path: tp.Optional[str] = None
  initialize_policies_from: tp.Optional[str] = None

  seed: int = 0
  version: int = saving.VERSION
  platform: str = Platform.JAX.value


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
      cached: bool = False,
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
    self.cached = cached

    # Initialize hidden state (and shard if multi-device)
    hidden_state = learner.initial_state(data_source.batch_size, self.rngs)
    if data_sharding is not None:
      hidden_state = jax_utils.shard_pytree(hidden_state, data_sharding)
    self.hidden_state = hidden_state

    self.prefetch = prefetch
    if prefetch > 0:
      self.frames_queue = queue.Queue(maxsize=prefetch)
      self.stop_requested = threading.Event()
      self.data_thread = threading.Thread(target=self.fetch_batches)
      self.data_thread.start()

    self.cached_fetch_batch = functools.cache(self.fetch_batch)

  def fetch_batch(self) -> tuple[data_lib.Batch, float, data_lib.Frames]:
    batch, epoch = next(self.data_source)
    epoch += self.epoch_offset

    if np.any(batch.is_resetting[:, 1:]):
      raise ValueError("Unexpected mid-episode reset.")

    frames = self.learner.prepare_frames(batch)
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
      # if self.prefetch > 0:
      #   stats.update(frames_queue_size=self.frames_queue.qsize())
      #   batch, epoch, frames = self.frames_queue.get()
      # else:
      #   batch_fn = self.cached_fetch_batch if self.cached else self.fetch_batch
      #   batch, epoch, frames = batch_fn()
      batch, epoch = next(self.data_source)

    self.last_epoch = epoch

    with self.step_profiler:
      learner_stats, self.hidden_state = self.learner.step(
          batch, self.hidden_state, **self.step_kwargs)
      stats.update(learner_stats)

    return stats, batch

def print_losses(name: str, stats: dict):
  spl = stats[learner_lib.SAMPLE_POLICY]['loss']
  v_uev = stats[learner_lib.Q_FUNCTION]['v']['uev']
  q_uev = stats[learner_lib.Q_FUNCTION]['q']['uev']
  v_loss = stats[learner_lib.Q_FUNCTION]['v']['loss']
  q_loss = stats[learner_lib.Q_FUNCTION]['q']['loss']
  qpl = stats[learner_lib.Q_POLICY]['q_loss']

  spl, v_uev, q_uev, v_loss, q_loss, qpl = map(train_lib.mean, (spl, v_uev, q_uev, v_loss, q_loss, qpl))

  print(f'{name}: spl={spl:.4f} v_uev={v_uev:.4f} q_uev={q_uev:.4f} v_loss={v_loss:.4f} q_loss={q_loss:.4f} qpl={qpl:.4f}')

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

  runtime = config.runtime

  # Training state. TODO: use orbax?
  step = 0
  train_time = 0.0
  best_eval_loss = float('inf')
  total_frames = 0

  name_map: tp.Optional[dict[str, int]] = None  # initialized later, after train/test split

  pickle_path = os.path.join(expt_dir, 'latest.pkl')

  # attempt to restore parameters
  restored = False
  if config.restore_path:
    logging.info('restoring from %s', config.restore_path)
    combined_state = saving.load_state_from_disk(config.restore_path)
    restored = True
  elif os.path.exists(pickle_path):
    logging.info('restoring from %s', pickle_path)
    combined_state = saving.load_state_from_disk(pickle_path)
    restored = True
  else:
    logging.info('not restoring any params')

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
    # if restore_config.policy.delay != config.policy.delay:
    #   logging.warning(
    #       f'Changing delay from {restore_config.policy.delay} to {config.policy.delay}.')
    #   best_eval_loss = float('inf')  # Old losses don't apply to new delay.

    # These we can't change after the fact.
    for key in ['policy', 'network', 'controller_head', 'embed', 'observation']:
      current = getattr(config, key)
      previous = getattr(restore_config, key)
      if current != previous:
        logging.warning(
            f'Requested {key} config doesn\'t match, overriding from checkpoint.')
        setattr(config, key, previous)

  if not restored and config.initialize_policies_from:
    logging.info(f'Initializing policies from {config.initialize_policies_from}')
    imitation_state = saving.load_state_from_disk(config.initialize_policies_from)

    sample_policy = saving.load_policy_from_state(imitation_state)
    q_policy = saving.load_policy_from_state(imitation_state)

    # Overwrite policy config
    imitation_config = flag_utils.dataclass_from_dict(
        train_lib.Config,
        saving.upgrade_config(imitation_state['config'])
    )
    for key in ['policy', 'network', 'controller_head', 'embed', 'observation']:
      setattr(config, key, getattr(imitation_config, key))

    if config.learner.train_sample_policy:
      logging.warning('Continuing to train sample policy')

    name_map = imitation_state['name_map']
    del imitation_state
  else:
    config_dict = dataclasses.asdict(config)
    sample_policy = saving.policy_from_config_dict(config_dict)
    q_policy = saving.policy_from_config_dict(config_dict)

  if not config.initialize_policies_from and not config.learner.train_sample_policy:
    if restored:
      logging.warning('Not training uninitialized sample_policy.')
    else:
      config.learner.train_sample_policy = True
      logging.warning('Training sample policy.')

  rngs = nnx.Rngs(config.seed)

  q_function = q_lib.QFunction(
      rngs=rngs,
      network_config=config.q_function.network,
      embed_action=q_policy.controller_head.controller_embedding,
      embed_config=config.embed,
      num_names=config.max_names,
  )

  # Multi-device setup
  runtime = config.runtime
  mesh: tp.Optional[jax.sharding.Mesh] = None
  data_sharding = None
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

  mesh = jax_utils.get_mesh()
  data_sharding = jax_utils.data_sharding(mesh)

  learner = learner_lib.Learner(
      config=config.learner,
      q_function=q_function,
      sample_policy=sample_policy,
      q_policy=q_policy,
      rngs=rngs,
      mesh=mesh,
      data_sharding=data_sharding,
  )

  logging.info("Network configuration")
  for comp in ['network', 'controller_head']:
    logging.info(f'Using {comp}: {getattr(config, comp)["name"]}')

  ### Dataset Creation ###
  dataset_config = config.dataset

  train_replays, test_replays = data_lib.train_test_split(dataset_config)
  logging.info(f'Training on {len(train_replays)} replays, testing on {len(test_replays)}')

  if name_map is None:
    name_map = train_lib.create_name_map(train_replays, config.max_names)

  name_map_path = os.path.join(expt_dir, 'name_map.json')
  # Record name map
  print(name_map)
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
      extra_frames=1 + q_policy.delay,
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

  # TrainManager should probably be a proper context manager.
  exit_stack.callback(train_manager.stop)
  exit_stack.callback(test_manager.stop)

  print_losses('initial', train_manager.step()[0])

  if restored:
    assert isinstance(combined_state, dict)  # appease type checker
    jax_utils.set_module_state(learner, combined_state['state'])
    print_losses('post-restore', train_manager.step()[0])
    del combined_state

  # TODO: use orbax instead?
  def save(eval_loss=None):
    nonlocal best_eval_loss
    # Local Save
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
        # dataset_metrics=dataset_metrics,
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
    """Do a test step, then log both train and test stats."""
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

    # [eval_steps, batch_size], mean taken over time
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

    # Calculate the mean eval loss
    eval_loss = mean_stats[learner_lib.Q_FUNCTION]['q']['loss']

    # Save if the eval loss is the best so far
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

    # TODO: Log losses aggregated by name.

  if config.rl_evaluator.use:
    if config.rl_evaluator.opponent:
      opponent_state = saving.load_state_from_disk(
          path=config.rl_evaluator.opponent)
    else:
      # TODO: default to sample policy?
      raise NotImplementedError('opponent not specified')

    AGENT_PORT = 1
    OPPONENT_PORT = 2

    dolphin_kwargs = config.rl_evaluator.dolphin.to_kwargs()
    # Characters are set in the evaluator from the data config.
    players = {
        AGENT_PORT: dolphin_lib.AI(),
        # TODO: allow playing against the in-game CPU like run_evaluator.py
        OPPONENT_PORT: dolphin_lib.AI(),
    }
    dolphin_kwargs.update(players=players)

    common_agent_kwargs = dataclasses.asdict(config.rl_evaluator.agent)

    agent_kwargs: dict[int, dict] = {}
    agent_state = dict(
        state=get_tf_state(),  # could drop everything but 'policy' key
        config=dataclasses.asdict(config),
        name_map=name_map,
    )
    agent_kwargs[AGENT_PORT] = dict(
        state=agent_state,
        **common_agent_kwargs,
    )
    agent_kwargs[OPPONENT_PORT] = dict(
        state=opponent_state,
        **common_agent_kwargs,
    )
    agent_kwargs[OPPONENT_PORT].update(
        name=config.rl_evaluator.opponent_name,
    )

    env_kwargs = {}
    if config.rl_evaluator.async_envs:
      env_kwargs.update(
          num_steps=config.rl_evaluator.num_env_steps,
          inner_batch_size=config.rl_evaluator.inner_batch_size,
      )

    rl_evaluator = evaluators.Evaluator(
        agent_kwargs=agent_kwargs,
        dolphin_kwargs=dolphin_kwargs,
        num_envs=config.rl_evaluator.num_envs,
        async_envs=config.rl_evaluator.async_envs,
        env_kwargs=dict(
            num_steps=config.rl_evaluator.num_env_steps,
            inner_batch_size=config.rl_evaluator.inner_batch_size,
        ),
        use_gpu=config.rl_evaluator.gpu_inference,
        use_fake_envs=config.rl_evaluator.use_fake_envs,
    )

    rl_evaluator.start()

    num_rl_evals = 0

    def rl_evaluate():
      nonlocal num_rl_evals
      num_rl_evals += 1  # increment here to skip first reset

      if num_rl_evals % config.rl_evaluator.reset_every_n_evals == 0:
        logging.info('Resetting Environment')
        rl_evaluator.reset_env()

      rl_evaluator.update_variables({AGENT_PORT: q_policy.variables})
      start_time = time.perf_counter()
      run_time = 0
      rewards = []
      while run_time < config.rl_evaluator.runtime_seconds:
        metrics, timings = rl_evaluator.rollout(
            num_steps=config.rl_evaluator.rollout_length)
        rewards.append(metrics[AGENT_PORT].reward)

        run_time = time.perf_counter() - start_time

      total_reward = sum(rewards)
      num_frames = (
          config.rl_evaluator.num_envs
          * config.rl_evaluator.rollout_length
          * len(rewards))
      num_minutes = num_frames / (60 * 60)
      kdpm = total_reward / num_minutes
      fps = num_frames / run_time

      to_log = dict(
          ko_diff=kdpm,
          num_rollouts=len(rewards),
          num_minutes=num_minutes,
          fps=fps,
          timings=timings,
      )

      print(f'EVAL: kdpm={kdpm}, minutes={num_minutes:.2f}, fps={fps:.1f}\n')
      to_log['time'] = run_time

      total_steps = step.numpy()
      train_lib.log_stats(dict(rl_eval=to_log), total_steps)

    stop_rl_evaluator = rl_evaluator.stop
  else:
    rl_evaluate = lambda: None
    stop_rl_evaluator = lambda: None

  maybe_rl_eval = utils.Periodically(rl_evaluate, config.rl_evaluator.interval_seconds)

  try:

    # For burnin.
    # maybe_eval(force=True)
    maybe_rl_eval()  # guaranteed to run the first time

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

    maybe_eval(force=True)
    rl_evaluate()

  finally:
    stop_rl_evaluator()
