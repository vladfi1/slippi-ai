"""Train two JAX agents against each other."""

import concurrent.futures
import contextlib
import dataclasses
import itertools
import logging
import os
import pickle
import typing as tp

import melee
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from slippi_ai import (
    dolphin as dolphin_lib,
    eval_lib,
    evaluators,
    flag_utils,
    nametags,
    reward,
    utils,
)
from slippi_ai.jax import saving as jax_saving
from slippi_ai.jax import train_lib as jax_train_lib
from slippi_ai.jax import train_vf
from slippi_ai.jax.rl import learner as learner_lib
from slippi_ai.jax.rl import run_lib
from slippi_ai.types import Game

field = lambda f: dataclasses.field(default_factory=f)


@dataclasses.dataclass
class RuntimeConfig:
  expt_root: str = 'experiments/jax/train_two'
  expt_dir: tp.Optional[str] = None
  tag: tp.Optional[str] = None

  max_step: int = 10
  max_runtime: tp.Optional[int] = None
  log_interval: int = 10   # seconds between logging
  save_interval: int = 300  # seconds between saving to disk

  reset_every_n_steps: tp.Optional[int] = None
  burnin_steps_after_reset: int = 0


@dataclasses.dataclass
class AgentConfig:
  teacher: tp.Optional[str] = None
  # Will override a value function in the teacher checkpoint. Necessary if the
  # teacher has no value function i.e. was produced by policy-only training.
  value_function: tp.Optional[str] = None
  name: list[str] = field(lambda: [nametags.DEFAULT_NAME])
  # Rating to condition the agent on. Required when the teacher was trained
  # with ratings (embed.with_rating); ignored otherwise.
  rating: tp.Optional[float] = None
  # Character to play. If None, inferred from the teacher checkpoint (only
  # works if it was trained on a single character).
  char: tp.Optional[melee.Character] = None
  compile: bool = True
  batch_steps: int = 0
  async_inference: bool = False
  override_delay: tp.Optional[int] = None
  jax: run_lib.JaxAgentConfig = field(run_lib.JaxAgentConfig)

def default_learner_config():
  return learner_lib.LearnerConfig(
      optimizer_burnin_epochs=0,
      value_burnin_epochs=0,
  )

@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = field(RuntimeConfig)

  dolphin: dolphin_lib.DolphinConfig = field(dolphin_lib.DolphinConfig)
  # Base learner config — learner1/learner2 override individual fields.
  learner: learner_lib.LearnerConfig = field(default_learner_config)
  learner1: learner_lib.LearnerConfig = field(default_learner_config)
  learner2: learner_lib.LearnerConfig = field(default_learner_config)

  actor: run_lib.ActorConfig = field(run_lib.ActorConfig)
  p1: AgentConfig = field(AgentConfig)
  p2: AgentConfig = field(AgentConfig)


DEFAULT_CONFIG = Config()
DEFAULT_CONFIG.dolphin.console_timeout = 30

Logger = run_lib.Logger

PORTS = [1, 2]
ENEMY_PORTS = {1: 2, 2: 1}

AGENT_CONFIG_KEY = 'agent_config'


def assign_devices(
    ports: tp.Sequence[int] = PORTS,
) -> dict[int, tp.Optional[jax.Device]]:
  """Give each port's agent/learner its own device when several are available."""
  devices = jax.local_devices()
  if len(devices) < 2:
    return {port: None for port in ports}

  assignment = {port: devices[i % len(devices)] for i, port in enumerate(ports)}
  for port, device in assignment.items():
    logging.info('Port %d: using device %s', port, device)
  return assignment


def _resolve_character(
    agent_config: AgentConfig,
    teacher_state: dict,
    port: int,
) -> melee.Character:
  """Determine the character for this agent from config or teacher checkpoint."""
  if agent_config.char is not None:
    return agent_config.char

  allowed_chars = eval_lib.allowed_characters(teacher_state['config'])
  if allowed_chars is not None and len(allowed_chars) == 1:
    return allowed_chars[0]

  raise ValueError(
      f'Port {port}: must specify --config.p{port}.char when teacher '
      'allows multiple characters.')


def _merge_value_function(rl_state: dict, vf_path: str, port: int):
  """Merges a separately-trained value function into an imitation state."""
  vf_state = jax_saving.load_state_from_disk(vf_path)
  errors = train_vf.check_compatibility(vf_state, rl_state)
  if errors:
    raise ValueError(
        f'Port {port}: incompatible Policy and VF:\n' + '\n'.join(errors))

  # TODO: share code with the merge_checkpoints script
  for key in vf_state['state']:
    if key in rl_state['state']:
      logging.warning(f'State "{key}" is being overridden from {vf_path}')

  rl_state['state'].update(vf_state['state'])  # key names are compatible
  rl_state['config']['value_function'] = dict(
      separate_network_config=True,
      network=vf_state['config']['network'],
  )


class AgentManager:

  def __init__(
      self,
      agent_config: AgentConfig,
      port: int,
      expt_dir: str,
      learner_config: learner_lib.LearnerConfig,
      device: tp.Optional[jax.Device] = None,
  ):
    self.agent_config = agent_config
    self.port = port
    self.enemy_port = ENEMY_PORTS[port]
    self.expt_dir = expt_dir
    self.device = device

    suffix = f'-{port}.pkl'
    self.found = False
    for name in os.listdir(expt_dir):
      if name.endswith(suffix):
        self.found = True
        break

    if self.found:
      self.save_path = os.path.join(expt_dir, name)
      logging.info(f'Port {port}: restoring from {self.save_path}')
      rl_state = jax_saving.load_state_from_disk(self.save_path)
      self.step: int = rl_state['step']
      restore_config = flag_utils.dataclass_from_dict(
          AgentConfig, rl_state[AGENT_CONFIG_KEY])

      for key in ['teacher', 'value_function']:
        requested = getattr(agent_config, key)
        previous = getattr(restore_config, key)
        if requested and requested != previous:
          raise ValueError(
              f'Port {port}: requested {key} does not match checkpoint: '
              f'{requested} (requested) != {previous} (checkpoint)')
        setattr(agent_config, key, previous)

      agent_config.name = restore_config.name
      if restore_config.char is not None:
        agent_config.char = restore_config.char
    elif not agent_config.teacher:
      raise ValueError(
          f'Port {port}: no checkpoint found in {expt_dir} and no teacher specified.')

    assert agent_config.teacher is not None
    teacher_state = jax_saving.load_state_from_disk(agent_config.teacher)
    self.character = _resolve_character(agent_config, teacher_state, port)

    if self.found:
      rl_delay = rl_state['config']['policy']['delay']
      teacher_delay = teacher_state['config']['policy']['delay']
      if rl_delay != teacher_delay:
        raise ValueError(
            f'Port {port}: teacher delay does not match RL state delay: '
            f'{teacher_delay} != {rl_delay}.')
    else:
      rl_state = teacher_state
      self.save_path = None
      self.step = 0

      # A restored checkpoint already contains the value function, so only
      # merge a separate one when initializing from the teacher.
      if agent_config.value_function:
        _merge_value_function(rl_state, agent_config.value_function, port)

      run_lib.reset_optimizer_steps(rl_state)

    if 'value_function' not in rl_state['config']:
      raise ValueError(
          f'Port {port}: teacher was not trained with a value function; pass '
          f'--config.p{port}.value_function to supply one.')

    if agent_config.override_delay is not None:
      for state in [rl_state, teacher_state]:
        state['config']['policy']['delay'] = agent_config.override_delay

    # Must be rl_state's config, not the teacher's: the two differ when a
    # separate value function is merged in (agent_config.value_function), and
    # saving the teacher's config would drop that override, breaking the next
    # restore.
    rl_config_dict = jax_saving.upgrade_config(rl_state['config'])

    self.to_save = {
        'config': rl_state['config'],
        'name_map': rl_state['name_map'],
        AGENT_CONFIG_KEY: dataclasses.asdict(agent_config),
    }
    self.name = agent_config.name

    teacher = jax_saving.load_policy_from_state(teacher_state)
    policy = jax_saving.load_policy_from_state(rl_state)

    pretraining_config = flag_utils.dataclass_from_dict(
        jax_train_lib.Config, rl_config_dict)
    value_function = jax_train_lib.value_function_from_config(
        pretraining_config, rngs=nnx.Rngs(port))

    self.learner = learner_lib.Learner(
        config=learner_config,
        teacher=teacher,
        policy=policy,
        value_function=value_function,
        device=device,
    )
    self.learner.restore_from_imitation(rl_state['state'])

  def set_opponent(self, character: melee.Character):
    char_name = self.character.name.lower()
    opp_name = character.name.lower()
    delay = self.learner.policy.delay
    save_name = f'{char_name}_delay_{delay}_vs_{opp_name}-{self.port}.pkl'
    save_path = os.path.join(self.expt_dir, save_name)
    if self.save_path is not None:
      assert save_path == self.save_path
    self.save_path = save_path
    self.to_save['opponent'] = opp_name

  def policy_variables(self):
    # Note: the parameters returned with to_numpy=False will be invalidated by
    # jax buffer donation on the next learner update, but we don't need to make
    # a copy because we use them only for the next rollout.
    return self.learner.policy_variables(to_numpy=False)

  def get_state(self) -> dict:
    return dict(
        state=self.learner.get_state(),
        **self.to_save,
    )

  def save(self, step: int):
    if self.save_path is None:
      raise ValueError('save_path not set for agent on port {}'.format(self.port))

    state = self.get_state()
    state['step'] = step
    pickled_state = pickle.dumps(state)
    logging.info('saving state to %s', self.save_path)
    with open(self.save_path, 'wb') as f:
      f.write(pickled_state)

  def agent_kwargs(self) -> dict:
    jax_kwargs = dataclasses.asdict(self.agent_config.jax)
    # Run inference on the same device as this port's learner, unless
    # explicitly overridden in the config.
    if jax_kwargs.get('device_index') is None and self.device is not None:
      jax_kwargs['device_index'] = jax.local_devices().index(self.device)
    return dict(
        state=self.get_state(),
        compile=self.agent_config.compile,
        batch_steps=self.agent_config.batch_steps,
        async_inference=self.agent_config.async_inference,
        rating=self.agent_config.rating,
        jax=jax_kwargs,
    )


class ExperimentManager:

  def __init__(
      self,
      config: Config,
      agents: dict[int, AgentManager],
      build_actor: tp.Callable[[], evaluators.AbstractRolloutWorker],
      exit_stack: tp.Optional[contextlib.ExitStack] = None,
  ):
    self._config = config
    self._agents = agents
    self._build_actor = build_actor
    self._unroll_length = config.actor.rollout_length
    self._num_ppo_batches = config.learner.ppo.num_batches
    self._burnin_steps_after_reset = config.runtime.burnin_steps_after_reset

    self._learner_dtype = jnp.float32
    self._agent_dtypes = {
        port: agent.agent_config.jax.dtype.dtype
        for port, agent in agents.items()
    }

    batch_size = config.actor.num_envs

    self._learners = {port: agent.learner for port, agent in agents.items()}
    self._hidden_states = {
        port: learner.initial_state(batch_size)
        for port, learner in self._learners.items()
    }
    # Fold per-frame rollouts into frame-skipped trajectories, one per port.
    self._converters = {
        port: learner_lib.FrameSkipConverter(
            frame_skip=learner.policy.frame_skip,
            batch_size=batch_size,
            dummy_sample_outputs=learner.policy.controller_head.dummy_sample_outputs(
                [batch_size]),
            reward_config=learner._config.reward,
        )
        for port, learner in self._learners.items()
    }

    # Dispatching a learner update is substantial host-side work: XLA:GPU
    # executes a scan/While loop by launching each iteration's kernels from
    # the calling thread, so a jitted call over our 80-step unrolls occupies
    # the host for roughly the module's execution time (at large batch the
    # CUDA launch queue also fills up, making "dispatch" fully synchronous).
    # A single thread therefore cannot keep two devices busy no matter the
    # dispatch order. Run each learner's update on its own thread instead;
    # the XLA module executes with the GIL released, so the launch work
    # genuinely parallelizes (measured ~1.9x on 2x3080Ti at num_envs=2048,
    # learner phase 5.1s -> 2.66s).
    #
    # Notes from investigating alternatives (2026-08, jax 0.10.1):
    # - Removing host syncs and dispatching both learners from one thread is
    #   NOT sufficient; async dispatch is a myth for scan-heavy modules.
    #   Simple chained-matmul benchmarks DO overlap (~2x) and won't
    #   reproduce the problem.
    # - XLA command buffers (--xla_gpu_enable_command_buffer=...,WHILE)
    #   captured only the loop bodies into CUDA graphs; the While loop still
    #   iterates on the host (one small cuGraphLaunch per iteration), so
    #   learner host time was unchanged. Worth re-testing after XLA upgrades
    #   in case whole-loop capture (conditional graph nodes) lands.
    self._learner_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(self._learners))

    self.update_profiler = utils.Profiler(burnin=0)
    self.learner_profiler = utils.Profiler()
    self.rollout_profiler = utils.Profiler()
    self.reset_profiler = utils.Profiler(burnin=0)

    if not self._config.dolphin.infinite_time:
      logging.info('Finite time mode, disabling env resets')
      self.reset_interval = None
    elif config.actor.use_sim_envs:
      # JaxSimRolloutWorker.reset_env is not implemented.
      logging.info('Sim envs, disabling env resets')
      self.reset_interval = None
    else:
      self.reset_interval = config.runtime.reset_every_n_steps

    with self.reset_profiler:
      self.actor = self._build_actor()
      self.actor.start()

      if exit_stack is not None:
        exit_stack.callback(self.actor.stop)

      self.num_rollouts = 0
      self._burnin_after_reset()

  def _rollout(self) -> tuple[
      dict[int, learner_lib.FrameSkipTrajectory],
      tp.Mapping[int, evaluators.Trajectory],
      dict]:
    self.num_rollouts += 1
    trajectories, timings = self.actor.rollout(self._unroll_length)

    fs_trajectories = {
        port: self._converters[port].convert(trajectories[port])
        for port in PORTS
    }

    # Agents may run at a lower precision than the learner.
    for port, agent_dtype in self._agent_dtypes.items():
      if agent_dtype != self._learner_dtype:
        fs_trajectory = fs_trajectories[port]
        fs_trajectories[port] = fs_trajectory._replace(
            initial_state=run_lib.cast_floats(
                fs_trajectory.initial_state, dtype=self._learner_dtype))

    return fs_trajectories, trajectories, timings

  def unroll(self):
    """Advance hidden states without training (for burnin)."""
    fs_trajectories, _, _ = self._rollout()
    for port, learner in self._learners.items():
      _, self._hidden_states[port] = learner.unroll(
          fs_trajectories[port], self._hidden_states[port])

  def _burnin_after_reset(self):
    for _ in range(self._burnin_steps_after_reset):
      self.unroll()

  def step(
      self,
      step: int,
  ) -> tuple[dict[int, list[evaluators.Trajectory]], dict]:
    if self.reset_interval and self.num_rollouts >= self.reset_interval:
      logging.info('Resetting environments')
      with self.reset_profiler:
        self.actor.reset_env()
        self.num_rollouts = 0
        self._burnin_after_reset()

    with self.update_profiler:
      variables = {
          port: agent.policy_variables()
          for port, agent in self._agents.items()
      }
      self.actor.update_variables(variables)

    with self.rollout_profiler:
      trajectories: dict[int, list[evaluators.Trajectory]] = {
          port: [] for port in PORTS
      }
      fs_trajectories: dict[int, list[learner_lib.FrameSkipTrajectory]] = {
          port: [] for port in PORTS
      }
      actor_metrics = []
      for _ in range(self._num_ppo_batches):
        fs_trajectory, trajectory, timings = self._rollout()
        for port in PORTS:
          trajectories[port].append(trajectory[port])
          fs_trajectories[port].append(fs_trajectory[port])
        actor_metrics.append(timings)

      for metrics in actor_metrics:
        metrics.pop('completed_games', None)
      actor_metrics = utils.map_nt(lambda *xs: np.mean(xs), *actor_metrics)

    with self.learner_profiler:
      futures = {
          port: self._learner_pool.submit(
              learner.ppo,
              fs_trajectories[port], self._hidden_states[port], step=step)
          for port, learner in self._learners.items()
      }
      metrics = {}
      for port, future in futures.items():
        self._hidden_states[port], metrics[port] = future.result()

      for port, learner in self._learners.items():
        learner.check_actor_kl(metrics[port])

    return trajectories, dict(learner=metrics, actor=actor_metrics)


def run(config: Config):
  with contextlib.ExitStack() as exit_stack:
    _run(config, exit_stack)

def _run(config: Config, exit_stack: contextlib.ExitStack):
  tag = config.runtime.tag or jax_train_lib.get_experiment_tag()
  expt_dir = config.runtime.expt_dir
  if expt_dir is None:
    expt_dir = os.path.join(config.runtime.expt_root, tag)
    os.makedirs(expt_dir, exist_ok=True)
    config.runtime.expt_dir = expt_dir
  logging.info('experiment directory: %s', expt_dir)

  batch_size = config.actor.num_envs

  port_devices = assign_devices()

  agents: dict[int, AgentManager] = {}
  for port in PORTS:
    agents[port] = AgentManager(
        agent_config=getattr(config, f'p{port}'),
        port=port,
        expt_dir=expt_dir,
        learner_config=getattr(config, f'learner{port}'),
        device=port_devices[port],
    )

  # Set up opponent info (used for save file naming).
  for port, agent in agents.items():
    agent.set_opponent(agents[ENEMY_PORTS[port]].character)

  # Build per-port name lists by cycling over all name combinations.
  name_combinations = list(itertools.product(
      config.p1.name, config.p2.name))
  name_combination_batch = list(itertools.islice(
      itertools.cycle(name_combinations), batch_size))

  port_to_names: dict[int, list[str]] = {}
  for port, *names in zip(PORTS, *name_combination_batch):
    port_to_names[port] = names

  agent_kwargs = {port: agent.agent_kwargs() for port, agent in agents.items()}
  for port, names in port_to_names.items():
    agent_kwargs[port]['name'] = names

  dolphin_kwargs = dict(
      players={
          port: dolphin_lib.AI(character=agent.character)
          for port, agent in agents.items()
      },
      **config.dolphin.to_kwargs(),
  )

  env_kwargs: dict[str, tp.Any] = dict(
      swap_ports=config.actor.num_envs > 1)
  if config.actor.async_envs:
    env_kwargs.update(
        num_steps=config.actor.num_env_steps,
        inner_batch_size=config.actor.get_inner_batch_size(),
    )

  build_actor: tp.Callable[[], evaluators.AbstractRolloutWorker]
  build_actor = lambda: evaluators.RolloutWorker(
      agent_kwargs=agent_kwargs,
      dolphin_kwargs=dolphin_kwargs,
      env_kwargs=env_kwargs,
      num_envs=config.actor.num_envs,
      async_envs=config.actor.async_envs,
      use_gpu=config.actor.gpu_inference,
      use_fake_envs=config.actor.use_fake_envs,
  )

  if config.actor.use_sim_envs:
    if config.actor.async_envs and (
        config.actor.num_envs % config.actor.get_inner_batch_size()):
      raise ValueError(
          f'num_envs={config.actor.num_envs} must be divisible by '
          f'inner_batch_size={config.actor.get_inner_batch_size()} for sim RL.')

    for port, agent in agents.items():
      batch_steps = agent.agent_config.batch_steps
      delay = agent.learner.policy.delay
      if batch_steps > delay:
        raise ValueError(
            f'Port {port}: agent.batch_steps={batch_steps} exceeds policy '
            f'delay {delay} for sim RL.')
      if config.actor.rollout_length % max(1, batch_steps):
        raise ValueError(
            f'Port {port}: agent.batch_steps must divide rollout_length '
            'for sim RL.')

    def build_actor() -> evaluators.AbstractRolloutWorker:
      from slippi_ai.sim_env import jax_rollout

      # Unlike self-play, the two ports have distinct policies, so each gets
      # its own agent instead of a single one batched over both ports.
      return jax_rollout.JaxSimRolloutWorker(
          agent_kwargs=agent_kwargs,
          dolphin_kwargs=dolphin_kwargs,
          num_envs=config.actor.num_envs,
          rollout_length=config.actor.rollout_length,
          use_fake_envs=config.actor.use_fake_envs,
          async_envs=config.actor.async_envs,
          inner_batch_size=config.actor.get_inner_batch_size(),
          # When there's a single ppo batch we immediately use the trajectory
          # data without invalidating it by calling rollout again, so there's
          # no need to make a copy of the data.
          copy_data=config.learner.ppo.num_batches > 1,
          keep_agent_outputs_on_device=config.actor.keep_agent_outputs_on_device,
      )

  experiment_manager = ExperimentManager(
      config=config,
      agents=agents,
      build_actor=build_actor,
      exit_stack=exit_stack,
  )

  step_profiler = utils.Profiler()

  MINUTES_PER_FRAME = 60 * 60

  def get_log_data(
      all_trajectories: dict[int, list[evaluators.Trajectory]],
      metrics: dict,
  ) -> dict:
    main_port, other_port = PORTS
    trajectories = all_trajectories[main_port]

    step_time = step_profiler.mean_time()
    sps = 1 / step_time
    steps_per_rollout = config.actor.num_envs * config.actor.rollout_length
    fps = len(trajectories) * steps_per_rollout * sps
    mps = fps / MINUTES_PER_FRAME

    timings = dict(
        rollout=experiment_manager.rollout_profiler.mean_time(),
        learner=experiment_manager.learner_profiler.mean_time(),
        reset=experiment_manager.reset_profiler.mean_time(),
        total=step_time,
        sps=sps,
        fps=fps,
        mps=mps,
    )
    timings['actor'] = utils.map_nt(
        lambda x: x * 1000, metrics['actor'].pop('timing'))

    states: Game = utils.map_nt(
        lambda *xs: np.stack(xs, axis=1),
        *[t.states for t in trajectories])

    learner1_config = config.learner1
    learner2_config = config.learner2
    p0_stats = reward.player_stats(
        states.p0, states.p1, states.stage,
        stalling_threshold=learner1_config.reward.stalling_threshold)
    p1_stats = reward.player_stats(
        states.p1, states.p0, states.stage,
        stalling_threshold=learner2_config.reward.stalling_threshold)
    ko_diff = p1_stats['deaths'] - p0_stats['deaths']

    log_data = dict(
        p0=p0_stats,
        p1=p1_stats,
        ko_diff=ko_diff,
        timings=timings,
        actor=metrics['actor'],
        learner=metrics['learner'][main_port],
        learner2=metrics['learner'][other_port],
    )

    return log_data

  logger = Logger()
  steps_per_epoch = config.learner.ppo.num_batches
  frames_per_step = config.actor.num_envs * config.actor.rollout_length

  def flush(step: int):
    total_steps = step * steps_per_epoch
    total_frames = total_steps * frames_per_step
    extras = dict(
        total_frames=total_frames,
    )

    metrics = logger.flush(total_steps, extras=extras)
    if metrics is None:
      return

    for profiler in [
        step_profiler, experiment_manager.rollout_profiler,
        experiment_manager.learner_profiler, experiment_manager.reset_profiler,
    ]:
      profiler.reset()

    print('\nStep:', step)

    timings: dict = metrics['timings']
    timings = utils.map_nt(lambda v: f'{v:.3f}', timings)
    print(timings)

    learner_metrics = metrics['learner']
    post_update = learner_metrics['post_update']
    mean_actor_kl = post_update['actor_kl']['mean']
    max_actor_kl = post_update['actor_kl']['max']
    print(f'actor_kl: mean={mean_actor_kl:.3g} max={max_actor_kl:.3g}')
    teacher_kl = post_update['teacher_kl']
    print(f'teacher_kl: {teacher_kl:.3g}')
    print(f'uev: {learner_metrics["value"]["uev"]:.3f}')

  maybe_flush = utils.Periodically(flush, config.runtime.log_interval)

  def save(step: int):
    for agent in agents.values():
      agent.save(step)

  maybe_save = utils.Periodically(save, config.runtime.save_interval)

  agent0 = agents[PORTS[0]]
  step = agent0.step  # Both agents share the same step counter.

  # TODO: flush logger at optimizer/value burnin boundaries
  for learner_config in [config.learner1, config.learner2]:
    assert learner_config.optimizer_burnin_epochs == 0
    assert learner_config.value_burnin_epochs == 0

  logging.info('Main training loop')

  while step < config.runtime.max_step:
    with step_profiler:
      trajectories, metrics = experiment_manager.step(step)

    if experiment_manager.learner_profiler.num_calls > 0:
      logger.record(get_log_data(trajectories, metrics))
      maybe_flush(step)

    step += 1
    maybe_save(step)

  save(step)
