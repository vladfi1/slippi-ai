"""Train two JAX agents against each other."""

import dataclasses
import itertools
import logging
import os
import pickle
import typing as tp

import melee
import numpy as np
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
  name: list[str] = field(lambda: [nametags.DEFAULT_NAME])
  # Character to play. If None, inferred from the teacher checkpoint (only
  # works if it was trained on a single character).
  char: tp.Optional[melee.Character] = None
  compile: bool = True
  batch_steps: int = 0
  async_inference: bool = False


@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = field(RuntimeConfig)

  dolphin: dolphin_lib.DolphinConfig = field(dolphin_lib.DolphinConfig)
  # Base learner config — learner1/learner2 override individual fields.
  learner: learner_lib.LearnerConfig = field(learner_lib.LearnerConfig)
  learner1: learner_lib.LearnerConfig = field(learner_lib.LearnerConfig)
  learner2: learner_lib.LearnerConfig = field(learner_lib.LearnerConfig)

  actor: run_lib.ActorConfig = field(run_lib.ActorConfig)
  p1: AgentConfig = field(AgentConfig)
  p2: AgentConfig = field(AgentConfig)


DEFAULT_CONFIG = Config()
DEFAULT_CONFIG.dolphin.console_timeout = 30

Logger = run_lib.Logger

PORTS = [1, 2]
ENEMY_PORTS = {1: 2, 2: 1}

AGENT_CONFIG_KEY = 'agent_config'


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


class AgentManager:

  def __init__(
      self,
      agent_config: AgentConfig,
      port: int,
      expt_dir: str,
      learner_config: learner_lib.LearnerConfig,
  ):
    self.agent_config = agent_config
    self.port = port
    self.enemy_port = ENEMY_PORTS[port]
    self.expt_dir = expt_dir

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
      self.step = rl_state['step']
      restore_config = flag_utils.dataclass_from_dict(
          AgentConfig, rl_state[AGENT_CONFIG_KEY])
      agent_config.name = restore_config.name
      agent_config.teacher = restore_config.teacher
      if restore_config.char is not None:
        agent_config.char = restore_config.char

    teacher_state = jax_saving.load_state_from_disk(agent_config.teacher)
    self.character = _resolve_character(agent_config, teacher_state, port)

    if not self.found:
      rl_state = teacher_state
      self.save_path = None
      self.step = 0

    teacher_config_dict = jax_saving.upgrade_config(teacher_state['config'])

    self.to_save = {
        'config': teacher_state['config'],
        'name_map': teacher_state['name_map'],
        AGENT_CONFIG_KEY: dataclasses.asdict(agent_config),
    }
    self.name = agent_config.name

    teacher = jax_saving.load_policy_from_state(teacher_state)
    policy = jax_saving.load_policy_from_state(rl_state)

    pretraining_config = flag_utils.dataclass_from_dict(
        jax_train_lib.Config, teacher_config_dict)
    value_function = jax_train_lib.value_function_from_config(
        pretraining_config, rngs=nnx.Rngs(port))

    self.learner = learner_lib.Learner(
        config=learner_config,
        teacher=teacher,
        policy=policy,
        value_function=value_function,
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
    return self.learner.policy_variables()

  def get_state(self) -> dict:
    return dict(
        state=self.learner.get_state(),
        **self.to_save,
    )

  def save(self, step: int):
    state = self.get_state()
    state['step'] = step
    pickled_state = pickle.dumps(state)
    logging.info('saving state to %s', self.save_path)
    with open(self.save_path, 'wb') as f:
      f.write(pickled_state)

  def agent_kwargs(self) -> dict:
    return dict(
        state=self.get_state(),
        compile=self.agent_config.compile,
        batch_steps=self.agent_config.batch_steps,
        async_inference=self.agent_config.async_inference,
    )


class ExperimentManager:

  def __init__(
      self,
      config: Config,
      agents: dict[int, AgentManager],
      build_actor: tp.Callable[[], evaluators.RolloutWorker],
  ):
    self._config = config
    self._agents = agents
    self._build_actor = build_actor
    self._unroll_length = config.actor.rollout_length
    self._num_ppo_batches = config.learner.ppo.num_batches
    self._burnin_steps_after_reset = config.runtime.burnin_steps_after_reset

    batch_size = config.actor.num_envs

    self._learners = {port: agent.learner for port, agent in agents.items()}
    self._hidden_states = {
        port: learner.initial_state(batch_size)
        for port, learner in self._learners.items()
    }

    self.update_profiler = utils.Profiler(burnin=0)
    self.learner_profiler = utils.Profiler()
    self.rollout_profiler = utils.Profiler()
    self.reset_profiler = utils.Profiler(burnin=0)

    with self.reset_profiler:
      self.actor = self._build_actor()
      self.actor.start()
      self.num_rollouts = 0
      self._burnin_after_reset()

  def _rollout(self) -> tuple[dict[int, evaluators.Trajectory], dict]:
    self.num_rollouts += 1
    return self.actor.rollout(self._unroll_length)

  def unroll(self):
    """Advance hidden states without training (for burnin)."""
    trajectory, _ = self._rollout()
    for port, learner in self._learners.items():
      _, self._hidden_states[port] = learner.unroll(
          trajectory[port], self._hidden_states[port])

  def _burnin_after_reset(self):
    for _ in range(self._burnin_steps_after_reset):
      self.unroll()

  def step(
      self,
      ppo_steps: tp.Optional[int] = None,
  ) -> tuple[dict[int, list[evaluators.Trajectory]], dict]:
    reset_interval = self._config.runtime.reset_every_n_steps
    if reset_interval and self.num_rollouts >= reset_interval:
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
      actor_metrics = []
      for _ in range(self._num_ppo_batches):
        trajectory, timings = self._rollout()
        for port in PORTS:
          trajectories[port].append(trajectory[port])
        actor_metrics.append(timings)

      actor_metrics = utils.map_nt(lambda *xs: np.mean(xs), *actor_metrics)

    with self.learner_profiler:
      metrics = {}
      for port, learner in self._learners.items():
        self._hidden_states[port], metrics[port] = learner.ppo(
            trajectories[port], self._hidden_states[port],
            num_epochs=ppo_steps)

    return trajectories, dict(learner=metrics, actor=actor_metrics)


def run(config: Config):
  tag = config.runtime.tag or jax_train_lib.get_experiment_tag()
  expt_dir = config.runtime.expt_dir
  if expt_dir is None:
    expt_dir = os.path.join(config.runtime.expt_root, tag)
    os.makedirs(expt_dir, exist_ok=True)
    config.runtime.expt_dir = expt_dir
  logging.info('experiment directory: %s', expt_dir)

  batch_size = config.actor.num_envs

  agents: dict[int, AgentManager] = {}
  for port in PORTS:
    agents[port] = AgentManager(
        agent_config=getattr(config, f'p{port}'),
        port=port,
        expt_dir=expt_dir,
        learner_config=getattr(config, f'learner{port}'),
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
        inner_batch_size=config.actor.inner_batch_size,
    )

  build_actor = lambda: evaluators.RolloutWorker(
      agent_kwargs=agent_kwargs,
      dolphin_kwargs=dolphin_kwargs,
      env_kwargs=env_kwargs,
      num_envs=config.actor.num_envs,
      async_envs=config.actor.async_envs,
      use_gpu=config.actor.gpu_inference,
      use_fake_envs=config.actor.use_fake_envs,
  )

  experiment_manager = ExperimentManager(
      config=config,
      agents=agents,
      build_actor=build_actor,
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
    steps_per_rollout = config.actor.num_envs * config.actor.rollout_length
    fps = len(trajectories) * steps_per_rollout / step_time
    mps = fps / MINUTES_PER_FRAME

    timings = dict(
        rollout=experiment_manager.rollout_profiler.mean_time(),
        learner=experiment_manager.learner_profiler.mean_time(),
        reset=experiment_manager.reset_profiler.mean_time(),
        total=step_time,
        fps=fps,
        mps=mps,
    )
    actor_timing = metrics['actor'].pop('timing')
    for key in ['env_pop', 'env_push']:
      timings[key] = actor_timing[key]
    for key in ['agent_step']:
      timings[key] = actor_timing[key][main_port]

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

    log_data = dict(
        p0=p0_stats,
        p1=p1_stats,
        timings=timings,
        actor=metrics['actor'],
        learner=metrics['learner'][main_port],
        learner2=metrics['learner'][other_port],
    )

    return log_data

  logger = Logger()
  steps_per_epoch = config.learner.ppo.num_batches

  def flush(step: int):
    metrics = logger.flush(step * steps_per_epoch)
    if metrics is None:
      return

    print('\nStep:', step)

    timings: dict = metrics['timings']
    timing_str = ', '.join(
        '{k}: {v:.3f}'.format(k=k, v=v) for k, v in timings.items())
    print(timing_str)

    learner_metrics = metrics['learner']
    pre_update = learner_metrics['ppo_step']['0']
    mean_actor_kl = pre_update['actor_kl']['mean']
    max_actor_kl = pre_update['actor_kl']['max']
    print(f'actor_kl: mean={mean_actor_kl:.3g} max={max_actor_kl:.3g}')
    teacher_kl = pre_update['teacher_kl']
    print(f'teacher_kl: {teacher_kl:.3g}')
    print(f'uev: {learner_metrics["value"]["uev"]:.3f}')

  maybe_flush = utils.Periodically(flush, config.runtime.log_interval)

  def save(step: int):
    for agent in agents.values():
      agent.save(step)

  maybe_save = utils.Periodically(save, config.runtime.save_interval)

  agent0 = agents[PORTS[0]]
  step = agent0.step  # Both agents share the same step counter.

  try:
    logging.info('Main training loop')

    while step < config.runtime.max_step:
      with step_profiler:
        trajectories, metrics = experiment_manager.step()

      if experiment_manager.learner_profiler.num_calls > 0:
        logger.record(get_log_data(trajectories, metrics))
        maybe_flush(step)

      step += 1
      maybe_save(step)

    save(step)

  finally:
    experiment_manager.actor.stop()
