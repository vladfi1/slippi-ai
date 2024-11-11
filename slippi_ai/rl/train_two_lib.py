"""Train two agents against each other."""

import dataclasses
import enum
import logging
import os
import pickle
import typing as tp

import numpy as np
import tensorflow as tf

from slippi_ai import (
    data,
    dolphin as dolphin_lib,
    embed,
    evaluators,
    flag_utils,
    nametags,
    reward,
    saving,
    tf_utils,
    train_lib,
    utils,
)

import melee

from slippi_ai import value_function as vf_lib
from slippi_ai.rl import learner as learner_lib
from slippi_ai.rl import run_lib

field = lambda f: dataclasses.field(default_factory=f)

@dataclasses.dataclass
class RuntimeConfig:
  expt_root: str = 'experiments/train_two'
  expt_dir: tp.Optional[str] = None
  tag: tp.Optional[str] = None

  max_step: int = 10  # maximum training step
  max_runtime: tp.Optional[int] = None  # maximum runtime in seconds
  log_interval: int = 10  # seconds between logging
  save_interval: int = 300  # seconds between saving to disk

  # Periodically reset the environments to deal with memory leaks in dolphin.
  reset_every_n_steps: tp.Optional[int] = None
  # Without burnin, we see a spike in teacher_kl after every reset. My guess is
  # that this is because the trajectories and therefore gradients become highly
  # correlated, which is bad. Empirically 10 is a good value to set this to.
  burnin_steps_after_reset: int = 0

@dataclasses.dataclass
class AgentConfig:
  teacher: tp.Optional[str] = None
  name: str = nametags.DEFAULT_NAME

  compile: bool = True
  jit_compile: bool = False
  batch_steps: int = 0
  async_inference: bool = False


@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = field(RuntimeConfig)

  # num_actors: int = 1
  dolphin: dolphin_lib.DolphinConfig = field(dolphin_lib.DolphinConfig)
  learner: learner_lib.LearnerConfig = field(learner_lib.LearnerConfig)
  learner1: learner_lib.LearnerConfig = field(learner_lib.LearnerConfig)
  learner2: learner_lib.LearnerConfig = field(learner_lib.LearnerConfig)

  actor: run_lib.ActorConfig = field(run_lib.ActorConfig)
  p1: AgentConfig = field(AgentConfig)
  p2: AgentConfig = field(AgentConfig)

  # Take learner steps without changing the parameters to burn-in the
  # optimizer state for RL.
  optimizer_burnin_steps: int = 0
  # Take some steps to update just the value function before doing RL.
  # Useful if we're training against the level 9 cpu.
  value_burnin_steps: int = 0

DEFAULT_CONFIG = Config()
DEFAULT_CONFIG.dolphin.console_timeout = 30

Logger = run_lib.Logger

def get_pretraining_character(
    config: train_lib.Config) -> tp.Optional[melee.Character]:
  allowed_characters = config.dataset.allowed_characters
  character_list = data.chars_from_string(allowed_characters)
  if character_list is not None and len(character_list) == 1:
    return character_list[0]
  return None

PORTS = [1, 2]
ENEMY_PORTS = {
   1: 2,
   2: 1,
}

AGENT_CONFIG_KEY = 'agent_config'

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
    self.learner_config = learner_config

    suffix = f'-{port}.pkl'
    self.found = False
    for name in os.listdir(expt_dir):
      if name.endswith(suffix):
        self.found = True
        break

    if self.found:
      self.save_name = name
      self.save_path = os.path.join(expt_dir, name)
      logging.info(f'Restoring from {self.save_path}')
      rl_state = saving.load_state_from_disk(self.save_path)
      self.step = rl_state['step']
      restore_config = flag_utils.dataclass_from_dict(
          AgentConfig, rl_state[AGENT_CONFIG_KEY])
      agent_config.name = restore_config.name
      agent_config.teacher = restore_config.teacher

    # TODO: check if teacher is itself an RL-trained model
    teacher_state = saving.load_state_from_disk(agent_config.teacher)
    teacher_config = flag_utils.dataclass_from_dict(
        train_lib.Config, teacher_state['config'])
    self.character = get_pretraining_character(teacher_config)

    if self.character is None:
      raise ValueError('Must be pretrained on single character')

    if not self.found:
      rl_state = teacher_state
      self.save_path = None
      self.step = 0

    self.to_save = {
        'config': teacher_state['config'],
        'name_map': teacher_state['name_map'],
        AGENT_CONFIG_KEY: dataclasses.asdict(agent_config),
    }
    self.name = agent_config.name

    # Make sure we don't train the teacher
    with tf_utils.non_trainable_scope():
      teacher = saving.load_policy_from_state(teacher_state)
    self.policy = saving.load_policy_from_state(rl_state)

    # TODO: put this code into saving.py or train_lib.py
    vf_config = teacher_config.value_function
    value_function = None
    if vf_config.train_separate_network:
      value_net_config = teacher_config.network
      if vf_config.separate_network_config:
        value_net_config = vf_config.network
      value_function = vf_lib.ValueFunction(
          network_config=value_net_config,
          embed_state_action=self.policy.embed_state_action,
      )

    self.learner = learner_lib.Learner(
        config=learner_config,
        teacher=teacher,
        policy=self.policy,
        value_function=value_function,
    )
    self.learning_rate = self.learner.learning_rate

    # Initialize and restore variables
    self.learner.initialize(run_lib.dummy_trajectory(self.policy, 1, 1))
    self.learner.restore_from_imitation(rl_state['state'])

  def set_opponent(self, character: melee.Character):
    self_char = self.character.name.lower()
    opp_char = character.name.lower()
    name = f'{self_char}_delay_{self.policy.delay}_vs_{opp_char}-{self.port}.pkl'
    save_path = os.path.join(self.expt_dir, name)
    if self.save_path:
      assert save_path == self.save_path
    self.save_path = save_path
    self.to_save['opponent'] = opp_char

  def get_state(self):
    return dict(
        state=self.learner.get_state(),
        **self.to_save,
    )

  def save(self, step: int):
    # Note: this state is valid as an imitation state.
    state = self.get_state()
    state['step'] = step
    pickled_state = pickle.dumps(state)

    logging.info('saving state to %s', self.save_path)
    with open(self.save_path, 'wb') as f:
      f.write(pickled_state)

  def agent_kwargs(self) -> dict:
    """Kwargs for eval_lib.build_delayed_agent."""
    if self.agent_config.jit_compile:
      logging.warning('jit_compile leads to instability')
    return dict(
        state=self.get_state(),
        name=self.agent_config.name,
        compile=self.agent_config.compile,
        jit_compile=self.agent_config.jit_compile,
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
        for port, learner in self._learners.items()}

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
    trajectory, _ = self._rollout()
    for port, learner in self._learners.items():
      _, self._hidden_states[port] = learner.compiled_unroll(
          trajectory[port], self._hidden_states[port])

  def _burnin_after_reset(self):
    for _ in range(self._burnin_steps_after_reset):
      self.unroll()

  def step(self, ppo_steps: int = None) -> tuple[list[evaluators.Trajectory], dict]:
    reset_interval = self._config.runtime.reset_every_n_steps
    if reset_interval and self.num_rollouts >= reset_interval:
      logging.info("Resetting environments")
      with self.reset_profiler:
        self.actor.reset_env()
        self.num_rollouts = 0
        self._burnin_after_reset()

    with self.update_profiler:
      variables = {
          port: agent.learner.policy_variables()
          for port, agent in self._agents.items()
      }
      self.actor.update_variables(variables)

    with self.rollout_profiler:
      trajectories = {port: [] for port in PORTS}
      actor_metrics = []
      for _ in range(self._num_ppo_batches):
        trajectory, timings = self._rollout()
        for port in PORTS:
          trajectories[port].append(trajectory[port])
        actor_metrics.append(timings)

      actor_metrics = tf.nest.map_structure(
          lambda *xs: np.mean(xs), *actor_metrics)

    with self.learner_profiler:
      metrics = {}

      for port, learner in self._learners.items():
        self._hidden_states[port], metrics[port] = learner.ppo(
            trajectories[port], self._hidden_states[port], num_epochs=ppo_steps)

    return trajectories, dict(learner=metrics, actor=actor_metrics)

def run(config: Config):
  tag = config.runtime.tag or train_lib.get_experiment_tag()
  # Might want to use wandb.run.dir instead, but it doesn't seem
  # to be set properly even when we try to override it.
  expt_dir = config.runtime.expt_dir
  if expt_dir is None:
    expt_dir = os.path.join(config.runtime.expt_root, tag)
    os.makedirs(expt_dir, exist_ok=True)
    config.runtime.expt_dir = expt_dir
  logging.info('experiment directory: %s', expt_dir)

  agents: dict[int, AgentManager] = {}
  for port in PORTS:
    agents[port] = AgentManager(
        agent_config=getattr(config, f'p{port}'),
        port=port,
        expt_dir=expt_dir,
        learner_config=getattr(config, f'learner{port}'),
    )

  # Purely for naming the save files.
  for port, agent in agents.items():
    agent.set_opponent(agents[ENEMY_PORTS[port]].character)

  agent_kwargs = {
      port: agent.agent_kwargs()
      for port, agent in agents.items()
  }

  dolphin_kwargs = dict(
      players={
          port: dolphin_lib.AI(character=agent.character)
          for port, agent in agents.items()
      },
      **config.dolphin.to_kwargs(),
  )

  # Allow testing with one env; swap_ports generally needs an even number.
  env_kwargs = dict(swap_ports=config.actor.num_envs > 1)
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
      # Rewards are overridden in the learner.
  )

  experiment_manager = ExperimentManager(
      config=config,
      agents=agents,
      build_actor=build_actor,
  )

  step_profiler = utils.Profiler()

  def get_log_data(
      all_trajectories: dict[int, list[evaluators.Trajectory]],
      metrics: dict,
  ) -> dict:
    main_port, other_port = PORTS
    trajectories = all_trajectories[main_port]
    timings = {}

    # TODO: we shouldn't take the mean over these timings
    step_time = step_profiler.mean_time()
    steps_per_rollout = config.actor.num_envs * config.actor.rollout_length
    fps = len(trajectories) * steps_per_rollout / step_time
    mps = fps / (60 * 60)  # in-game minutes per second

    timings.update(
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
    for key in ['agent_pop', 'agent_step']:
      timings[key] = actor_timing[key][main_port]

    # concatenate along the batch dimension
    states: embed.Game = tf.nest.map_structure(
        lambda *xs: np.concatenate(xs, axis=1),
        *[t.states for t in trajectories])

    p0_stats = reward.player_stats(states.p0, states.p1, states.stage)
    p1_stats = reward.player_stats(states.p1, states.p0, states.stage)
    ko_diff = p1_stats['deaths'] - p0_stats['deaths']

    return dict(
        p0=p0_stats,
        p1=p1_stats,
        ko_diff=ko_diff,
        timings=timings,
        actor=metrics['actor'],
        learner=metrics['learner'][main_port],
        learner2=metrics['learner'][other_port],
    )

  logger = Logger()

  def flush(step: int):
    metrics = logger.flush(step * steps_per_epoch)
    if metrics is None:
      return

    print('\nStep:', step)

    timings: dict = metrics['timings']
    timing_str = ', '.join(
        ['{k}: {v:.3f}'.format(k=k, v=v) for k, v in timings.items()])
    print(timing_str)

    ko_diff = metrics['ko_diff']
    print(f'KO_diff_per_minute: {ko_diff:.3f}')

    learner_metrics = metrics['learner']
    pre_update = learner_metrics['ppo_step']['0']
    mean_actor_kl = pre_update['actor_kl']['mean']
    max_actor_kl = pre_update['actor_kl']['max']
    print(f'actor_kl: mean={mean_actor_kl:.3g} max={max_actor_kl:.3g}')
    teacher_kl = pre_update['teacher_kl']
    print(f'teacher_kl: {teacher_kl:.3g}')
    print(f'uev: {learner_metrics["value"]["uev"]:.3f}')

  maybe_flush = utils.Periodically(flush, config.runtime.log_interval)

  # The OpponentType enum sadly needs to be converted.
  rl_config_jsonnable = dataclasses.asdict(config)
  rl_config_jsonnable = tf.nest.map_structure(
      lambda x: x.value if isinstance(x, enum.Enum) else x,
      rl_config_jsonnable
  )

  def save(step: int):
    for agent in agents.values():
      agent.save(step)

  maybe_save = utils.Periodically(save, config.runtime.save_interval)

  try:
    steps_per_epoch = config.learner.ppo.num_batches

    agent0 = agents[PORTS[0]]
    step = agent0.step  # Will be the same for both agents

    # We only save after burnin, so if the save file exists that means
    # we've already done burning in and don't need to do it again.
    if not agent0.found:
      logging.info('Optimizer burnin')

      for agent in agents.values():
        agent.learning_rate.assign(0)
      for _ in range(config.optimizer_burnin_steps // steps_per_epoch):
        with step_profiler:
          experiment_manager.step(ppo_steps=1)

      logging.info('Value function burnin')

      for agent in agents.values():
        agent.learning_rate.assign(agent.learner_config.learning_rate)

      for _ in range(config.value_burnin_steps // steps_per_epoch):
        with step_profiler:
          trajectories, metrics = experiment_manager.step(ppo_steps=0)

        if experiment_manager.learner_profiler.num_calls > 0:
          logger.record(get_log_data(trajectories, metrics))
          maybe_flush(step)

        step += 1

      # Need flush here because logging structure changes based on ppo_steps.
      flush(step)

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
