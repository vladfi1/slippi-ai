"""RL training loop using Nash policy learning instead of PPO."""

import dataclasses
import itertools
import os
import pickle
import typing as tp

import melee
import numpy as np
from absl import logging
from flax import nnx

from slippi_ai import (
  dolphin as dolphin_lib,
  evaluators,
  flag_utils,
  reward as reward_lib,
  utils,
)
from slippi_ai.evaluators import Trajectory, Rank2
from slippi_ai.types import Action, Game
from slippi_ai.jax.networks import RecurrentState

from slippi_ai.jax import (
  jax_utils,
  saving as jax_saving,
  train_lib,
)
from slippi_ai.jax.nash import (
  q_function as q_lib,
  rl_learner,
  train_q_fn,
)
from slippi_ai.jax.rl import run_lib
from slippi_ai.jax.rl.learner import FrameSkipTrajectory

Rank3 = tuple[int, int, int]

field = lambda f: dataclasses.field(default_factory=f)


@dataclasses.dataclass
class RuntimeConfig:
  expt_root: str = 'experiments/jax/nash_rl'
  expt_dir: tp.Optional[str] = None
  tag: tp.Optional[str] = None

  max_step: int = 10
  max_runtime: tp.Optional[int] = None
  log_interval: int = 10
  save_interval: int = 300

  reset_every_n_steps: tp.Optional[int] = None
  burnin_steps_after_reset: int = 0


@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = field(RuntimeConfig)

  dolphin: dolphin_lib.DolphinConfig = field(dolphin_lib.DolphinConfig)
  learner: rl_learner.LearnerConfig = field(rl_learner.LearnerConfig)
  actor: run_lib.ActorConfig = field(run_lib.ActorConfig)
  agent: run_lib.AgentConfig = field(run_lib.AgentConfig)
  opponent: run_lib.OpponentConfig = field(run_lib.OpponentConfig)
  reward: reward_lib.RewardConfig = field(reward_lib.RewardConfig)

  # Exactly one of teacher or restore must be set.
  teacher: tp.Optional[str] = None
  restore: tp.Optional[str] = None

  # Required: path to a pre-trained Q-function checkpoint.
  q_function: tp.Optional[str] = None

  override_delay: tp.Optional[int] = None


DEFAULT_CONFIG = Config()
DEFAULT_CONFIG.dolphin.console_timeout = 30


def stack_trajectories(
    trajectories: list[Trajectory[Rank2, Action, RecurrentState]],  # N x [T, B]
) -> Trajectory[Rank3, Action, RecurrentState]:  # [T, B] -> [T, B, N]
  return utils.map_nt(
      lambda axis, *ts: utils.batch_nest_nt(ts, axis=axis+1),
      Trajectory.batch_dims(), *trajectories)



class LearnerManager(tp.Generic[Action]):

  def __init__(
    self,
    learner: rl_learner.Learner[Action],
    config: Config,
    build_actor: tp.Callable[[], evaluators.RolloutWorker],
    ports: tuple[int, int],
  ):
    self._config = config
    self._learner = learner
    self._build_actor = build_actor
    self._rollout_length = config.actor.rollout_length
    self._ports = ports
    self._burnin = config.runtime.burnin_steps_after_reset

    self.batch_size = config.actor.num_envs
    self._hidden_state = learner.initial_state(self.batch_size, nnx.Rngs(0))

    self.update_profiler = utils.Profiler(burnin=0)
    self.learner_profiler = utils.Profiler()
    self.rollout_profiler = utils.Profiler()
    self.reset_profiler = utils.Profiler(burnin=0)

    self.frame_skip = learner.nash_policy.frame_skip

    self._prev_actions = [
        learner.nash_policy.controller_head.dummy_sample_outputs([self.batch_size, 2])
    ] * (self.frame_skip - 1)
    self._prev_is_resetting = np.full([self.frame_skip - 1, self.batch_size, 2], False)

    with self.reset_profiler:
      self.actor = self._build_actor()
      self.actor.start()
      for _ in range(self._burnin):
        self._burnin_step()


  def _rollout(self):
    trajectories, timings = self.actor.rollout(self._rollout_length)
    assert len(trajectories) == 2

    trajectory: Trajectory[Rank3, Action, RecurrentState] = stack_trajectories(
        [trajectories[p] for p in self._ports])

    assert not trajectory.delayed_actions, 'Not implemented'

    # Previous actions for time steps [-FS+1, -1]
    prev_actions = utils.map_nt(lambda x: x[np.newaxis], self._prev_actions)

    # Create full action sequence of length for time steps [-FS+1, U]
    actions = utils.map_nt(
        lambda *xs: np.concatenate(xs, axis=0),
        *prev_actions,
        trajectory.actions,
    )
    # Split into skipped (previous) actions for time steps [0, U / FS]
    actions = [
        utils.map_nt(lambda t: t[i::self.frame_skip], actions)
        for i in range(self.frame_skip)
    ]
    self._prev_actions = utils.map_nt(lambda x: x[-1], actions[:-1])

    state = utils.map_single_structure(
      lambda x: x[::self.frame_skip], trajectory.states)

    is_resetting = np.concatenate([
        self._prev_is_resetting,
        trajectory.is_resetting,
    ], axis=0)
    self._prev_is_resetting = is_resetting[-self.frame_skip:-1]
    is_resetting = is_resetting.reshape(
      (-1, self.frame_skip, self.batch_size, 2)).any(axis=1)

    rewards = reward_lib.compute_rewards(
        trajectory.states,
        **dataclasses.asdict(self._config.reward))
    rewards = rewards.reshape((-1, self.frame_skip, self.batch_size, 2)).sum(axis=1)

    fs_trajectory = FrameSkipTrajectory(
        states=state,
        name=trajectory.name[::self.frame_skip],
        actions=actions,
        rewards=rewards,
        is_resetting=is_resetting,
        initial_state=trajectory.initial_state,
        delayed_actions=trajectory.delayed_actions,
    )

    return fs_trajectory, timings


  def _burnin_step(self):
    trajectory, _ = self._rollout()
    _, self._hidden_state = self._learner.step(
      trajectory, self._hidden_state, train=False)

  def reset_env(self):
    with self.reset_profiler:
      self.actor.reset_env()
      for _ in range(self._burnin):
        self._burnin_step()

  def step(self):
    with self.update_profiler:
      policy_variables = self._learner.policy_variables()
      self.actor.update_variables(
        {p: policy_variables for p in self._ports})

    with self.rollout_profiler:
      trajectory, actor_metrics = self._rollout()

    with self.learner_profiler:
      metrics, self._hidden_state = self._learner.step(
        trajectory, self._hidden_state, train=True)

    return trajectory, dict(learner=metrics, actor=actor_metrics)


def run(config: Config):
  tag = config.runtime.tag or train_lib.get_experiment_tag()
  expt_dir = config.runtime.expt_dir
  if expt_dir is None:
    expt_dir = os.path.join(config.runtime.expt_root, tag)
    os.makedirs(expt_dir, exist_ok=True)
  logging.info('experiment directory: %s', expt_dir)

  if config.agent.path is not None:
    raise ValueError('Main agent path is not used; use `restore` instead.')
  if config.teacher and config.restore:
    raise ValueError('Must pass exactly one of "teacher" and "restore".')

  pickle_path = os.path.join(expt_dir, 'latest.pkl')
  restore_from_checkpoint = False
  restore_path = None

  if os.path.exists(pickle_path):
    logging.info('Restoring from checkpoint %s', pickle_path)
    restore_path = pickle_path
    restore_from_checkpoint = True
  elif config.restore:
    restore_path = config.restore

  if restore_path:
    rl_state = jax_saving.load_state_from_disk(restore_path)
    previous_config = flag_utils.dataclass_from_dict(
      Config, rl_state['rl_config'])
    previous_teacher = previous_config.teacher
    assert previous_teacher is not None

    if config.teacher and config.teacher != previous_teacher:
      assert restore_from_checkpoint
      raise ValueError(
        f'Requested teacher does not match checkpoint: '
        f'{config.teacher} != {previous_teacher}')

    config.teacher = previous_teacher
    logging.info('Using teacher: %s', config.teacher)
    teacher_state = jax_saving.load_state_from_disk(config.teacher)
    step = rl_state['step']
    logging.info('Restored at step %d', step)
  elif config.teacher:
    logging.info('Initializing from teacher: %s', config.teacher)
    teacher_state = jax_saving.load_state_from_disk(config.teacher)
    run_lib.reset_optimizer_steps(teacher_state)
    rl_state = teacher_state
    step = 0
  else:
    raise ValueError('Must pass exactly one of "teacher" and "restore".')

  if config.override_delay is not None:
    teacher_state['config']['policy']['delay'] = config.override_delay

  teacher = jax_saving.load_policy_from_state(teacher_state)
  policy = jax_saving.load_policy_from_state(rl_state)

  pretraining_config = flag_utils.dataclass_from_dict(
      train_lib.Config,
      jax_saving.upgrade_config(teacher_state['config']))

  if config.q_function is None:
    raise ValueError('Must provide a Q-function checkpoint via `q_function`.')

  with open(config.q_function, 'rb') as f:
    q_fn_state = pickle.load(f)

  q_fn_config = flag_utils.dataclass_from_dict(
    train_q_fn.Config, q_fn_state['config'])

  if q_fn_config.observation != pretraining_config.observation:
    raise ValueError(
      'Q-function observation config does not match policy: '
      f'{q_fn_config.observation} vs {pretraining_config.observation}')
  if q_fn_config.delay != pretraining_config.policy.delay:
    raise ValueError(
      'Q-function delay does not match policy delay: '
      f'{q_fn_config.delay} vs {pretraining_config.policy.delay}')

  q_function = q_lib.build_q_function(nnx.Rngs(0), q_fn_config.q_function)
  jax_utils.set_module_state(q_function, q_fn_state['state']['q_function'])

  # mesh = jax_utils.get_mesh()

  learner = rl_learner.Learner(
      config=config.learner,
      q_function=q_function,
      policy=policy,
      teacher=teacher,
      rngs=nnx.Rngs(0),
      # mesh=mesh,
      # frame_skip=frame_skip,
      nash_policy_optimizer_state=rl_state['state']['policy_optimizer'],
      q_function_optimizer_state=q_fn_state['state']['q_function_optimizer'],
  )

  if restore_path:
    jax_utils.set_module_state(learner, rl_state['state'])

  PORT = 1
  ENEMY_PORT = 2

  batch_size = config.actor.num_envs
  config.agent.check_allowed_chars(rl_state)

  main_agent_kwargs = config.agent.get_kwargs()
  main_agent_kwargs['state'] = rl_state

  main_players = [dolphin_lib.AI() for _ in range(batch_size)]
  if config.opponent.type == run_lib.OpponentType.CPU:
    opponent_players = [dolphin_lib.CPU() for _ in range(batch_size)]
  else:
    opponent_players = [dolphin_lib.AI() for _ in range(batch_size)]

  if config.opponent.type == run_lib.OpponentType.CPU:
    names = list(itertools.islice(itertools.cycle(config.agent.name), batch_size))
    main_agent_kwargs['name'] = names
    if config.agent.char is not None:
      chars = itertools.islice(itertools.cycle(config.agent.char), batch_size)
      for char, player in zip(chars, main_players):
        player.character = char
    agent_kwargs = {PORT: main_agent_kwargs}
  elif config.opponent.type == run_lib.OpponentType.SELF:
    opponent_kwargs = main_agent_kwargs.copy()
    name_combinations = list(itertools.product(
      config.agent.name, config.agent.name))
    name_combination_batch = list(itertools.islice(
      itertools.cycle(name_combinations), batch_size))
    main_agent_names, opp_names = zip(*name_combination_batch)
    main_agent_kwargs['name'] = list(main_agent_names)
    opponent_kwargs['name'] = list(opp_names)
    agent_kwargs = {PORT: main_agent_kwargs, ENEMY_PORT: opponent_kwargs}
  else:
    raise ValueError(f'Unknown opponent type: {config.opponent.type}')

  dolphin_kwargs = [
    dict(
      players={PORT: main_players[i], ENEMY_PORT: opponent_players[i]},
      **config.dolphin.to_kwargs(),
    ) for i in range(batch_size)
  ]

  env_kwargs: dict[str, tp.Any] = dict(swap_ports=False)
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

  learner_manager = LearnerManager(
    config=config,
    learner=learner,
    ports=(PORT, ENEMY_PORT),
    build_actor=build_actor,
  )

  step_profiler = utils.Profiler()
  rl_config_dict = dataclasses.asdict(config)

  def save(step: int):
    combined_state = dict(
      state=jax_utils.get_module_state(learner),
      config=teacher_state['config'],
      name_map=teacher_state['name_map'],
      step=step,
      rl_config=rl_config_dict,
    )
    pickled_state = pickle.dumps(combined_state)
    logging.info('saving state to %s', pickle_path)
    with open(pickle_path, 'wb') as f:
      f.write(pickled_state)

  maybe_save = utils.Periodically(save, config.runtime.save_interval)

  logger = run_lib.Logger()

  def get_log_data(
    trajectory: FrameSkipTrajectory,
    metrics: dict,
  ) -> dict:
    step_time = step_profiler.mean_time()
    fps = config.actor.num_envs * config.actor.rollout_length / step_time
    mps = fps / (60 * 60)

    timings = dict(
      rollout=learner_manager.rollout_profiler.mean_time(),
      learner=learner_manager.learner_profiler.mean_time(),
      reset=learner_manager.reset_profiler.mean_time(),
      total=step_time,
      fps=fps,
      mps=mps,
    )
    actor_timing = metrics['actor'].pop('timing')
    for key in ['env_pop', 'env_push']:
      timings[key] = actor_timing[key]
    timings['agent_step'] = actor_timing['agent_step'][PORT]

    metrics.update(timings=timings)
    return metrics

  def flush(step: int):
    metrics = logger.flush(step)
    if metrics is None:
      return

    print('\nStep:', step)
    timings: dict = metrics['timings']
    timing_str = ', '.join(f'{k}: {v:.3f}' for k, v in timings.items())
    print(timing_str)

    learner_metrics = metrics['learner']
    nash_xent = np.mean(
      learner_metrics[rl_learner.NASH_POLICY]['nash_cross_entropy'])
    nash_ent = np.mean(
      learner_metrics[rl_learner.NASH_POLICY]['nash_entropy'])
    print(f'nash_xent={nash_xent:.3f} nash_ent={nash_ent:.3f}')

  maybe_flush = utils.Periodically(flush, config.runtime.log_interval)

  reset_interval = config.runtime.reset_every_n_steps
  if not config.dolphin.infinite_time:
    logging.info('Finite time mode, disabling env resets')
    reset_interval = None

  try:
    for i in range(config.runtime.max_step):
      with step_profiler:
        if i > 0 and reset_interval and i % reset_interval == 0:
          logging.info('Resetting environments')
          learner_manager.reset_env()

        trajectory, metrics = learner_manager.step()

      logger.record(get_log_data(trajectory, metrics))
      maybe_flush(step)
      maybe_save(step)
      step += 1

    save(step)

  finally:
    learner_manager.actor.stop()
