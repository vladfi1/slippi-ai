"""RL training loop using Nash policy learning instead of PPO."""

import contextlib
import dataclasses
import itertools
import os
import pickle
import typing as tp

import jax
import numpy as np
from absl import logging
from flax import nnx
from pandas import isna

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

  profile_trace_dir: tp.Optional[str] = None


@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = field(RuntimeConfig)

  dolphin: dolphin_lib.DolphinConfig = field(dolphin_lib.DolphinConfig)
  learner: rl_learner.LearnerConfig = field(rl_learner.LearnerConfig)
  actor: run_lib.ActorConfig = field(run_lib.ActorConfig)
  agent: run_lib.AgentConfig = field(run_lib.AgentConfig)
  reward: reward_lib.RewardConfig = field(reward_lib.RewardConfig)

  # Exactly one of teacher or restore must be set.
  teacher: tp.Optional[str] = None
  restore: tp.Optional[str] = None

  # Required: path to a pre-trained Q-function checkpoint.
  q_function: tp.Optional[str] = None

  remat: bool = True

  override_delay: tp.Optional[int] = None


DEFAULT_CONFIG = Config()
DEFAULT_CONFIG.dolphin.console_timeout = 30


def stack_trajectories(
    trajectories: list[Trajectory[Rank2, Action, RecurrentState]],  # N x [T, B]
) -> Trajectory[Rank3, Action, RecurrentState]:  # [T, B] -> [T, B, N]
  return utils.map_nt(
      lambda axis, *ts: utils.batch_nest_nt(ts, axis=axis+1),
      Trajectory.batch_dims(), *trajectories)

def unmerge_trajectory(
    trajectory: Trajectory[Rank2, Action, RecurrentState],
) -> Trajectory[Rank3, Action, RecurrentState]:
  # Converts [..., 2B] -> [..., B, 2]
  def unmerge_array(axis: int, x: np.ndarray) -> np.ndarray:
    x = x.reshape(x.shape[:axis] + (2, x.shape[axis] // 2) + x.shape[axis+1:])
    return np.swapaxes(x, axis, axis+1)

  return jax.tree.map(
    lambda axis, substruct: jax.tree.map(
      lambda x: unmerge_array(axis, x), substruct),
      Trajectory.batch_dims(), trajectory)


class LearnerManager(tp.Generic[Action]):

  def __init__(
    self,
    learner: rl_learner.Learner[Action],
    config: Config,
    actor: evaluators.AbstractRolloutWorker,
    ports: tuple[int, int],
  ):
    self._config = config
    self._learner = learner
    self.actor = actor
    self._rollout_length = config.actor.rollout_length
    self._ports = ports
    self._burnin = config.runtime.burnin_steps_after_reset

    self.batch_size = config.actor.num_envs
    self._hidden_state = learner.initial_state(self.batch_size, nnx.Rngs(0))

    self.update_profiler = utils.Profiler(burnin=0)
    self.learner_profiler = utils.Profiler()
    self.rollout_profiler = utils.Profiler()
    self.reset_profiler = utils.Profiler(burnin=0)

    self.frame_skip = learner.policy.frame_skip

    self._prev_actions = [
        learner.policy.controller_head.dummy_sample_outputs([self.batch_size, 2])
    ] * (self.frame_skip - 1)
    self._prev_is_resetting = np.full([self.frame_skip - 1, self.batch_size, 2], False)

    for _ in range(self._burnin):
      self._burnin_step()

  def _rollout(self):
    # TODO: might be better to do trajectory manipulation in jax, as parts of
    # the data are already on device (e.g. actions and initial state)

    trajectories, metrics = self.actor.rollout(self._rollout_length)
    if len(trajectories) == 1:
      assert trajectories[self._ports[0]].is_resetting.shape[1] == 2 * self.batch_size
      trajectory = unmerge_trajectory(trajectories[self._ports[0]])
    elif len(trajectories) == 2:
      trajectory: Trajectory[Rank3, Action, RecurrentState] = stack_trajectories(
          [trajectories[p] for p in self._ports])
    else:
      raise ValueError(f'Expected 1 or 2 trajectories, got {len(trajectories)}')

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
    # Note: actions are not owned by the rollout worker so we don't need to
    # make a copy here.
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
    rewards = rewards.reshape((-1, self.frame_skip, self.batch_size, 2))
    discounts = self._learner.discount ** np.arange(self.frame_skip)

    rewards = np.sum(rewards * np.expand_dims(discounts, axis=[1, 2]), axis=1)

    fs_trajectory = FrameSkipTrajectory(
        states=state,
        name=trajectory.name[::self.frame_skip],
        rating=trajectory.rating[::self.frame_skip],
        actions=actions,
        rewards=rewards,
        is_resetting=is_resetting,
        initial_state=trajectory.initial_state,
        delayed_actions=trajectory.delayed_actions,
    )

    # Remove unsupported metrics from sim env
    metrics.pop('completed_games', None)

    return fs_trajectory, metrics


  def _burnin_step(self):
    trajectory, _ = self._rollout()
    _, self._hidden_state = self._learner.step(
      trajectory, self._hidden_state, train=False, step=0)

  def reset_env(self):
    with self.reset_profiler:
      self.actor.reset_env()
      for _ in range(self._burnin):
        self._burnin_step()

  def step(self, step: int):
    with self.update_profiler:
      policy_variables = self._learner.policy_variables()
      self.actor.update_variables(
        {p: policy_variables for p in self._ports})

    with self.rollout_profiler:
      trajectory, actor_metrics = self._rollout()

    with self.learner_profiler:
      metrics, self._hidden_state = self._learner.step(
        trajectory, self._hidden_state, train=True, step=step)

    return trajectory, dict(learner=metrics, actor=actor_metrics)


def run(config: Config):
  with contextlib.ExitStack() as exit_stack:
    _run(config, exit_stack)


def _run(config: Config, exit_stack: contextlib.ExitStack):
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

  # All "_state" objects only have numpy arrays.
  jax_state = {}

  if restore_path:
    rl_state = jax_saving.load_state_from_disk(restore_path)
    jax_state = rl_state['state']
    policy_config = rl_state['config']

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
    policy_config = teacher_state['config']
    run_lib.reset_optimizer_steps(teacher_state)
    rl_state = teacher_state
    for key in ['policy', 'policy_optimizer']:
      jax_state[key] = teacher_state['state'][key]
    jax_state['teacher'] = jax_state['nash_policy'] = jax_state['policy']
    step = 0
  else:
    raise ValueError('Must pass exactly one of "teacher" and "restore".')

  name_map = rl_state['name_map']

  # TODO: handling all of these "state" objects that may take up a lot of
  # memory is error prone; we should find a cleaner way to do this.
  del teacher_state

  if config.override_delay is not None:
    policy_config['policy']['delay'] = config.override_delay

  policy_config['network']['tx_like']['remat'] = config.remat
  policy_config['controller_head']['autoregressive']['remat'] = config.remat

  pretraining_config = flag_utils.dataclass_from_dict(
      train_lib.Config, policy_config)

  if config.q_function is None:
    raise ValueError('Must provide a Q-function checkpoint via `q_function`.')

  with open(config.q_function, 'rb') as f:
    q_fn_state = pickle.load(f)

  # Note: the original Q-function must exist on-disk even when restoring from a
  # checkpoint just for the config.
  # TODO: just save the Q-function config in the checkpoint to avoid this.
  # TODO: support training a Q-function from scratch?
  q_fn_config = flag_utils.dataclass_from_dict(
    train_q_fn.Config, q_fn_state['config'])
  if not restore_path:
    for key in ['q_function', 'q_function_optimizer']:
      jax_state[key] = q_fn_state['state'][key]
  del q_fn_state

  frame_skip = pretraining_config.policy.frame_skip

  # TODO: move these into a helper function
  if q_fn_config.observation != pretraining_config.observation:
    raise ValueError(
      'Q-function observation config does not match policy: '
      f'{q_fn_config.observation} vs {pretraining_config.observation}')
  if q_fn_config.delay != pretraining_config.policy.delay:
    raise ValueError(
      'Q-function delay does not match policy delay: '
      f'{q_fn_config.delay} vs {pretraining_config.policy.delay}')
  if q_fn_config.q_function.frame_skip != frame_skip:
    raise ValueError(
      'Q-function frame skip does not match policy frame skip: '
      f'{q_fn_config.q_function.frame_skip} vs {frame_skip}')

  if config.actor.rollout_length % frame_skip != 0:
    raise ValueError(
      'Rollout length must be divisible by frame skip: '
      f'{config.actor.rollout_length} vs {frame_skip}')

  # mesh = jax_utils.get_mesh()

  learner = rl_learner.Learner(
      config=config.learner,
      q_function_config=q_fn_config.q_function,
      policy_config=policy_config,
      rngs=nnx.Rngs(0),
      state=jax_state,
      # mesh=mesh,
      # frame_skip=frame_skip,
  )
  del jax_state

  PORT = 1
  ENEMY_PORT = 2

  batch_size = config.actor.num_envs
  config.agent.check_allowed_chars(rl_state)

  agent_state = rl_state.copy()
  agent_state_keys = ['policy']
  # Save memory by only keeping the keys we need.
  agent_state['state'] = {k: agent_state['state'][k] for k in agent_state_keys}
  del rl_state

  main_agent_kwargs = config.agent.get_kwargs()
  main_agent_kwargs['state'] = agent_state

  main_players = [dolphin_lib.AI() for _ in range(batch_size)]
  opponent_players = [dolphin_lib.AI() for _ in range(batch_size)]

  # TODO: should p1/p2 have different names?
  opponent_kwargs = main_agent_kwargs.copy()
  name_combinations = list(itertools.product(
    config.agent.name, config.agent.name))
  name_combination_batch = list(itertools.islice(
    itertools.cycle(name_combinations), batch_size))
  main_agent_names, opp_names = zip(*name_combination_batch)
  main_agent_kwargs['name'] = list(main_agent_names)
  opponent_kwargs['name'] = list(opp_names)
  agent_kwargs = {PORT: main_agent_kwargs, ENEMY_PORT: opponent_kwargs}

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

  if config.actor.use_sim_envs:
    if config.actor.async_envs:
      if config.actor.num_envs % config.actor.inner_batch_size:
        raise ValueError(
            f'num_envs={config.actor.num_envs} must be divisible by '
            f'inner_batch_size={config.actor.inner_batch_size} for sim RL.')
    if config.agent.batch_steps > pretraining_config.policy.delay:
      raise ValueError(
          f'agent.batch_steps={config.agent.batch_steps} exceeds policy delay '
          f'{pretraining_config.policy.delay} for sim RL.')
    if config.actor.rollout_length % max(1, config.agent.batch_steps):
      raise ValueError('agent.batch_steps must divide rollout_length for sim RL.')

    from slippi_ai.sim_env import jax_rollout

    merged_kwargs = main_agent_kwargs.copy()
    merged_kwargs['name'] = main_agent_names + opp_names
    rollout_agent_kwargs = {(PORT, ENEMY_PORT): merged_kwargs}

    actor = jax_rollout.JaxSimRolloutWorker(
        agent_kwargs=rollout_agent_kwargs,  # type: ignore
        dolphin_kwargs=dolphin_kwargs,
        num_envs=config.actor.num_envs,
        rollout_length=config.actor.rollout_length,
        use_fake_envs=config.actor.use_fake_envs,
        async_envs=config.actor.async_envs,
        inner_batch_size=config.actor.inner_batch_size,
        copy_data=False,
        # We need to manipulate agent outputs in python.
        keep_agent_outputs_on_device=False,
    )
  else:
    actor = evaluators.RolloutWorker(
      agent_kwargs=agent_kwargs,
      dolphin_kwargs=dolphin_kwargs,
      env_kwargs=env_kwargs,
      num_envs=config.actor.num_envs,
      async_envs=config.actor.async_envs,
      use_gpu=config.actor.gpu_inference,
      use_fake_envs=config.actor.use_fake_envs,
    )

  exit_stack.enter_context(actor.run())

  learner_manager = LearnerManager(
    config=config,
    learner=learner,
    ports=(PORT, ENEMY_PORT),
    actor=actor,
  )

  step_profiler = utils.Profiler()
  rl_config_dict = dataclasses.asdict(config)

  def save(step: int):
    if config.runtime.save_interval < 0:
      return

    combined_state = dict(
      state=jax_utils.get_module_state(learner),
      config=policy_config,
      name_map=name_map,
      step=step,
      rl_config=rl_config_dict,
    )
    pickled_state = pickle.dumps(combined_state)
    logging.info('saving state to %s', pickle_path)
    with open(pickle_path, 'wb') as f:
      f.write(pickled_state)

  maybe_save = utils.Periodically(save, config.runtime.save_interval)

  logger = run_lib.Logger()

  MINUTES_PER_FRAME = 60 * 60
  frames_per_step = 2 * config.actor.num_envs * config.actor.rollout_length

  def get_log_data(
    trajectory: FrameSkipTrajectory,
    metrics: dict,
  ) -> dict:
    step_time = step_profiler.mean_time()
    fps = frames_per_step / step_time
    mps = fps / MINUTES_PER_FRAME

    timings = dict(
      rollout=learner_manager.rollout_profiler.mean_time(),
      learner=learner_manager.learner_profiler.mean_time(),
      # reset=learner_manager.reset_profiler.mean_time(),
      total=step_time,
      fps=fps,
      mps=mps,
      sps=1 / step_time,
    )
    actor_timing = metrics['actor'].pop('timing')
    for key in ['env_pop', 'env_push']:
      timings[key] = 1000 * actor_timing[key]
    timings['agent_step'] = 1000 * actor_timing['agent_step'][PORT]

    metrics.update(timings=timings)
    return metrics

  def flush(step: int):
    total_frames = step * frames_per_step
    extras = dict(
        total_frames=total_frames,
    )

    metrics = logger.flush(step, extras=extras)
    if metrics is None:
      return

    print('\nStep:', step)
    timings: dict = metrics['timings']
    timing_str = ', '.join(f'{k}: {v:.2f}' for k, v in timings.items())
    print(timing_str)

    learner_metrics = metrics['learner']

    to_print = dict(
        ufrac = learner_metrics[rl_learner.NASH_POLICY]['unique_fraction'],
        ifrac = learner_metrics[rl_learner.Q_FUNCTION]['information_fraction'],
        ent = learner_metrics[rl_learner.NASH_POLICY]['entropy'],
        akl = learner_metrics[rl_learner.NASH_POLICY]['actor_kl'],
        tkl = learner_metrics[rl_learner.NASH_POLICY]['teacher_kl'],
        nent = learner_metrics[rl_learner.NASH_POLICY]['nash_entropy'],
        nxent = learner_metrics[rl_learner.NASH_POLICY]['nash_cross_entropy'],
    )
    to_print = utils.map_nt(train_lib.mean, to_print)

    items = []
    for key, value in to_print.items():
      items.append(f'{key}={value:.3f}')

    items.extend([
      f'nsmean={learner_metrics[rl_learner.NASH]['num_steps']:.1f}',
      f'nsmax={learner_metrics[rl_learner.NASH]['num_steps_max']:.1f}',
    ])
    print(' '.join(items))

  maybe_flush = utils.Periodically(flush, config.runtime.log_interval)

  reset_interval = config.runtime.reset_every_n_steps
  if not config.dolphin.infinite_time:
    logging.info('Finite time mode, disabling env resets')
    reset_interval = None

  if config.runtime.profile_trace_dir is not None:
    if config.runtime.max_step is None:
      raise ValueError('max_step must be set when profile_trace_dir is set to limit the trace size.')
    jax.profiler.start_trace(config.runtime.profile_trace_dir)

  try:
    for i in range(config.runtime.max_step - step):
      with step_profiler:
        if i > 0 and reset_interval and i % reset_interval == 0:
          logging.info('Resetting environments')
          learner_manager.reset_env()

        trajectory, metrics = learner_manager.step(step=step)

      # TODO: allow some sort of runahead?
      logger.record(get_log_data(trajectory, metrics))
      maybe_flush(step)
      maybe_save(step)
      step += 1

    save(step)

  finally:
    learner_manager.actor.stop()

    if config.runtime.profile_trace_dir is not None:
      jax.profiler.stop_trace()
