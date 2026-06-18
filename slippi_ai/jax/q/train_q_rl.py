"""Online RL training loop that trains a Q-policy by argmax over a Q-function.

This is the single-player analog of `nash/train_nash_rl.py`. It mirrors the
single-player RL loop in `slippi_ai/jax/rl/run_lib.py` (self-play or a fixed
CPU/OTHER opponent), but trains a Q-policy via argmax over an online-trained
Q-function instead of running PPO.
"""

import contextlib
import dataclasses
import itertools
import os
import pickle
import typing as tp

import melee
import numpy as np
import jax
from absl import logging
from flax import nnx

from slippi_ai import (
  dolphin as dolphin_lib,
  evaluators,
  flag_utils,
  reward as reward_lib,
  utils,
)
from slippi_ai.types import Action
from slippi_ai.jax.networks import RecurrentState

from slippi_ai.jax import (
  jax_utils,
  saving as jax_saving,
  train_lib,
)
from slippi_ai.jax.q import (
  rl_learner,
  train_q_fn,
)
from slippi_ai.jax.rl import run_lib
from slippi_ai.jax.rl.learner import FrameSkipTrajectory

Rank2 = tuple[int, int]

field = lambda f: dataclasses.field(default_factory=f)


@dataclasses.dataclass
class RuntimeConfig:
  expt_root: str = 'experiments/jax/q_rl'
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
  reward: reward_lib.RewardConfig = field(reward_lib.RewardConfig.default)

  # Exactly one of teacher or restore must be set.
  teacher: tp.Optional[str] = None
  restore: tp.Optional[str] = None

  # Required: path to a pre-trained (single-player) Q-function checkpoint.
  q_function: tp.Optional[str] = None

  remat: bool = True
  override_delay: tp.Optional[int] = None


DEFAULT_CONFIG = Config()
DEFAULT_CONFIG.dolphin.console_timeout = 30

cast_floats = jax_utils.jit(
  jax_utils.cast_floats_to_dtype, static_argnames='dtype')


class LearnerManager(tp.Generic[Action]):
  """Mirrors rl/run_lib.LearnerManager, but drives the Q-policy learner."""

  def __init__(
    self,
    learner: rl_learner.Learner[Action],
    config: Config,
    actor: evaluators.AbstractRolloutWorker,
    port: int,
    enemy_port: int,
    exit_stack: contextlib.ExitStack,
  ):
    self._config = config
    self._learner = learner
    self._unroll_length = config.actor.rollout_length
    self._port = port
    self._enemy_port = enemy_port
    self._burnin = config.runtime.burnin_steps_after_reset

    self.batch_size = config.actor.num_envs
    if config.opponent.should_train():
      self.batch_size *= 2
    self._hidden_state = learner.initial_state(self.batch_size, nnx.Rngs(0))

    self.update_profiler = utils.Profiler(burnin=0)
    self.learner_profiler = utils.Profiler()
    self.rollout_profiler = utils.Profiler()
    self.reset_profiler = utils.Profiler(burnin=0)

    self.frame_skip = learner.policy.frame_skip

    self._prev_actions = [
        learner.policy.controller_head.dummy_sample_outputs([self.batch_size])
    ] * (self.frame_skip - 1)
    self._prev_is_resetting = np.full([self.frame_skip - 1, self.batch_size], False)

    self.actor = actor
    self.actor.start()
    exit_stack.callback(self.actor.stop)
    for _ in range(self._burnin):
      self._burnin_step()

  def _rollout(self):
    trajectories, timings = self.actor.rollout(self._unroll_length)

    if self._config.opponent.should_train() and len(trajectories) == 2:
      ports = [self._port, self._enemy_port]
      trajectory = evaluators.Trajectory.batch(
          [trajectories[p] for p in ports])
    else:
      trajectory = trajectories[self._port]

    trajectory: evaluators.Trajectory[Rank2, Action, RecurrentState]

    assert not trajectory.delayed_actions, 'Not implemented'

    # Previous actions for time steps [-FS+1, -1]
    prev_actions = utils.map_nt(lambda x: x[np.newaxis], self._prev_actions)

    # Create full action sequence for time steps [-FS+1, U]
    actions = utils.map_nt(
        lambda *xs: np.concatenate(xs, axis=0),
        *prev_actions,
        trajectory.actions,
    )
    # Split into frame-skipped action lists for time steps [0, U / FS]
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
      (-1, self.frame_skip, self.batch_size)).any(axis=1)

    rewards = reward_lib.compute_rewards(
        trajectory.states,
        **dataclasses.asdict(self._config.reward))
    rewards = rewards.reshape((-1, self.frame_skip, self.batch_size)).sum(axis=1)

    fs_trajectory = FrameSkipTrajectory(
        states=state,
        name=trajectory.name[::self.frame_skip],
        actions=actions,
        rewards=rewards,
        is_resetting=is_resetting,
        initial_state=trajectory.initial_state,
        delayed_actions=trajectory.delayed_actions,
    )

    # Remove unsupported metrics from sim env
    timings.pop('completed_games', None)

    return fs_trajectory, trajectory, timings

  def _burnin_step(self):
    trajectory, _, _ = self._rollout()
    _, self._hidden_state = self._learner.step(
      trajectory, self._hidden_state, train=False, step=0)

  def reset_env(self):
    with self.reset_profiler:
      self.actor.reset_env()
      for _ in range(self._burnin):
        self._burnin_step()

  def step(self, step: int):
    with self.update_profiler:
      variables = {self._port: self._learner.policy_variables(to_numpy=False)}
      if self._config.opponent.should_update(step):
        variables[self._enemy_port] = self._learner.policy_variables(to_numpy=False)
      self.actor.update_variables(variables)

    with self.rollout_profiler:
      fs_trajectory, trajectory, actor_metrics = self._rollout()

    with self.learner_profiler:
      metrics, self._hidden_state = self._learner.step(
        fs_trajectory, self._hidden_state, train=True, step=step)

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

  # All "_state" objects only contain numpy arrays.
  jax_state: dict = {}

  if restore_path:
    rl_state = jax_saving.load_state_from_disk(restore_path)
    jax_state = rl_state['state']
    policy_config = rl_state['config']

    previous_config = flag_utils.dataclass_from_dict(Config, rl_state['rl_config'])
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
    jax_state['teacher'] = jax_state['q_policy'] = jax_state['policy']
    step = 0
  else:
    raise ValueError('Must pass exactly one of "teacher" and "restore".')

  name_map = rl_state['name_map']
  del teacher_state

  if config.override_delay is not None:
    policy_config['policy']['delay'] = config.override_delay

  policy_config['network']['tx_like']['remat'] = config.remat
  policy_config['controller_head']['autoregressive']['remat'] = config.remat

  pretraining_config = flag_utils.dataclass_from_dict(train_lib.Config, policy_config)

  if config.q_function is None:
    raise ValueError('Must provide a Q-function checkpoint via `q_function`.')

  with open(config.q_function, 'rb') as f:
    q_fn_state = pickle.load(f)

  # Note: the original Q-function must exist on disk even when restoring, just
  # for its config.
  q_fn_config = flag_utils.dataclass_from_dict(
    train_q_fn.Config, q_fn_state['config'])
  if not restore_path:
    for key in ['q_function', 'q_function_optimizer']:
      jax_state[key] = q_fn_state['state'][key]
  del q_fn_state

  frame_skip = pretraining_config.policy.frame_skip

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

  learner = rl_learner.Learner(
      config=config.learner,
      q_function_config=q_fn_config.q_function,
      policy_config=policy_config,
      rngs=nnx.Rngs(0),
      state=jax_state,
  )
  del jax_state

  PORT = 1
  ENEMY_PORT = 2

  batch_size = config.actor.num_envs
  config.agent.check_allowed_chars(rl_state)

  agent_state = rl_state.copy()
  # Save memory by only keeping the policy params for the actor.
  agent_state['state'] = {'policy': agent_state['state']['policy']}
  del rl_state

  main_agent_kwargs = config.agent.get_kwargs()
  main_agent_kwargs['state'] = agent_state
  agent_kwargs = {PORT: main_agent_kwargs}
  del agent_state

  main_players = [dolphin_lib.AI() for _ in range(batch_size)]
  if config.opponent.type == run_lib.OpponentType.CPU:
    opponent_players = [dolphin_lib.CPU() for _ in range(batch_size)]
  else:
    opponent_players = [dolphin_lib.AI() for _ in range(batch_size)]

  if config.opponent.type == run_lib.OpponentType.CPU:
    names = itertools.islice(itertools.cycle(config.agent.name), batch_size)
    main_agent_kwargs['name'] = list(names)
    if config.agent.char is not None:
      chars = itertools.islice(itertools.cycle(config.agent.char), batch_size)
      for char, player in zip(chars, main_players):
        player.character = char
  else:
    if config.opponent.type == run_lib.OpponentType.SELF:
      opponent_kwargs = main_agent_kwargs.copy()
      opponent_names = config.agent.name
      opponent_chars = config.agent.char
    elif config.opponent.type == run_lib.OpponentType.OTHER:
      opponent_kwargs = config.opponent.other.get_kwargs()
      opponent_names = config.opponent.other.name
      opponent_chars = config.opponent.other.char
    else:
      raise ValueError(f'Unknown opponent type: {config.opponent.type}')

    name_combinations = list(itertools.product(config.agent.name, opponent_names))
    name_combination_batch = list(itertools.islice(
        itertools.cycle(name_combinations), batch_size))
    main_agent_names, opp_names = zip(*name_combination_batch)
    main_agent_kwargs['name'] = list(main_agent_names)
    opponent_kwargs['name'] = list(opp_names)
    agent_kwargs[ENEMY_PORT] = opponent_kwargs
    del opponent_kwargs

    main_chars = [None] if config.agent.char is None else config.agent.char
    opp_chars_list = [None] if opponent_chars is None else opponent_chars
    char_combinations = list(itertools.product(main_chars, opp_chars_list))
    char_combination_batch = list(itertools.islice(
        itertools.cycle(char_combinations), batch_size))
    main_agent_chars, opp_agent_chars = zip(*char_combination_batch)
    for player, char in zip(main_players, main_agent_chars):
      if char is not None:
        player.character = char
    for player, char in zip(opponent_players, opp_agent_chars):
      if char is not None:
        player.character = char

  del main_agent_kwargs

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
    if config.opponent.type == run_lib.OpponentType.CPU:
      raise ValueError('Sim env only supports AI-vs-AI opponents.')
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

    rollout_agent_kwargs: dict[int | tuple[int, ...], dict]
    if config.opponent.should_train():
      merged_kwargs = agent_kwargs[PORT].copy()
      merged_kwargs['name'] = main_agent_names + opp_names  # type: ignore
      rollout_agent_kwargs = {(PORT, ENEMY_PORT): merged_kwargs}
    else:
      rollout_agent_kwargs = {k: v for k, v in agent_kwargs.items()}

    actor = jax_rollout.JaxSimRolloutWorker(
        agent_kwargs=rollout_agent_kwargs,
        dolphin_kwargs=dolphin_kwargs,
        num_envs=config.actor.num_envs,
        rollout_length=config.actor.rollout_length,
        use_fake_envs=config.actor.use_fake_envs,
        async_envs=config.actor.async_envs,
        inner_batch_size=config.actor.inner_batch_size,
        copy_data=False,
        keep_agent_outputs_on_device=False,
    )
    del rollout_agent_kwargs
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
  del agent_kwargs

  learner_manager = LearnerManager(
      config=config,
      learner=learner,
      port=PORT,
      enemy_port=ENEMY_PORT,
      actor=actor,
      exit_stack=exit_stack,
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
  frames_per_step = config.actor.num_envs * config.actor.rollout_length
  if config.opponent.should_train():
    frames_per_step *= 2

  def get_log_data(
      trajectory: evaluators.Trajectory[Rank2, Action, RecurrentState],
      metrics: dict,
  ) -> dict:
    step_time = step_profiler.mean_time()
    fps = frames_per_step / step_time
    mps = fps / (60 * 60)

    timings = dict(
      rollout=learner_manager.rollout_profiler.mean_time(),
      learner=learner_manager.learner_profiler.mean_time(),
      # reset=learner_manager.reset_profiler.mean_time(),
      total=step_time,
      fps=fps,
      mps=mps,
      sps=1 / step_time,
    )
    timings['actor'] = utils.map_nt(
        lambda x: x * 1000, metrics['actor'].pop('timing'))

    states = trajectory.states

    p0_stats = reward_lib.player_stats(
        states.p0, states.p1, states.stage,
        stalling_threshold=config.reward.stalling_threshold)

    if not config.opponent.should_train():
      metrics['ko_diff'] = reward_lib.ko_diff(states) * MINUTES_PER_FRAME

    metrics.update(
        timings=timings,
        p0=p0_stats,
    )
    return metrics

  frames_per_step = config.actor.num_envs * config.actor.rollout_length
  if config.opponent.should_train():
    frames_per_step *= 2

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
    timings = utils.map_nt(lambda v: f'{v:.2f}', timings)
    print(timings)

    q_policy_metrics = metrics['learner'][rl_learner.Q_POLICY]
    q_loss = np.mean(q_policy_metrics['q_loss'])
    actor_kl = np.mean(q_policy_metrics['post_update_actor_kl'])
    entropy = np.mean(q_policy_metrics['entropy'])
    print(f'q_loss={q_loss:.4f} actor_kl={actor_kl:.3g} entropy={entropy:.3f}')
    if not config.opponent.should_train():
      print(f'ko_diff: {metrics["ko_diff"]:.3f}')

  maybe_flush = utils.Periodically(flush, config.runtime.log_interval)

  reset_interval = config.runtime.reset_every_n_steps
  if not config.dolphin.infinite_time:
    logging.info('Finite time mode, disabling env resets')
    reset_interval = None
  elif config.actor.use_sim_envs:
    logging.info('Sim envs, disabling env resets')
    reset_interval = None

  for i in range(config.runtime.max_step - step):
    with step_profiler:
      if i > 0 and reset_interval and i % reset_interval == 0:
        logging.info('Resetting environments')
        learner_manager.reset_env()

      trajectory, metrics = learner_manager.step(step=step)

    logger.record(get_log_data(trajectory, metrics))
    maybe_flush(step)
    step += 1
    maybe_save(step)

  save(step)
