import dataclasses
import logging
import os
import pickle
import typing as tp

import numpy as np
import tensorflow as tf
import wandb

from slippi_ai import (
    dolphin as dolphin_lib,
    evaluators,
    flag_utils,
    reward,
    saving,
    tf_utils,
    train_lib,
    utils,
)

from slippi_ai.types import Game
from slippi_ai import value_function as vf_lib
from slippi_ai.rl import mixture_learner as learner_lib
from slippi_ai.rl import run_lib

Port = evaluators.Port

field = run_lib.field

@dataclasses.dataclass
class Config(run_lib.Config):
  learner: learner_lib.LearnerConfig = field(learner_lib.LearnerConfig)
  opponent: bool = False  # unused

  # Number of steps to train the exploiter agent against the old mixture policy.
  # After this many steps, we update the opponent to the new mixture policy.
  # Note: unlike the env/optimizer/value function burnin steps, this measures
  # ppo epochs, so is effectively multiplied by the number of ppo batches.
  exploiter_train_steps: int = 2
  # After updating the opponent to the new mixture policy, we can also reset the
  # exploiter policy to the same new mixture policy. This reduces correlation
  # between exploiter training phases, but we may prefer to start from the old
  # exploiter policy as it is likely still good against the new mixture policy.
  reset_exploiter: bool = True

  # Set the exploiter mixture weight to 1 / (num_phases + 1); in principle this
  # mixes uniformly across all previous exploiter training phases. The
  # alternative is to use a fixed exploiter weight which is results in an
  # exponential moving average over the exploiter policies.
  scale_exploiter_weight: bool = False

DEFAULT_CONFIG = Config()
DEFAULT_CONFIG.dolphin.console_timeout = 30
DEFAULT_CONFIG.runtime.expt_root = 'experiments/mixture'


class ExperimentManager:

  def __init__(
      self,
      learner: learner_lib.Learner,
      config: Config,
      build_actor: tp.Callable[[], evaluators.RolloutWorker],
      exploiter_port: int,
      mixture_port: int,
      step: int,
      pickle_path: str,
      teacher_state: dict,
  ):
    self._config = config
    self._learner = learner
    self._build_actor = build_actor
    self._unroll_length = config.actor.rollout_length
    self._exploiter_port = exploiter_port
    self._mixture_port = mixture_port
    self._num_ppo_batches = config.learner.ppo.num_batches
    self._step = step
    self._pickle_path = pickle_path
    self._teacher_state = teacher_state

    self._maybe_save = utils.Periodically(self.save, config.runtime.save_interval)

    self._logger = run_lib.Logger()
    self._maybe_flush = utils.Periodically(self.flush, config.runtime.log_interval)

    if 2 * config.runtime.burnin_steps_after_reset > config.runtime.reset_every_n_steps:
      raise UserWarning('Spending more than half of the time in env burnin.')

    batch_size = config.actor.num_envs
    self._hidden_state = learner.initial_state(batch_size)

    self.step_profiler = utils.Profiler()
    self.update_profiler = utils.Profiler(burnin=0)
    self.learner_profiler = utils.Profiler()
    self.rollout_profiler = utils.Profiler()
    self.reset_profiler = utils.Profiler(burnin=0)

    with self.reset_profiler:
      self.actor = self._build_actor()
      self.actor.start()
      self.num_steps_since_env_reset = 0
      self._env_burnin()

  def save(self):
    tf_state = tf.nest.map_structure(lambda t: t.numpy(), self._learner.get_vars())

    # Note: this state is valid as an imitation state.
    combined_state = dict(
        state=tf_state,
        config=self._teacher_state['config'],
        name_map=self._teacher_state['name_map'],
        step=self._step,
        # Use the same key as run_lib for compatibility with eval_lib.
        rl_config=dataclasses.asdict(self._config),
    )
    pickled_state = pickle.dumps(combined_state)

    logging.info('saving state to %s', self._pickle_path)
    with open(self._pickle_path, 'wb') as f:
      f.write(pickled_state)

  def flush(self):
    num_frames = (
        self._config.actor.num_envs * self._config.actor.rollout_length
        * self._num_ppo_batches * self._step
    )
    metrics = self._logger.flush(
        step=self._step * self._num_ppo_batches,
        extras=dict(num_frames=num_frames),
    )

    if metrics is None:
      return

    print('\nStep:', self._step)

    timings: dict = metrics['timings']
    timing_str = ', '.join(
        ['{k}: {v:.3f}'.format(k=k, v=v) for k, v in timings.items()])
    print(timing_str)

    ko_diff = metrics['ko_diff']
    print(f'KO_diff_per_minute: {ko_diff:.3f}')

    learner_metrics = metrics['learner']
    pre_update = learner_metrics['rl']['ppo_step']['0']
    mean_actor_kl = pre_update['actor_kl']['mean']
    max_actor_kl = pre_update['actor_kl']['max']
    print(f'actor_kl: mean={mean_actor_kl:.3g} max={max_actor_kl:.3g}')
    teacher_kl = pre_update['teacher_kl']
    print(f'teacher_kl: {teacher_kl:.3g}')
    print(f'uev: {learner_metrics["rl"]["value"]["uev"]:.3f}')

  def get_log_data(
      self,
      trajectories: list[evaluators.Trajectory],
      metrics: dict,
  ):
    timings = {}

    # TODO: we shouldn't take the mean over these timings
    step_time = self.step_profiler.mean_time()
    steps_per_rollout = self._config.actor.num_envs * self._config.actor.rollout_length
    fps = len(trajectories) * steps_per_rollout / step_time
    mps = fps / (60 * 60)  # in-game minutes per second

    timings.update(
        rollout=self.rollout_profiler.mean_time(),
        learner=self.learner_profiler.mean_time(),
        reset=self.reset_profiler.mean_time(),
        total=step_time,
        fps=fps,
        mps=mps,
    )
    actor_timing = metrics['actor'].pop('timing')
    for key in ['env_pop', 'env_push']:
      timings[key] = actor_timing[key]
    for key in ['agent_pop', 'agent_step']:
      timings[key] = actor_timing[key][self._exploiter_port]

    # concatenate along the batch dimension
    states: Game = tf.nest.map_structure(
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
        **metrics,
    )

  def _env_burnin(self):
    for _ in range(self._config.runtime.burnin_steps_after_reset):
      self.unroll()

  def reset_env(self):
    with self.reset_profiler:
      logging.info('Resetting environments')
      self.actor.reset_env()
      self.num_steps_since_env_reset = 0
      self._env_burnin()

  def _rollout(self):
    self.num_steps_since_env_reset += 1
    return self.actor.rollout(self._unroll_length)

  def unroll(self):
    trajectories, _ = self._rollout()
    _, self._hidden_state = self._learner.compiled_unroll(
        exploiter_trajectory=trajectories[self._exploiter_port],
        mixture_trajectory=trajectories[self._mixture_port],
        initial_state=self._hidden_state,
    )

  def should_reset_env(self):
    reset_interval = self._config.runtime.reset_every_n_steps
    if reset_interval is None:
      return False
    return self.num_steps_since_env_reset >= reset_interval

  def update_variables(self):
    variables = {self._exploiter_port: self._learner.exploiter_variables()}

    new_training_phase = self._step % self._config.exploiter_train_steps == 0
    if new_training_phase:
      logging.info('Starting new exploiter training phase')
      variables[self._mixture_port] = self._learner.mixture_variables()
      if self._config.reset_exploiter:
        self._learner.reset_exploiter_policy()

    self.actor.update_variables(variables)

    if new_training_phase:
      self.value_function_burnin()

  def step(
      self,
      ppo_steps: int = None,  # Set to 0 to train only the value function.
      train_mixture_policy: bool = True,
  ) -> tuple[list[evaluators.Trajectory], dict]:
    ppo_steps = ppo_steps or self._config.learner.ppo.num_epochs

    if self.should_reset_env():
      self.reset_env()

    with self.rollout_profiler:
      exploiter_trajectories = []
      mixture_trajectories = []
      actor_metrics = []
      for _ in range(self._num_ppo_batches):
        trajectories, timings = self._rollout()
        exploiter_trajectories.append(trajectories[self._exploiter_port])
        mixture_trajectories.append(trajectories[self._mixture_port])
        actor_metrics.append(timings)

      actor_metrics = tf.nest.map_structure(
          lambda *xs: np.mean(xs), *actor_metrics)

    with self.learner_profiler:
      if self._config.scale_exploiter_weight:
        num_phases = 1 + self._step // self._config.exploiter_train_steps
        exploiter_weight = 1 / (num_phases + 1)
      else:
        exploiter_weight = self._config.learner.exploiter_weight

      self._hidden_state, learner_metrics = self._learner.step(
          exploiter_trajectories, mixture_trajectories, self._hidden_state,
          num_ppo_epochs=ppo_steps, train_mixture_policy=train_mixture_policy,
          exploiter_weight=exploiter_weight)

    learner_metrics.update(
        exploiter_weight=exploiter_weight,
        ppo_steps=ppo_steps,
    )
    stats = dict(learner=learner_metrics, actor=actor_metrics)

    return exploiter_trajectories, stats

  def step_and_log(self, **kwargs):
    with self.step_profiler:
      trajectories, stats = self.step(**kwargs)

    log_data = self.get_log_data(trajectories, stats)
    self._logger.record(log_data)
    self._maybe_flush()

  def optimizer_burnin(self):
    logging.info('Optimizer burnin')
    learning_rate = self._learner._config.learning_rate  # TODO: hacky
    assert isinstance(learning_rate, tf.Variable)
    learning_rate.assign(0)
    for _ in range(self._config.optimizer_burnin_steps // self._num_ppo_batches):
      self.step()  # don't log this data
    learning_rate.assign(self._config.learner.learning_rate)

  def value_function_burnin(self):
    # Flush logger before and after because log structure changes.
    self.flush()

    logging.info('Value function burnin')
    for _ in range(self._config.value_burnin_steps // self._num_ppo_batches):
      self._step += 1
      self.step_and_log(ppo_steps=0, train_mixture_policy=False)
    self.flush()

    logging.info('Finished value function burnin')

  def run(self):
    try:
      if self._step == 0:
        self.optimizer_burnin()

      logging.info('Main training loop')

      for _ in range(self._config.runtime.max_step):
        self.update_variables()  # Will do value function burnin on step 0.

        self._step += 1
        self.step_and_log()

        self._maybe_save()

      self.save()
    finally:
      self.actor.stop()

def run(config: Config):
  tag = config.runtime.tag or train_lib.get_experiment_tag()
  # Might want to use wandb.run.dir instead, but it doesn't seem
  # to be set properly even when we try to override it.
  expt_dir = config.runtime.expt_dir
  if expt_dir is None:
    expt_dir = os.path.join(config.runtime.expt_root, tag)
    os.makedirs(expt_dir, exist_ok=True)
  logging.info('experiment directory: %s', expt_dir)

  # Restore from existing save file if it exists.
  restore_path = None
  restore_from_checkpoint = False
  pickle_path = os.path.join(expt_dir, 'latest.pkl')
  if os.path.exists(pickle_path):
    logging.info('Restoring from checkpoint %s', pickle_path)
    restore_path = pickle_path
    restore_from_checkpoint = True
  elif config.restore:
    restore_path = config.restore

  if config.teacher and config.restore:
    raise ValueError('Must pass exactly one of "teacher" and "restore".')

  # TODO: Port code to run_lib
  if restore_path:
    mixture_state = saving.load_state_from_disk(restore_path)

    previous_config = flag_utils.dataclass_from_dict(
        Config, mixture_state['rl_config'])

    if (restore_from_checkpoint and previous_config.restore
        and previous_config.restore != config.restore):
      raise ValueError(
          'Requested restore path does not match checkpoint: '
          f'{config.restore} (requested) != {previous_config.restore} (checkpoint)')

    # Older configs used agent.path instead of config.teacher
    previous_teacher = previous_config.teacher or previous_config.agent.path

    if config.teacher and config.teacher != previous_teacher:
      assert restore_from_checkpoint
      raise ValueError(
          'Requested teacher does not match checkpoint: '
          f'{config.teacher} (requested) != {previous_teacher} (checkpoint)')

    logging.info(f'Using teacher: {previous_teacher}')
    teacher_state = saving.load_state_from_disk(previous_teacher)
    wandb.config.update(dict(teacher=previous_teacher))

    # All that matters here are the 'config' and 'state.policy' fields.
    mixture_policy_state = mixture_state
    exploiter_policy_state = dict(
        mixture_state,
        state=dict(policy=mixture_state['state']['rl']['policy']),
    )
    step = mixture_state['step']
  elif config.teacher:
    logging.info(f'Initializing from teacher: {config.teacher}')
    teacher_state = saving.load_state_from_disk(config.teacher)
    mixture_policy_state = teacher_state
    exploiter_policy_state = teacher_state
    step = 0
  else:
    raise ValueError('Must pass exactly one of "teacher" and "restore".')

  if config.override_delay is not None:
    teacher_state['config']['policy']['delay'] = config.override_delay

  # Make sure we don't train the teacher
  with tf_utils.non_trainable_scope():
    teacher = saving.load_policy_from_state(teacher_state)

  mixture_policy = saving.load_policy_from_state(mixture_policy_state)
  exploiter_policy = saving.load_policy_from_state(exploiter_policy_state)

  pretraining_config = flag_utils.dataclass_from_dict(
      train_lib.Config, teacher_state['config'])

  # TODO: put this code into saving.py or train_lib.py
  vf_config = pretraining_config.value_function
  value_function = None
  if vf_config.train_separate_network:
    value_net_config = pretraining_config.network
    if vf_config.separate_network_config:
      value_net_config = vf_config.network
    value_function = vf_lib.ValueFunction(
        network_config=value_net_config,
        embed_state_action=mixture_policy.embed_state_action,
    )

  # This allows us to only update the optimizer state.
  learning_rate = tf.Variable(
      config.learner.learning_rate,
      name='learning_rate',
      trainable=False)
  learner_config = dataclasses.replace(
      config.learner, learning_rate=learning_rate)

  learner = learner_lib.Learner(
      config=learner_config,
      exploiter_policy=exploiter_policy,
      mixture_policy=mixture_policy,
      teacher=teacher,
      value_function=value_function,
  )

  # Initialize and restore Learner variables
  learner.initialize(run_lib.dummy_trajectory(mixture_policy, 1, 1))
  if restore_path:
    tf.nest.map_structure(
        lambda var, val: var.assign(val),
        learner.get_vars(), mixture_state['state'])
  else:
    learner.restore_from_imitation(teacher_state['state'])

  EXPLOITER_PORT = 1
  MIXTURE_PORT = 2

  dolphin_kwargs = dict(
      players={
          EXPLOITER_PORT: dolphin_lib.AI(),
          MIXTURE_PORT: dolphin_lib.AI(),
      },
      **config.dolphin.to_kwargs(),
  )

  common_agent_kwargs = dict(
      compile=config.agent.compile,
      name=config.agent.name,
  )
  agent_kwargs = {
      EXPLOITER_PORT: dict(common_agent_kwargs, state=exploiter_policy_state),
      MIXTURE_PORT: dict(common_agent_kwargs, state=mixture_policy_state),
  }

  env_kwargs = {}
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
      learner=learner,
      exploiter_port=EXPLOITER_PORT,
      mixture_port=MIXTURE_PORT,
      build_actor=build_actor,
      step=step,
      pickle_path=pickle_path,
      teacher_state=teacher_state,
  )

  experiment_manager.run()
