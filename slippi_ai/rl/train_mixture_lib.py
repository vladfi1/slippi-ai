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
  exploiter_train_steps: int = 2
  # After updating the opponent to the new mixture policy, we can also reset the
  # exploiter policy to the same new mixture policy. This reduces correlation
  # between exploiter training phases, but we may prefer to start from the old
  # exploiter policy as it is likely still good against the new mixture policy.
  reset_exploiter: bool = True

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
  ):
    self._config = config
    self._learner = learner
    self._build_actor = build_actor
    self._unroll_length = config.actor.rollout_length
    self._exploiter_port = exploiter_port
    self._mixture_port = mixture_port
    self._num_ppo_batches = config.learner.ppo.num_batches

    if 2 * config.runtime.burnin_steps_after_reset > config.runtime.reset_every_n_steps:
      raise UserWarning('Spending more than half of the time in env burnin.')

    batch_size = config.actor.num_envs
    self._hidden_state = learner.initial_state(batch_size)

    self.update_profiler = utils.Profiler(burnin=0)
    self.learner_profiler = utils.Profiler()
    self.rollout_profiler = utils.Profiler()
    self.reset_profiler = utils.Profiler(burnin=0)

    with self.reset_profiler:
      self.actor = self._build_actor()
      self.actor.start()
      self.num_steps_since_reset = 0
      self._burnin()

  def _burnin(self):
    for _ in range(self._config.runtime.burnin_steps_after_reset):
      self.unroll()

  def reset_env(self):
    with self.reset_profiler:
      logging.info('Resetting environments')
      self.actor.reset_env()
      self.num_steps_since_reset = 0
      self._burnin()

  def _rollout(self):
    self.num_steps_since_reset += 1
    return self.actor.rollout(self._unroll_length)

  def unroll(self):
    trajectories, _ = self._rollout()
    _, self._hidden_state = self._learner.compiled_unroll(
        exploiter_trajectory=trajectories[self._exploiter_port],
        mixture_trajectory=trajectories[self._mixture_port],
        initial_state=self._hidden_state,
    )

  def should_reset(self):
    reset_interval = self._config.runtime.reset_every_n_steps
    if reset_interval is None:
      return False
    return self.num_steps_since_reset >= reset_interval

  def step(
      self,
      step: int,
      ppo_steps: int = None,
  ) -> tuple[list[evaluators.Trajectory], dict]:
    with self.update_profiler:
      variables = {self._exploiter_port: self._learner.exploiter_variables()}
      if step % self._config.exploiter_train_steps == 0:
        logging.info('Starting new exploiter training phase')
        # TODO: might be good to do value function burnin again, particularly
        # if we are resetting the exploiter policy.
        variables[self._mixture_port] = self._learner.mixture_variables()
        if self._config.reset_exploiter:
          self._learner.reset_exploiter_policy()
      self.actor.update_variables(variables)

    if self.should_reset():
      self.reset_env()

    with self.rollout_profiler:
      exploiter_trajectories = []
      mixture_trajectories = []
      actor_timings = []
      for _ in range(self._num_ppo_batches):
        trajectories, timings = self._rollout()
        exploiter_trajectories.append(trajectories[self._exploiter_port])
        mixture_trajectories.append(trajectories[self._mixture_port])
        actor_timings.append(timings)

      actor_timings = tf.nest.map_structure(
          lambda *xs: np.mean(xs), *actor_timings)

    with self.learner_profiler:
      self._hidden_state, metrics = self._learner.step(
          exploiter_trajectories, mixture_trajectories,
          self._hidden_state, num_ppo_epochs=ppo_steps)

    return exploiter_trajectories, dict(learner=metrics, actor_timing=actor_timings)


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

  rl_config_dict = dataclasses.asdict(config)

  def save(step: int):
    tf_state = tf.nest.map_structure(lambda t: t.numpy(), learner.get_vars())

    # Note: this state is valid as an imitation state.
    combined_state = dict(
        state=tf_state,
        config=teacher_state['config'],
        name_map=teacher_state['name_map'],
        step=step,
        # Use the same key as run_lib for compatibility with eval_lib.
        rl_config=rl_config_dict,
    )
    pickled_state = pickle.dumps(combined_state)

    logging.info('saving state to %s', pickle_path)
    with open(pickle_path, 'wb') as f:
      f.write(pickled_state)

    # TODO: save to s3?

  maybe_save = utils.Periodically(save, config.runtime.save_interval)

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

  learner_manager = ExperimentManager(
      config=config,
      learner=learner,
      exploiter_port=EXPLOITER_PORT,
      mixture_port=MIXTURE_PORT,
      build_actor=build_actor,
  )

  step_profiler = utils.Profiler()

  def get_log_data(
      trajectories: list[evaluators.Trajectory],
      metrics: dict,
  ):
    timings = {}

    # TODO: we shouldn't take the mean over these timings
    step_time = step_profiler.mean_time()
    steps_per_rollout = config.actor.num_envs * config.actor.rollout_length
    fps = len(trajectories) * steps_per_rollout / step_time
    mps = fps / (60 * 60)  # in-game minutes per second

    timings.update(
        rollout=learner_manager.rollout_profiler.mean_time(),
        learner=learner_manager.learner_profiler.mean_time(),
        reset=learner_manager.reset_profiler.mean_time(),
        total=step_time,
        fps=fps,
        mps=mps,
    )
    actor_timing = metrics['actor_timing']
    for key in ['env_pop', 'env_push']:
      timings[key] = actor_timing[key]
    for key in ['agent_pop', 'agent_step']:
      timings[key] = actor_timing[key][EXPLOITER_PORT]

    learner_metrics = metrics['learner']

    # concatenate along the batch dimension
    states = tf.nest.map_structure(
        lambda *xs: np.concatenate(xs, axis=1),
        *[t.states for t in trajectories])
    kos = reward.compute_rewards(states, damage_ratio=0)
    kos_per_minute = kos.mean() * (60 * 60)

    return dict(
        ko_diff=kos_per_minute,
        timings=timings,
        learner=learner_metrics,
    )

  logger = run_lib.Logger()

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
    pre_update = learner_metrics['rl']['ppo_step']['0']
    actor_kl = pre_update['actor_kl']['mean']
    print(f'actor_kl: {actor_kl:.3g}')
    teacher_kl = pre_update['teacher_kl']
    print(f'teacher_kl: {teacher_kl:.3g}')
    print(f'uev: {learner_metrics["rl"]["value"]["uev"]:.3f}')

  maybe_flush = utils.Periodically(flush, config.runtime.log_interval)

  try:
    steps_per_epoch = config.learner.ppo.num_batches

    # Optimizer burnin
    learning_rate.assign(0)
    for _ in range(config.optimizer_burnin_steps // steps_per_epoch):
      learner_manager.step(0, ppo_steps=1)

    logging.info('Value function burnin')

    learning_rate.assign(config.learner.learning_rate)
    for i in range(config.value_burnin_steps // steps_per_epoch):
      with step_profiler:
        trajectories, metrics = learner_manager.step(step, ppo_steps=0)

      if i > 0:
        logger.record(get_log_data(trajectories, metrics))
        maybe_flush(step)

      step += 1

    # Need flush here because logging structure changes based on ppo_steps.
    flush(step)

    logging.info('Main training loop')

    for i in range(config.runtime.max_step):
      with step_profiler:
        trajectories, metrics = learner_manager.step(step)

      if i > 0:
        logger.record(get_log_data(trajectories, metrics))
        maybe_flush(step)

      step += 1
      maybe_save(step)

    save(step)

  finally:
    learner_manager.actor.stop()
