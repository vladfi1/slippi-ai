import dataclasses
import enum
import logging
import os
import pickle
import typing as tp

import numpy as np
import tensorflow as tf

from slippi_ai import (
    dolphin,
    eval_lib,
    evaluators,
    flag_utils,
    policies,
    reward,
    saving,
    tf_utils,
    train_lib,
    utils,
)

from slippi_ai import value_function as vf_lib
from slippi_ai.rl import learner as learner_lib

field = lambda f: dataclasses.field(default_factory=f)

@dataclasses.dataclass
class RuntimeConfig:
  expt_root: str = 'experiments/rl'
  expt_dir: tp.Optional[str] = None
  tag: tp.Optional[str] = None

  max_step: int = 10  # maximum training step
  max_runtime: tp.Optional[int] = None  # maximum runtime in seconds
  log_interval: int = 10  # seconds between logging
  save_interval: int = 300  # seconds between saving to disk
  use_fake_data: bool = False

  # Periodically reset the environments to deal with memory leaks in dolphin.
  reset_every_n_steps: tp.Optional[int] = None
  # Without burnin, we see a spike in teacher_kl after every reset. My guess is
  # that this is because the trajectories and therefore gradients become highly
  # correlated, which is bad. Empirically 10 is a good value to set this to.
  burnin_steps_after_reset: int = 0

@dataclasses.dataclass
class ActorConfig:
  rollout_length: int = 64
  num_envs: int = 1
  async_envs: bool = False
  num_env_steps: int = 0
  inner_batch_size: int = 1
  async_inference: bool = False
  gpu_inference: bool = False
  num_agent_steps: int = 0

@dataclasses.dataclass
class AgentConfig:
  # TODO: merge with ActorConfig?
  path: tp.Optional[str] = None
  tag: tp.Optional[str] = None
  compile: bool = True
  name: str = 'Master Player'

  def get_kwargs(self) -> dict:
    state = eval_lib.load_state(path=self.path, tag=self.tag)
    return dict(
        state=state,
        compile=self.compile,
        name=self.name,
    )

@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = field(RuntimeConfig)

  # num_actors: int = 1
  dolphin: eval_lib.DolphinConfig = field(eval_lib.DolphinConfig)
  learner: learner_lib.LearnerConfig = field(learner_lib.LearnerConfig)
  actor: ActorConfig = field(ActorConfig)
  agent: AgentConfig = field(AgentConfig)

  # Take learner steps without changing the parameters to burn-in the
  # optimizer state for RL.
  optimizer_burnin_steps: int = 0
  # Take some steps to update just the value function before doing RL.
  # Useful if we're training against the level 9 cpu.
  value_burnin_steps: int = 0


class LearnerManager:

  def __init__(
      self,
      learner: learner_lib.Learner,
      batch_size: int,
      unroll_length: int,
      build_actor: tp.Callable[[], evaluators.RolloutWorker],
      port: int,
      num_ppo_batches: int = 1,
      burnin_steps_after_reset: int = 0,
  ):
    self._learner = learner
    self._hidden_state = learner.initial_state(batch_size)
    self._build_actor = build_actor
    self._unroll_length = unroll_length
    self._port = port
    self._num_ppo_batches = num_ppo_batches
    self._burnin_steps_after_reset = burnin_steps_after_reset

    self.update_profiler = utils.Profiler(burnin=0)
    self.learner_profilers = {True: utils.Profiler(), False: utils.Profiler()}
    self.rollout_profiler = utils.Profiler()
    self.reset_profiler = utils.Profiler(burnin=0)

    with self.reset_profiler:
      self.initialize_actor()

  def initialize_actor(self):
    self.actor = self._build_actor()
    self.actor.start()
    for _ in range(self._burnin_steps_after_reset):
      self.unroll()

  def reset_actor(self):
    with self.reset_profiler:
      self.actor.stop()
      self.initialize_actor()

  def unroll(self):
    with self.rollout_profiler:
      trajectories, _ = self.actor.rollout(self._unroll_length)
      trajectory = trajectories[self._port]

    with self.learner_profilers[False]:
      _, self._hidden_state = self._learner.compiled_unroll(
          trajectory, self._hidden_state)

  def step(self, ppo_steps: int = None) -> tuple[list[evaluators.Trajectory], dict]:
    with self.update_profiler:
      variables = {self._port: self._learner.policy_variables()}
      self.actor.update_variables(variables)

    with self.rollout_profiler:
      trajectories = []
      actor_timings = []
      for _ in range(self._num_ppo_batches):
        trajectory, timings = self.actor.rollout(self._unroll_length)
        trajectories.append(trajectory[self._port])
        actor_timings.append(timings)

      actor_timings = tf.nest.map_structure(
          lambda *xs: np.mean(xs), *actor_timings)

    with self.learner_profilers[True]:
      self._hidden_state, metrics = self._learner.ppo(
          trajectories, self._hidden_state, num_epochs=ppo_steps)

    return trajectories, dict(learner=metrics, actor_timing=actor_timings)

class Logger:

  def __init__(self):
    self.buffer = []

  def record(self, to_log):
    to_log = utils.map_single_structure(train_lib.mean, to_log)
    self.buffer.append(to_log)

  def flush(self, step: int) -> tp.Optional[dict]:
    if not self.buffer:
      return None

    to_log = tf.nest.map_structure(lambda *xs: np.mean(xs), *self.buffer)
    train_lib.log_stats(to_log, step, take_mean=False)
    self.buffer = []
    return to_log

def dummy_trajectory(
    policy: policies.Policy,
    unroll_length: int,
    batch_size: int,
) -> evaluators.Trajectory:
  embedders = dict(policy.embed_state_action.embedding)
  embed_controller = policy.controller_embedding
  shape = [unroll_length + 1, batch_size]
  return evaluators.Trajectory(
      states=embedders['state'].dummy(shape),
      name=embedders['name'].dummy(shape),
      actions=eval_lib.dummy_sample_outputs(embed_controller, shape),
      rewards=np.full([unroll_length, batch_size], 0, dtype=np.float32),
      is_resetting=np.full(shape, False),
      initial_state=policy.initial_state(batch_size),
      delayed_actions=[
          eval_lib.dummy_sample_outputs(embed_controller, [batch_size])
      ] * policy.delay,
  )

class DummyActor:

  def __init__(self, policy: policies.Policy, batch_size: int, port: int):
    self.policy = policy
    self.batch_size = batch_size
    self.port = port

  def rollout(self, unroll_length: int) -> tuple[evaluators.Trajectory, dict]:
    trajectory = dummy_trajectory(self.policy, unroll_length, self.batch_size)
    timings = {
        'env_pop': 0,
        'env_push': 0,
        'agent_pop': {self.port: 0},
        'agent_step': {self.port: 0},
    }
    return {self.port: trajectory}, timings

  def update_variables(self, variables):
    del variables

  def start(self):
    pass

  def stop(self):
    pass

def run(config: Config):
  tag = config.runtime.tag or train_lib.get_experiment_tag()
  # Might want to use wandb.run.dir instead, but it doesn't seem
  # to be set properly even when we try to override it.
  expt_dir = config.runtime.expt_dir
  if expt_dir is None:
    expt_dir = os.path.join(config.runtime.expt_root, tag)
    os.makedirs(expt_dir, exist_ok=True)
  logging.info('experiment directory: %s', expt_dir)

  pretraining_state = eval_lib.load_state(
      tag=config.agent.tag, path=config.agent.path)

  saving.upgrade_config(pretraining_state['config'])
  pretraining_config = flag_utils.dataclass_from_dict(
      train_lib.Config, pretraining_state['config'])

  # Make sure we don't train the teacher
  with tf_utils.non_trainable_scope():
    teacher = saving.load_policy_from_state(pretraining_state)
  policy = saving.load_policy_from_state(pretraining_state)

  rl_state = pretraining_state
  rl_state['step'] = 0

  # TODO: put this code into saving.py or train_lib.py
  vf_config = pretraining_config.value_function
  value_function = None
  if vf_config.train_separate_network:
    value_net_config = pretraining_config.network
    if vf_config.separate_network_config:
      value_net_config = vf_config.network
    value_function = vf_lib.ValueFunction(
        network_config=value_net_config,
        embed_state_action=policy.embed_state_action,
    )

  # This allows us to only update the optimizer state.
  learning_rate = tf.Variable(
      config.learner.learning_rate,
      name='learning_rate',
      trainable=False)
  learner_config = dataclasses.replace(
      config.learner, learning_rate=learning_rate)

  batch_size = config.actor.num_envs
  learner = learner_lib.Learner(
      config=learner_config,
      teacher=teacher,
      policy=policy,
      value_function=value_function,
  )

  # Initialize and restore variables
  learner.initialize(dummy_trajectory(policy, 1, 1))
  learner.restore_from_imitation(pretraining_state['state'])

  PORT = 1
  ENEMY_PORT = 2
  dolphin_kwargs = dict(
      players={
          PORT: dolphin.AI(),
          ENEMY_PORT: dolphin.CPU(),
      },
      **dataclasses.asdict(config.dolphin),
  )

  if config.runtime.use_fake_data:
    build_actor = lambda: DummyActor(
        policy, batch_size=config.actor.num_envs, port=PORT)
  else:
    agent_kwargs = dict(
        state=rl_state,
        compile=config.agent.compile,
        name=config.agent.name,
    )

    env_kwargs = {}
    if config.actor.async_envs:
      env_kwargs.update(
          num_steps=config.actor.num_env_steps,
          inner_batch_size=config.actor.inner_batch_size,
      )

    build_actor = lambda: evaluators.RolloutWorker(
        agent_kwargs={PORT: agent_kwargs},
        dolphin_kwargs=dolphin_kwargs,
        env_kwargs=env_kwargs,
        num_envs=config.actor.num_envs,
        async_envs=config.actor.async_envs,
        async_inference=config.actor.async_inference,
        use_gpu=config.actor.gpu_inference,
    )

  learner_manager = LearnerManager(
      learner=learner,
      batch_size=batch_size,
      unroll_length=config.actor.rollout_length,
      port=PORT,
      build_actor=build_actor,
      burnin_steps_after_reset=config.runtime.burnin_steps_after_reset,
      num_ppo_batches=config.learner.ppo.num_batches,
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
        learner=learner_manager.learner_profilers[True].mean_time(),
        reset=learner_manager.reset_profiler.mean_time(),
        total=step_time,
        fps=fps,
        mps=mps,
    )
    actor_timing = metrics['actor_timing']
    for key in ['env_pop', 'env_push']:
      timings[key] = actor_timing[key]
    for key in ['agent_pop', 'agent_step']:
      timings[key] = actor_timing[key][PORT]

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
    actor_kl = pre_update['actor_kl']['mean']
    print(f'actor_kl: {actor_kl:.3g}')
    teacher_kl = pre_update['teacher_kl']
    print(f'teacher_kl: {teacher_kl:.3g}')
    print(f'uev: {learner_metrics["value"]["uev"]:.3f}')

  maybe_flush = utils.Periodically(flush, config.runtime.log_interval)

  pickle_path = os.path.join(expt_dir, 'latest.pkl')

  def save(step: int):
    # Note: this state is valid as an imitation state.
    combined_state = dict(
        state=learner.get_state(),
        config=pretraining_state['config'],
        name_map=pretraining_state['name_map'],
        step=step,
        rl_config=dataclasses.asdict(config),
    )
    pickled_state = pickle.dumps(combined_state)

    logging.info('saving state to %s', pickle_path)
    with open(pickle_path, 'wb') as f:
      f.write(pickled_state)

    # TODO: save to s3?

  maybe_save = utils.Periodically(save, config.runtime.save_interval)

  try:
    steps_per_epoch = config.learner.ppo.num_batches

    reset_interval = config.runtime.reset_every_n_steps
    if reset_interval:
      reset_interval = reset_interval // steps_per_epoch

    # Optimizer burnin
    learning_rate.assign(0)
    for _ in range(config.optimizer_burnin_steps // steps_per_epoch):
      learner_manager.step(ppo_steps=1)

    step = 0

    logging.info('Value function burnin')

    learning_rate.assign(config.learner.learning_rate)
    for _ in range(config.value_burnin_steps // steps_per_epoch):
      with step_profiler:
        trajectories, metrics = learner_manager.step(ppo_steps=0)

      if step > 0:
        logger.record(get_log_data(trajectories, metrics))
        maybe_flush(step)

      step += 1

    # Need flush here because logging structure changes based on ppo_steps.
    flush(step)

    logging.info('Main training loop')

    for _ in range(config.runtime.max_step):
      with step_profiler:
        if step > 0 and reset_interval and step % reset_interval == 0:
          logging.info('Resetting environments')
          learner_manager.reset_actor()
        trajectories, metrics = learner_manager.step()

      if step > 0:
        logger.record(get_log_data(trajectories, metrics))
        maybe_flush(step)

      step += 1
      maybe_save(step)

    save(step)

  finally:
    learner_manager.actor.stop()
