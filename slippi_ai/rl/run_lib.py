import dataclasses
import typing as tp

import numpy as np
import tensorflow as tf

from slippi_ai import (
    data,
    dolphin,
    eval_lib,
    evaluators,
    flag_utils,
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
  max_step: int = 10  # maximum training step
  max_runtime: tp.Optional[int] = None  # maximum runtime in seconds
  log_interval: int = 10  # seconds between logging
  save_interval: int = 300  # seconds between saving to disk

  # Periodically reset the environments to deal with memory leaks in dolphin.
  reset_every_n_steps: tp.Optional[int] = None

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


def run(config: Config):
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

  # TODO: we only keep this here for save/restore compatibility with imitation
  # learning. We should get rid of "Variable" learning_rates in both places.
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
      batch_size=batch_size,
  )

  # Initialize variables before restoring.
  embedders = dict(policy.embed_state_action.embedding)
  embed_controller = policy.controller_embedding
  dummy_trajectory = evaluators.Trajectory(
      states=embedders['state'].dummy([2, batch_size]),
      name=embedders['name'].dummy([2, batch_size]),
      actions=eval_lib.dummy_sample_outputs(embed_controller, [2, batch_size]),
      rewards=np.full([1, batch_size], 0, dtype=np.float32),
      is_resetting=np.full([2, batch_size], False),
      initial_state=policy.initial_state(batch_size),
      delayed_actions=[
          eval_lib.dummy_sample_outputs(embed_controller, [batch_size])
      ] * policy.delay,
  )
  learner.step(dummy_trajectory)

  # Restore variables from pretraining state.
  tf_state = dict(
      policy=policy.variables,
      value_function=value_function.variables if value_function else [],
      optimizer=learner.optimizer.variables,
  )

  # Drop "step".
  # TODO: add a separate RL "step"?
  pretraining_tf_state = {
      k: pretraining_state['state'][k]
      for k in tf_state
  }

  tf.nest.map_structure(
      lambda var, val: var.assign(val),
      tf_state, pretraining_tf_state)
  # Hack: restoration from IL will overwrite the RL learning rate.
  learning_rate.assign(config.learner.learning_rate)

  PORT = 1
  ENEMY_PORT = 2
  dolphin_kwargs = dict(
      players={
          PORT: dolphin.AI(),
          ENEMY_PORT: dolphin.CPU(),
      },
      **dataclasses.asdict(config.dolphin),
  )

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

  rollout_profiler = utils.Profiler()
  learner_profiler = utils.Profiler()

  def log(
      step: int,
      trajectory: evaluators.Trajectory,
      learner_metrics: dict,
  ):
    print('\nStep:', step)

    timings = {}
    if step > 0:
      rollout_time = rollout_profiler.mean_time()
      learner_time = learner_profiler.mean_time()
      total_time = rollout_time + learner_time

      steps_per_rollout = config.actor.num_envs * config.actor.rollout_length
      fps = steps_per_rollout / total_time
      mps = fps / (60 * 60)  # in-game minutes per second

      timings.update(
          rollout=rollout_time,
          learner=learner_time,
          fps=fps,
          mps=mps,
      )
      timings.update(learner=learner_profiler.mean_time())

      timing_str = ', '.join(
          ['{k}: {v:.2f}'.format(k=k, v=v) for k, v in timings.items()])
      print(timing_str)

    kos = reward.compute_rewards(trajectory.states, damage_ratio=0)
    kos_per_minute = kos.mean() * (60 * 60)
    print(f'KO_diff_per_minute: {kos_per_minute:.3f}')

    for key in ['teacher_kl', 'actor_kl']:
      print(f'{key}: {learner_metrics[key].numpy().mean():.3f}')
    print(f'uev: {learner_metrics["value"]["uev"].numpy().mean():.3f}')

    to_log = dict(
        ko_diff=kos_per_minute,
        timings=timings,
        learner=learner_metrics,
    )
    train_lib.log_stats(to_log, step)

  maybe_log = utils.Periodically(log, config.runtime.log_interval)

  reset_interval = config.runtime.reset_every_n_steps

  actor = build_actor()
  actor.start()

  try:
    for step in range(config.runtime.max_step):

      with rollout_profiler:

        if reset_interval and step > 0 and step % reset_interval == 0:
          actor.stop()
          actor = build_actor()
          actor.start()

        policy_vars = {PORT: learner.policy_variables()}
        actor.update_variables(policy_vars)

        trajectories, timings = actor.rollout(config.actor.rollout_length)
        trajectory = trajectories[PORT]
        del timings

      with learner_profiler:
        metrics = learner.compiled_step(trajectory)

      maybe_log(
          step=step,
          trajectory=trajectory,
          learner_metrics=metrics,
      )
  finally:
    actor.stop()
