import dataclasses
import typing as tp

from absl import app
import fancyflags as ff

from slippi_ai import (
    eval_lib,
    dolphin,
    flag_utils,
    saving,
    tf_utils,
)

from slippi_ai import evaluators
from slippi_ai.rl import learner as learner_lib

field = lambda f: dataclasses.field(default_factory=f)

@dataclasses.dataclass
class RuntimeConfig:
  max_step: int = 10  # maximum training step
  max_runtime: int = 1 * 60 * 60  # maximum runtime in seconds
  log_interval: int = 10  # seconds between logging
  save_interval: int = 300  # seconds between saving to disk

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


CONFIG = ff.DEFINE_dict(
    'config',
    **flag_utils.get_flags_from_dataclass(Config))


def run(config: Config):
  pretraining_state = eval_lib.load_state(
      tag=config.agent.tag, path=config.agent.path)

  # Make sure we don't train the teacher
  with tf_utils.non_trainable_scope():
    teacher = saving.load_policy_from_state(pretraining_state)
  policy = saving.load_policy_from_state(pretraining_state)

  rl_state = pretraining_state
  rl_state['step'] = 0

  batch_size = config.actor.num_envs
  learner = learner_lib.Learner(
      config=config.learner,
      teacher=teacher,
      policy=policy,
      batch_size=batch_size,
  )

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

  actor = evaluators.RolloutWorker(
      agent_kwargs={PORT: agent_kwargs},
      dolphin_kwargs=dolphin_kwargs,
      env_kwargs=env_kwargs,
      num_envs=config.actor.num_envs,
      async_envs=config.actor.async_envs,
      async_inference=config.actor.async_inference,
      use_gpu=config.actor.gpu_inference,
  )

  for _ in range(config.runtime.max_step):
    policy_vars = {PORT: learner.policy_variables()}
    actor.update_variables(policy_vars)

    trajectories, timings = actor.rollout(config.actor.rollout_length)
    del timings

    metrics = learner.step(trajectories[PORT])
    print(metrics['kl'].numpy().mean())

  actor.stop()

def main(_):
  config = flag_utils.dataclass_from_dict(Config, CONFIG.value)
  run(config)

if __name__ == '__main__':
  app.run(main)
