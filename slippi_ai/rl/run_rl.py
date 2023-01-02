import dataclasses
import typing as tp

from absl import app
import fancyflags as ff

from slippi_ai import (
    eval_lib,
    dolphin,
    flag_utils,
    paths,
    s3_lib,
    saving,
    utils,
)

from slippi_ai.rl import actor as actor_lib
from slippi_ai.rl import learner as learner_lib

@dataclasses.dataclass
class FileCacheConfig:
  use: bool = False
  path: tp.Optional[str] = None
  wipe: bool = False

@dataclasses.dataclass
class RuntimeConfig:
  max_step: int = 10  # maximum training step
  max_runtime: int = 1 * 60 * 60  # maximum runtime in seconds
  log_interval: int = 10  # seconds between logging
  save_interval: int = 300  # seconds between saving to disk

field = lambda f: dataclasses.field(default_factory=f)


@dataclasses.dataclass
class Config:
  # tag: str
  # pretraining_tag: str

  runtime: RuntimeConfig = field(RuntimeConfig)

  num_actors: int = 1
  dolphin: eval_lib.DolphinConfig = field(eval_lib.DolphinConfig)

  learner: learner_lib.LearnerConfig = field(learner_lib.LearnerConfig)

  rollout_length: int = 64


CONFIG = ff.DEFINE_dict(
    'config',
    **flag_utils.get_flags_from_dataclass(Config))


def run(config: Config):
  # actors = make_actors()

  # pretraining_state = saving.get_state_from_s3(
  #     config.pretraining_tag)

  import pickle
  with open(paths.DEMO_CHECKPOINT, 'rb') as f:
    pretraining_state = pickle.load(f)

  # try:
  #   rl_state = saving.get_state_from_s3(config.tag)
  # except KeyError:
  #   logging.info('no state found at %s', config.tag)
  rl_state = pretraining_state
  rl_state['step'] = 0

  learner = learner_lib.Learner(
      config=config.learner,
      teacher_state=pretraining_state,
      rl_state=rl_state,
      batch_size=config.num_actors,
  )

  PORT = 1
  ENEMY_PORT = 2
  players = {
      PORT: dolphin.AI(),
      ENEMY_PORT: dolphin.CPU(),
  }

  ports = eval_lib.get_open_ports(config.num_actors)

  actors: tp.List[actor_lib.Actor] = []
  for port in ports:
    dolphin_config = dataclasses.replace(config.dolphin)
    dolphin_config.slippi_port = port
    env_kwargs = dict(
        players=players,
        **dataclasses.asdict(dolphin_config),
    )

    actor = actor_lib.Actor(
        policy_states={PORT: rl_state},
        env_kwargs=env_kwargs,
        num_steps_per_rollout=config.rollout_length,
    )

    actors.append(actor)

  for _ in range(config.runtime.max_step):
    policy_vars = {PORT: learner.policy_variables()}

    results = [actor.rollout(policy_vars) for actor in actors]

    trajectories = [result.trajectories[PORT] for result in results]
    batched_trajectories = utils.batch_nest(trajectories)

    metrics = learner.step(batched_trajectories)

    print(metrics['kl'].numpy().mean())

  for actor in actors:
    actor.stop()

def main(_):
  config = flag_utils.dataclass_from_dict(Config, CONFIG.value)
  run(config)

if __name__ == '__main__':
  app.run(main)
