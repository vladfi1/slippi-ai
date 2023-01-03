"""Run RL on a single machine.

Learning runs on the main thread, acting runs on separate threads.
"""

import dataclasses
import pickle
import tree
import typing as tp

from absl import app
import ray
import fancyflags as ff
import wandb

from slippi_ai import (
    eval_lib,
    dolphin,
    flag_utils,
    paths,
    s3_lib,
    saving,
    types,
    utils,
)

from slippi_ai.rl import actor as actor_lib
from slippi_ai.rl import learner as learner_lib

PORT = 1
ENEMY_PORT = 2


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
  pretraining_tag: tp.Optional[str] = None
  pretraining_ckpt: str = str(paths.DEMO_CHECKPOINT)

  runtime: RuntimeConfig = field(RuntimeConfig)

  dolphin: eval_lib.DolphinConfig = field(eval_lib.DolphinConfig)

  learner: learner_lib.LearnerConfig = field(learner_lib.LearnerConfig)

  num_parallel_actors: int = 1  # controls env parallelism
  num_envs_per_actor: int = 2
  rollout_length: int = 64


CONFIG = ff.DEFINE_dict(
    'config',
    **flag_utils.get_flags_from_dataclass(Config))

# passed to wandb.init
WANDB = ff.DEFINE_dict(
    'wandb',
    project=ff.String('slippi-ai'),
    mode=ff.Enum('disabled', ['online', 'offline', 'disabled']),
    notes=ff.String(None),
)

make_remote_actor_pool = ray.remote(num_cpus=1)(actor_lib.ActorPool).remote

def log_actor(
    results: actor_lib.RolloutResults,
    profiler: utils.Profiler,
    step: int,
):
  total_time = profiler.mean_time()
  print(results.timings)

  trajectories = results.trajectories[PORT]
  mean_reward = trajectories.observations.reward.mean()

  reward_per_minute = mean_reward * 60 * 60
  print('mean_reward:', mean_reward)

  num_steps = trajectories.observations.reward.size
  sps = num_steps / total_time

  timings = types.nt_to_nest(results.timings)
  timings.update(total=total_time, sps=sps)

  to_log = dict(
      timings=timings,
      reward_per_minute=reward_per_minute,
  )

  wandb.log(dict(actor=to_log), step=step)

def log_learner(
    metrics: dict,
    profiler: utils.Profiler,
    step: int,
):
  print('learner time:', profiler.mean_time())

  metrics = tree.map_structure(lambda x: x.numpy().mean(), metrics)
  print('kl', metrics['kl'])

  metrics['step_time'] = profiler.mean_time()

  wandb.log(dict(learner=metrics), step=step)


def run(config: Config):
  if config.pretraining_tag is not None:
    pretraining_state = saving.get_state_from_s3(
        config.pretraining_tag)
  else:
    with open(config.pretraining_ckpt, 'rb') as f:
      pretraining_state = pickle.load(f)

  # TODO: load rl state from existing checkpoint
  rl_state = pretraining_state

  learner = learner_lib.Learner(
      config=config.learner,
      teacher_state=pretraining_state,
      rl_state=rl_state,
      batch_size=config.num_envs_per_actor * config.num_parallel_actors,
  )

  players = {
      PORT: dolphin.AI(),
      ENEMY_PORT: dolphin.CPU(),
  }
  env_kwargs = dict(
      players=players,
      **dataclasses.asdict(config.dolphin),
  )

  actor_pool_kwargs = dict(
      num_actors=config.num_envs_per_actor,
      policy_states={PORT: rl_state},
      env_kwargs=env_kwargs,
      num_steps_per_rollout=config.rollout_length,
  )
  remote_actor_pools = [
      make_remote_actor_pool(**actor_pool_kwargs)
      for _ in range(config.num_parallel_actors)
  ]

  actor_profiler = utils.Profiler()
  learner_profiler = utils.Profiler()

  for step in range(config.runtime.max_step):
    policy_vars = {PORT: learner.policy_variables()}

    with actor_profiler:
      results_futures = [
          actor_pool.rollout.remote(policy_vars)
          for actor_pool in remote_actor_pools
      ]
      results = ray.get(results_futures)
      results = actor_lib.merge_results(results)

    log_actor(results, actor_profiler, step)

    with learner_profiler:
      metrics = learner.step(results.trajectories[PORT])
    
    log_learner(metrics, learner_profiler, step)

  for actor_pool in remote_actor_pools:
    actor_pool.stop.remote()

def main(_):
  wandb.init(
      config=CONFIG.value,
      **WANDB.value,
  )
  config = flag_utils.dataclass_from_dict(Config, CONFIG.value)
  run(config)

if __name__ == '__main__':
  app.run(main)
