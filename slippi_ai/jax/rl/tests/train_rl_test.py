#!/usr/bin/env python
"""Test single-agent JAX RL training loop."""

import melee
from absl import app
import fancyflags as ff
import wandb

from slippi_ai import flag_utils, paths, reward as reward_lib
from slippi_ai.jax.rl import run_lib, learner as learner_lib

DEFAULT_CONFIG = run_lib.Config(
    teacher=str(paths.JAX_DEMO_CHECKPOINT),
    runtime=run_lib.RuntimeConfig(
        max_step=10,
        log_interval=0,
        burnin_steps_after_reset=1,
        reset_every_n_steps=6,
    ),
    learner=learner_lib.LearnerConfig(
        learning_rate=0,
        reward=reward_lib.RewardConfig(nana_ratio=0),
        ppo=learner_lib.PPOConfig(num_batches=2),
    ),
    actor=run_lib.ActorConfig(
        use_fake_envs=True,
        num_envs=2,
        rollout_length=64,
    ),
    agent=run_lib.AgentConfig(
        char=[melee.Character.FOX],
        name=['Diamond Player'],
    ),
    opponent=run_lib.OpponentConfig(type=run_lib.OpponentType.SELF),
)

if __name__ == '__main__':
  __spec__ = None  # https://github.com/python/cpython/issues/87115

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(DEFAULT_CONFIG))

  def main(_):
    wandb.init(mode='offline')
    config = flag_utils.dataclass_from_dict(run_lib.Config, CONFIG.value)
    run_lib.run(config)

  app.run(main)
