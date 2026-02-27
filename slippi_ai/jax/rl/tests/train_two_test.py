#!/usr/bin/env python
"""Test two-agent JAX RL training loop."""

import melee
from absl import app
import fancyflags as ff
import wandb

from slippi_ai import flag_utils, paths, reward as reward_lib
from slippi_ai.jax.rl import run_lib, learner as learner_lib, train_two_lib

DEFAULT_CONFIG = train_two_lib.Config(
    runtime=train_two_lib.RuntimeConfig(
        max_step=10,
        log_interval=0,
        burnin_steps_after_reset=1,
    ),
    learner=learner_lib.LearnerConfig(
        learning_rate=0,
        reward=reward_lib.RewardConfig(nana_ratio=0),
        ppo=learner_lib.PPOConfig(num_batches=2),
    ),
    actor=run_lib.ActorConfig(
        use_fake_envs=True,
        num_envs=1,
        rollout_length=64,
    ),
    p1=train_two_lib.AgentConfig(
        teacher=str(paths.JAX_DEMO_CHECKPOINT),
        char=melee.Character.FOX,
        name=['Diamond Player'],
    ),
    p2=train_two_lib.AgentConfig(
        teacher=str(paths.JAX_DEMO_CHECKPOINT),
        char=melee.Character.FOX,
        name=['Diamond Player'],
    ),
)

if __name__ == '__main__':
  __spec__ = None  # https://github.com/python/cpython/issues/87115

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(DEFAULT_CONFIG))

  def main(_):
    wandb.init(mode='offline')

    CONFIG.value['learner1'] = flag_utils.override_dict(
        CONFIG.value['learner'], CONFIG, ['learner1'])
    CONFIG.value['learner2'] = flag_utils.override_dict(
        CONFIG.value['learner'], CONFIG, ['learner2'])

    config = flag_utils.dataclass_from_dict(train_two_lib.Config, CONFIG.value)
    train_two_lib.run(config)

  app.run(main)
