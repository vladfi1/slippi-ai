#!/usr/bin/env python
"""Test Q-policy RL training loop with fake envs."""

import melee
from absl import app
import fancyflags as ff
import wandb

from slippi_ai import flag_utils, paths, reward
from slippi_ai.jax.q import train_q_rl
from slippi_ai.jax.q import rl_learner
from slippi_ai.jax.rl import run_lib

DEFAULT_CONFIG = train_q_rl.Config(
  teacher=str(paths.JAX_POLICY_CHECKPOINT),
  q_function=str(paths.JAX_Q_FN_CKPT),
  runtime=train_q_rl.RuntimeConfig(
    max_step=5,
    log_interval=0,
    expt_root='experiments/tests/jax/q_rl',
  ),
  learner=rl_learner.LearnerConfig(
    num_samples=1,
    epoch_length=2,
    microbatch_size=1,
  ),
  reward=reward.RewardConfig(
    nana_ratio=0,
  ),
  actor=run_lib.ActorConfig(
    use_fake_envs=True,
    num_envs=2,
    rollout_length=60,
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
    config = flag_utils.dataclass_from_dict(train_q_rl.Config, CONFIG.value)
    train_q_rl.run(config)

  app.run(main)
