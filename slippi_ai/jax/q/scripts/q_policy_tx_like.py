#!/usr/bin/env python
"""Test imitation learning training loop - JAX version."""

import dataclasses
import os

from absl import app, flags
import wandb
import fancyflags as ff

import melee
from slippi_ai import flag_utils, paths
from slippi_ai.jax import saving, train_lib
from slippi_ai.jax.q import train_q_policy

NET_NAME = 'tx_like'

def default_config():
  config = train_q_policy.Config()

  config.data.batch_size = 512
  config.data.unroll_length = 80
  config.data.damage_ratio = 0.01
  config.data.num_workers = 1
  config.data.balance_characters = True
  config.learner.learning_rate = 1e-4
  config.learner.reward_halflife = 8

  config.dataset.mirror = True
  config.dataset.allowed_opponents='all'
  # config.dataset.banned_names="${BANNED_NAMES}"
  config.dataset.data_dir = os.environ.get("DATA_DIR")
  config.dataset.meta_path = os.environ.get("META_PATH")
  config.runtime.log_interval = 300
  config.runtime.num_evals_per_epoch = 8

  return config

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  TOY_DATA = flags.DEFINE_bool('toy_data', False, 'Use toy data for quick testing')

  CHAR = flags.DEFINE_string('char', 'falco', 'Character to use')

  NUM_DAYS = flags.DEFINE_float('num_days', 14, 'Number of days to train for')

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(default_config()))

  # passed to wandb.init
  WANDB = ff.DEFINE_dict(
      'wandb',
      project=ff.String('slippi-ai'),
      mode=ff.Enum('online', ['online', 'offline', 'disabled']),
      group=ff.String('q_learning'),
      name=ff.String(None),
      notes=ff.String(None),
      dir=ff.String(None, 'directory to save logs'),
  )

  def main(_):
    config = flag_utils.dataclass_from_dict(train_q_policy.Config, CONFIG.value)
    config.runtime.max_runtime = int(NUM_DAYS.value * 24 * 60 * 60)

    assert config.initialize_policies_from is not None
    imitation_state = saving.load_state_from_disk(config.initialize_policies_from)
    imitation_config = flag_utils.dataclass_from_dict(
        train_lib.Config,
        saving.upgrade_config(imitation_state['config']))

    if TOY_DATA.value:
      config.dataset.data_dir = str(paths.TOY_DATA_DIR)
      config.dataset.meta_path = str(paths.TOY_META_PATH)
      config.dataset.test_ratio = 0.5
      char = 'all'
      config.data.cached = True
      config.data.num_workers = 0
      config.runtime.log_interval = 15
      config.runtime.num_evals_per_epoch = 0
    else:
      char = CHAR.value

      if config.tag is None:
        network = imitation_config.network
        assert network['name'] == NET_NAME, f"Expected network name {NET_NAME} but got {network['name']}"
        d = imitation_config.policy.delay
        n = network[NET_NAME]['num_layers']
        h = network[NET_NAME]['hidden_size']
        fs = imitation_config.observation.frame_skip.skip
        config.tag = f"{char}_d{d}_{NET_NAME}_{n}x{h}_fs{fs}"

    config.dataset.allowed_characters = char

    wandb_kwargs = dict(WANDB.value)
    if wandb_kwargs['name'] is None:
      wandb_kwargs['name'] = config.tag
      if TOY_DATA.value:
        wandb_kwargs['mode'] = 'disabled'

    wandb.init(
        config=dataclasses.asdict(config),
        **wandb_kwargs,
    )
    train_q_policy.train(config)

  app.run(main)
