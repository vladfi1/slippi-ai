#!/usr/bin/env python
"""Test imitation learning training loop - JAX version."""

import dataclasses
import os

os.environ["JAX_COMPILATION_CACHE_DIR"] = "./untracked/jax_cache"

from absl import app, flags
import wandb
import fancyflags as ff

from slippi_ai import flag_utils, paths
from slippi_ai.jax import saving, train_lib
from slippi_ai.jax.nash import train_nash_policy
from slippi_ai.jax.agents import DType

NET_NAME = 'tx_like'

def default_config():
  config = train_nash_policy.Config()

  config.data.batch_size = 256
  config.data.unroll_length = 84
  config.data.num_workers = 1
  config.data.unroll_chunks = 4
  config.learner.learning_rate = 3e-5

  # Match Nash RL reward config
  config.reward.damage_ratio = 0.01
  config.reward.ledge_grab_penalty = 0.02
  config.reward.stalling_penalty = 0.1
  config.reward.stalling_threshold = 50
  config.reward.approaching_factor = 1e-3

  config.learner.num_samples = 3

  config.dataset.mirror = False
  config.dataset.data_dir = os.environ.get("DATA_DIR")
  config.dataset.meta_path = os.environ.get("META_PATH")
  config.runtime.log_interval = 300
  config.runtime.num_evals_per_epoch = 8

  config.learner.sample_policy_dtype = DType.FP16
  config.learner.nash_policy_dtype = DType.BF16

  return config

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  TOY_DATA = flags.DEFINE_bool('toy_data', False, 'Use toy data for quick testing')

  NUM_DAYS = flags.DEFINE_float('num_days', 14, 'Number of days to train for')

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(default_config()))

  # passed to wandb.init
  WANDB = ff.DEFINE_dict(
      'wandb',
      project=ff.String('slippi-ai'),
      mode=ff.Enum('online', ['online', 'offline', 'disabled']),
      group=ff.String('nash'),
      name=ff.String(None),
      notes=ff.String(None),
      dir=ff.String(None, 'directory to save logs'),
  )

  def main(_):
    config = flag_utils.dataclass_from_dict(train_nash_policy.Config, CONFIG.value)
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
      char = imitation_config.dataset.allowed_characters

      if config.tag is None:
        network = imitation_config.network
        assert network['name'] == NET_NAME, f"Expected network name {NET_NAME} but got {network['name']}"
        d = imitation_config.policy.delay
        n = network[NET_NAME]['num_layers']
        h = network[NET_NAME]['hidden_size']
        ch_name = 'autoregressive'
        ch_config = imitation_config.controller_head
        assert ch_config['name'] == ch_name, f"Expected controller head name {ch_name} but got {ch_config['name']}"
        ch_config = ch_config[ch_name]
        assert ch_config['component']['name'] == NET_NAME, f"Expected controller head component name {NET_NAME} but got {ch_config['component']['name']}"
        chn = ch_config['component'][NET_NAME]['num_layers']
        chh = ch_config['component'][NET_NAME]['hidden_size']
        fs = imitation_config.policy.frame_skip
        ns = config.learner.num_samples

        config.tag = f"np_{char}_d{d}_{n}x{h}_ch{chn}x{chh}_rfs{fs}_ns{ns}"

        if config.learner.bf16:
          config.tag += '_bf16'

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
    train_nash_policy.train(config)

  app.run(main)
