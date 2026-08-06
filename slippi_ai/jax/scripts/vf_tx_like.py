#!/usr/bin/env python

"""Train a value function standalone."""

import os

from absl import app, flags
import fancyflags as ff

import wandb

from slippi_ai import flag_utils, paths
from slippi_ai.jax import embed, networks, train_lib, train_vf, saving

NET_NAME = networks.TransformerLike.name()

def default_config():
  config = train_vf.Config(max_names=32)

  config.data.batch_size = 512
  config.data.unroll_length = 80
  config.test_unroll_multiplier = 16
  config.data.num_workers = 1
  config.data.unroll_chunks = 4
  config.data.balance_characters = True
  config.learner.learning_rate = 1e-4
  config.learner.bf16 = True
  config.learner.reward_halflife = 4

  config.embed.controller.type = 'custom_v1'
  config.embed.player.with_nana = True
  config.embed.items.type = embed.ItemsType.FLAT
  config.embed.with_fod = True
  config.embed.with_randall = True

  config.network['name'] = NET_NAME
  config.network[NET_NAME].update(
      num_layers=1,
      hidden_size=512,
  )

  config.dataset.mirror = False
  config.dataset.allowed_opponents = 'all'
  config.dataset.dataset_path = os.environ.get("DATASET_PATH")
  config.dataset.data_dir = os.environ.get("DATA_DIR")
  config.dataset.meta_path = os.environ.get("META_PATH")
  config.data.wds.cache_dir = './wds-cache'
  config.runtime.log_interval = 300
  config.runtime.num_evals_per_epoch = 8

  return config

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  TOY_DATA = flags.DEFINE_bool('toy_data', False, 'Use toy data for quick testing')
  NUM_DAYS = flags.DEFINE_float('num_days', 2, 'Number of days to train for')

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(default_config()))

  WANDB = ff.DEFINE_dict(
      'wandb',
      project=ff.String('slippi-ai'),
      mode=ff.Enum('online', ['online', 'offline', 'disabled']),
      group=ff.String('value_function'),
      name=ff.String(None),
      notes=ff.String(None),
      dir=ff.String(None, 'directory to save logs'),
  )

  def main(_):
    config = flag_utils.dataclass_from_dict(train_vf.Config, CONFIG.value)
    config.runtime.max_runtime = int(NUM_DAYS.value * 24 * 60 * 60)

    if TOY_DATA.value:
      config.dataset.dataset_path = str(paths.TOY_DATASET)
      config.dataset.test_ratio = 0.5
      char = 'all'
      # config.data.cached = True
      # config.data.num_workers = 0
      config.runtime.log_interval = 15
      config.runtime.num_evals_per_epoch = 0
    else:
      assert config.compatible_policy is not None
      imitation_state = saving.load_state_from_disk(config.compatible_policy)
      imitation_config = flag_utils.dataclass_from_dict(
          train_lib.Config, imitation_state['config'])

      char = imitation_config.dataset.allowed_characters

      if config.tag is None:
        ops = config.dataset.allowed_opponents
        if ops == 'all':
          op = ''
        elif ops == char:
          op = '_ditto'
        else:
          op = f"_vs_{ops}"

        n = config.network[config.network['name']]['num_layers']
        h = config.network[config.network['name']]['hidden_size']
        um = config.test_unroll_multiplier

        config.tag = f"vf_{char}{op}_tx{n}x{h}_um{um}"

    wandb_kwargs = dict(WANDB.value)
    if wandb_kwargs['name'] is None:
      wandb_kwargs['name'] = config.tag
      if TOY_DATA.value:
        wandb_kwargs['mode'] = 'disabled'

    wandb.init(**wandb_kwargs)
    train_vf.train(config)

  app.run(main)
