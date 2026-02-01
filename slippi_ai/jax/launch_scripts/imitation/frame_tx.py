#!/usr/bin/env python

"""Train a model using imitation learning."""

import os

from absl import app, flags
import fancyflags as ff

import wandb

import melee
from slippi_ai import flag_utils, paths
from slippi_ai.jax import train_lib, embed

FRAME_TX = 'frame_tx'

def default_config():
  config = train_lib.Config()

  config.max_names = 128
  config.policy.delay = 21
  config.data.batch_size=512
  config.data.unroll_length=80
  config.data.damage_ratio=0.01
  config.data.num_workers=1
  config.data.balance_characters=True
  config.learner.learning_rate=1e-4
  config.learner.reward_halflife=8
  config.embed.controller.axis_spacing=32
  config.embed.controller.shoulder_spacing=10
  config.embed.player.with_nana = True
  # MLP Items embed not yet implemented for JAX
  config.embed.items.type = embed.ItemsType.FLAT
  # config.embed.items.type = embed.ItemsType.MLP
  # config.embed.items.mlp_sizes = (128, 32)
  config.embed.with_fod = True
  config.embed.with_randall = True

  config.network[FRAME_TX].update(
      num_layers=2,
      stack_layers=True,
  )

  config.value_function.separate_network_config = True
  config.value_function.network[FRAME_TX]['num_layers'] = 1

  ch_name = 'autoregressive'
  config.controller_head['name'] = ch_name
  config.controller_head[ch_name]['component_depth'] = 2
  config.controller_head[ch_name]['residual_size'] = 128
  config.dataset.mirror = True
  config.dataset.allowed_opponents='all'
  # config.dataset.banned_names="${BANNED_NAMES}"
  config.dataset.data_dir = os.environ.get("DATA_DIR")
  config.dataset.meta_path = os.environ.get("META_PATH")
  config.runtime.log_interval = 300
  config.runtime.num_evals_per_epoch = 4

  return config

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  NET = ff.DEFINE_dict(
      'net',
      name=ff.String(FRAME_TX),
      hidden_size=ff.Integer(64),
      num_heads=ff.Integer(4),
      rnn_cell=ff.String('gru'),
      stack_layers=ff.Boolean(False),
      stacked_rnns=ff.Boolean(True),
      remat_layers=ff.Boolean(True),
      remat_rnns=ff.Boolean(False),
      remat_controller_rnn=ff.Boolean(True),
  )
  TOY_DATA = flags.DEFINE_bool('toy_data', False, 'Use toy data for quick testing')

  TOY_VF = flags.DEFINE_bool('toy_vf', False, 'Use a toy value function for quick testing')

  CHAR = flags.DEFINE_string('char', 'falco', 'Character to use')

  NUM_DAYS = flags.DEFINE_float('num_days', 14, 'Number of days to train for')

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(default_config()))

  # passed to wandb.init
  WANDB = ff.DEFINE_dict(
      'wandb',
      project=ff.String('slippi-ai'),
      mode=ff.Enum('online', ['online', 'offline', 'disabled']),
      group=ff.String('imitation'),
      name=ff.String(None),
      notes=ff.String(None),
      dir=ff.String(None, 'directory to save logs'),
  )

  def main(_):
    config = flag_utils.dataclass_from_dict(train_lib.Config, CONFIG.value)
    config.runtime.max_runtime = int(NUM_DAYS.value * 24 * 60 * 60)

    net_config = dict(NET.value)
    net = net_config.pop('name')
    delay = config.policy.delay


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
      data_dir = config.dataset.data_dir
      meta_path = config.dataset.meta_path
      assert os.path.isdir(data_dir), data_dir
      assert os.path.isfile(meta_path), meta_path

      char = CHAR.value

      if config.tag is None:
        config.tag = f"{char}_d{delay}_{net}"

    config.dataset.allowed_characters = char

    config.network['name'] = net
    config.network[net].update(net_config)

    vf_net_config = config.value_function.network
    if TOY_VF.value:
      vf_net_config['name'] = 'mlp'
      vf_net_config['mlp'].update(depth=0)
    else:
      vf_net_config['name'] = net
      vf_net_config[net].update(net_config)

    wandb_kwargs = dict(WANDB.value)
    if wandb_kwargs['name'] is None:
      wandb_kwargs['name'] = config.tag
      if TOY_DATA.value:
        wandb_kwargs['mode'] = 'disabled'

    wandb.init(
        config=CONFIG.value,
        **wandb_kwargs,
    )
    train_lib.train(config)

  app.run(main)
