#!/usr/bin/env python
"""Launch script for q-only training - JAX version."""

import os

from absl import app, flags
import wandb
import fancyflags as ff

from slippi_ai import flag_utils, paths
from slippi_ai.jax import embed, saving, train_lib
from slippi_ai.jax.nash import train_q_fn

NET_NAME = 'tx_like'

def default_config():
  config = train_q_fn.Config()

  config.delay = 0
  config.data.batch_size = 512
  config.data.unroll_length = 80
  config.test_unroll_multiplier = 16
  config.data.damage_ratio = 0.01
  config.data.num_workers = 2
  # config.data.unroll_chunks = 4
  config.data.balance_characters = True
  config.learner.learning_rate = 1e-4
  config.learner.reward_halflife = 4

  q_fn = config.q_function
  q_fn.embed.controller.type = embed.ControllerType.CUSTOM_V1.value
  q_fn.embed.player.with_nana = True
  q_fn.embed.items.type = embed.ItemsType.FLAT
  q_fn.embed.with_fod = True
  q_fn.embed.with_randall = True

  config.dataset.mirror = False
  config.dataset.allowed_characters = 'fox'
  config.dataset.data_dir = os.environ.get("DATA_DIR")
  config.dataset.meta_path = os.environ.get("META_PATH")
  config.runtime.log_interval = 300
  config.runtime.num_evals_per_epoch = 8

  return config

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'
  os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
  os.environ['TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC'] = '-1'

  # Flag parsing might be too late to set these env vars
  # CUDA_MALLOC_ASYNC = flags.DEFINE_bool(
  #     'cuda_malloc_async', False, 'Whether to use CUDA malloc async allocator')

  NET = ff.DEFINE_dict(
      'net',
      name=ff.String(NET_NAME),
      hidden_size=ff.Integer(512),
      num_layers=ff.Integer(1),
      ffw_multiplier=ff.Integer(2),
      recurrent_layer=ff.String('lstm'),
  )
  EMBED = ff.DEFINE_dict(
      'embed',
      name=ff.String('enhanced'),
      simple=dict(),
      enhanced=dict(
          rnn_cell=ff.String('lstm'),
          use_controller_rnn=ff.Boolean(False),
      ),
  )

  TOY_DATA = flags.DEFINE_bool('toy_data', False, 'Use toy data for quick testing')

  CHAR = flags.DEFINE_string('char', 'fox', 'Character to use')

  NUM_DAYS = flags.DEFINE_float('num_days', 14, 'Number of days to train for')

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(default_config()))

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
    config = flag_utils.dataclass_from_dict(train_q_fn.Config, CONFIG.value)
    config.runtime.max_runtime = int(NUM_DAYS.value * 24 * 60 * 60)

    imitation_config = None
    if config.compatible_policy is not None:
      imitation_state = saving.load_state_from_disk(config.compatible_policy)
      imitation_config = flag_utils.dataclass_from_dict(
          train_lib.Config,
          saving.upgrade_config(imitation_state['config']))

    net_config = dict(NET.value)
    net = net_config.pop('name')

    char = CHAR.value

    if TOY_DATA.value:
      config.dataset.data_dir = str(paths.TOY_DATA_DIR)
      config.dataset.meta_path = str(paths.TOY_META_PATH)
      char = 'all'
      config.data.cached = True
      config.data.num_workers = 0
      config.runtime.log_interval = 15
      config.runtime.num_evals_per_epoch = 0.02
    else:
      char = CHAR.value

      if config.tag is None:
        n = config.q_function.network[net]['num_layers']
        h = net_config['hidden_size']
        net_str = f"{n}x{h}"

        head_str = f"{config.q_function.head.num_layers}x{config.q_function.head.hidden_size}"

        if imitation_config is not None:
          fs = imitation_config.observation.frame_skip.skip
        else:
          fs = config.observation.frame_skip.skip
        um = config.test_unroll_multiplier
        rh = int(config.learner.reward_halflife)

        config.tag = f"nash_q_{char}_d{config.delay}_{net_str}_qv{head_str}_fs{fs}_um{um}_rh{rh}"

    config.dataset.allowed_characters = char

    embed_config = dict(EMBED.value)
    embed_name = embed_config['name']
    embed_config['enhanced']['hidden_size'] = net_config['hidden_size'] // 4
    embed_config['enhanced']['use_self_nana'] = char in ['popo', 'all']

    def update_embed_config(config: dict):
      config['name'] = embed_name
      config[embed_name].update(embed_config[embed_name])

    def update_network_config(config: dict):
      config['name'] = net
      config[net].update(net_config)
      update_embed_config(config['embed'])

    update_network_config(config.q_function.network)

    wandb_kwargs = dict(WANDB.value)
    if wandb_kwargs['name'] is None:
      wandb_kwargs['name'] = config.tag
      if TOY_DATA.value:
        wandb_kwargs['mode'] = 'disabled'

    wandb.init(
        # config=dataclasses.asdict(config),
        **wandb_kwargs,
    )
    train_q_fn.train(config)

  app.run(main)
