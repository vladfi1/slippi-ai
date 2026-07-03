#!/usr/bin/env python
"""Launch script for q-only training - JAX version."""

import os

from absl import app, flags
import wandb
import fancyflags as ff

from slippi_ai import flag_utils, paths
from slippi_ai.jax import embed, saving, train_lib
from slippi_ai.jax.q import train_q_fn

NET_NAME = 'tx_like'

def default_config():
  config = train_q_fn.Config()

  config.delay = 0
  config.data.batch_size = 512
  config.data.unroll_length = 84
  config.test_unroll_multiplier = 16
  config.data.num_workers = 2
  config.data.balance_characters = True
  config.learner.learning_rate = 1e-4

  # Match Q RL reward config
  config.learner.reward_halflife = 4
  config.reward.damage_ratio = 0.01
  config.reward.ledge_grab_penalty = 0.02
  config.reward.stalling_penalty = 0.1
  config.reward.stalling_threshold = 50
  config.reward.approaching_factor = 1e-3

  q_fn = config.q_function
  q_fn.embed.controller.type = embed.ControllerType.CUSTOM_V1.value
  q_fn.embed.player.with_nana = True
  q_fn.embed.items.type = embed.ItemsType.FLAT
  q_fn.embed.with_fod = True
  q_fn.embed.with_randall = True

  config.dataset.mirror = False
  config.dataset.allowed_opponents = 'all'
  config.dataset.data_dir = os.environ.get("DATA_DIR")
  config.dataset.meta_path = os.environ.get("META_PATH")
  config.runtime.log_interval = 300
  config.runtime.num_evals_per_epoch = 8

  return config

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  CORE_NET = ff.DEFINE_dict(
      'core_net',
      hidden_size=ff.Integer(1024),
      num_layers=ff.Integer(1),
      ffw_multiplier=ff.Integer(2),
      recurrent_layer=ff.String('lstm'),
  )
  ACTION_NET = ff.DEFINE_dict(
      'action_net',
      hidden_size=ff.Integer(256),
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

  TAG_SUFFIX = flags.DEFINE_string('tag_suffix', None, 'Tag suffix to add to the experiment name')

  CHAR = flags.DEFINE_string('char', 'falco', 'Character to use')

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
      if imitation_config is not None:
        char = imitation_config.dataset.allowed_characters
        config.dataset.allowed_opponents = imitation_config.dataset.allowed_opponents
      else:
        char = CHAR.value

      if config.tag is None:
        parts = ['q', char, f'd{config.delay}']

        def net_str(net_config: dict):
          n = net_config['num_layers']
          h = net_config['hidden_size']
          return f"{n}x{h}"

        core_str = net_str(CORE_NET.value)
        action_str = net_str(ACTION_NET.value)
        head_str = f"{config.q_function.head.num_layers}x{config.q_function.head.hidden_size}"
        parts.extend(['c' + core_str, 'a' + action_str, 'qv' + head_str])

        if imitation_config is not None:
          fs = imitation_config.policy.frame_skip
        else:
          fs = config.q_function.frame_skip

        um = config.test_unroll_multiplier
        rh = int(config.learner.reward_halflife)
        parts.extend([f'rfs{fs}', f'um{um}', f'rh{rh}'])

        gae = config.learner.gae_lambda
        if gae != 0:
          parts.append(f'gae{gae:.1f}')

        lr = config.learner.learning_rate
        if lr != 1e-4:
          parts.append(f'lr{lr:.0e}')

        if EMBED.value['name'] == 'enhanced' and EMBED.value['enhanced']['use_controller_rnn']:
          parts.append('crnn')

        if TAG_SUFFIX.value is not None:
          parts.append(TAG_SUFFIX.value)

        config.tag = '_'.join(parts)

    config.dataset.allowed_characters = char

    embed_config = dict(EMBED.value)
    embed_name = embed_config['name']
    embed_config['enhanced']['hidden_size'] = CORE_NET.value['hidden_size'] // 4
    embed_config['enhanced']['use_self_nana'] = char in ['popo', 'all']

    config.q_function.core_net['name'] = NET_NAME
    config.q_function.core_net[NET_NAME].update(CORE_NET.value)
    config.q_function.core_net['embed']['name'] = embed_name
    config.q_function.core_net['embed'][embed_name].update(embed_config[embed_name])

    config.q_function.action_net['name'] = NET_NAME
    config.q_function.action_net[NET_NAME].update(ACTION_NET.value)

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
