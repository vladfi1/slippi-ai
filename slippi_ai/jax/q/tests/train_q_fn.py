#!/usr/bin/env python
"""Test q-only training loop - JAX version."""

from absl import app
import wandb
import fancyflags as ff

from slippi_ai import paths, flag_utils
from slippi_ai import data as data_lib
from slippi_ai.jax import networks
from slippi_ai.jax.q import train_q_fn

network_config = networks.default_config()
network_config['name'] = 'tx_like'
network_config['tx_like'].update(
    hidden_size=1,
    num_layers=1,
    ffw_multiplier=4,
    recurrent_layer='lstm',
    activation='gelu',
)

DEFAULT_CONFIG = train_q_fn.Config(
    dataset=data_lib.DatasetConfig(
        data_dir=str(paths.TOY_DATA_DIR),
        meta_path=str(paths.TOY_META_PATH),
        test_ratio=0.5,
    ),
    data=data_lib.DataConfig(
        compressed=True,
        balance_characters=True,
        batch_size=2,
        unroll_length=5,
    ),
    runtime=train_q_fn.RuntimeConfig(
        log_interval=4,
        max_runtime=10,
        num_evals_per_epoch=2,
        max_eval_steps=3,
    ),
    network=network_config,
    delay=0,
)

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(DEFAULT_CONFIG))

  def main(_):
    wandb.init(mode='offline')  # avoid network calls during tests

    config = flag_utils.dataclass_from_dict(
        train_q_fn.Config, CONFIG.value)

    train_q_fn.train(config)

  app.run(main)
