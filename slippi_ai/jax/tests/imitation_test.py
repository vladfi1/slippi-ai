#!/usr/bin/env python
"""Test imitation learning training loop - JAX version."""

from absl import app
import wandb
import fancyflags as ff

from slippi_ai import paths, flag_utils
from slippi_ai import data as data_lib
from slippi_ai.jax import train_lib

DEFAULT_CONFIG = train_lib.Config(
    dataset=data_lib.DatasetConfig(
        data_dir=str(paths.TOY_DATA_DIR),
        meta_path=str(paths.TOY_META_PATH),
        test_ratio=0.5,
    ),
    data=data_lib.DataConfig(
        compressed=True,
        balance_characters=True,
        batch_size=2,
    ),
    learner=train_lib.learner_lib.LearnerConfig(),
    runtime=train_lib.RuntimeConfig(
        log_interval=4,
        max_runtime=10,
        eval_every_n=50,
        num_eval_steps=2,
    ),
    network=dict(
        name='tx_like',
        tx_like=dict(
            hidden_size=2,
            num_layers=1,
            ffw_multiplier=4,
            recurrent_layer='lstm',
            activation='gelu',
        ),
    ),
    value_function=train_lib.ValueFunctionConfig(
        separate_network_config=False,
    ),
    controller_head=dict(
        name='autoregressive',
        autoregressive=dict(
            residual_size=2,
            component_depth=0,
        ),
    ),
)

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(DEFAULT_CONFIG))


  def main(_):
    wandb.init(mode='offline')  # avoid network calls during tests

    config = flag_utils.dataclass_from_dict(
        train_lib.Config, CONFIG.value)

    train_lib.train(config)

  app.run(main)
