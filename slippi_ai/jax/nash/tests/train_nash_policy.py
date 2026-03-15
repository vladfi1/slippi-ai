#!/usr/bin/env python
"""Test imitation learning training loop - JAX version."""

from absl import app
import wandb
import fancyflags as ff
import warnings

from slippi_ai import paths, flag_utils
from slippi_ai import data as data_lib
from slippi_ai.jax.nash import train_nash_policy

DEFAULT_CONFIG = train_nash_policy.Config(

    dataset=data_lib.DatasetConfig(
        data_dir=str(paths.TOY_DATA_DIR),
        meta_path=str(paths.TOY_META_PATH),
        test_ratio=0.5,
    ),
    data=data_lib.DataConfig(
        balance_characters=True,
        batch_size=2,
        unroll_length=5,
    ),
    runtime=train_nash_policy.RuntimeConfig(
        log_interval=4,
        max_runtime=10,
        num_evals_per_epoch=2,
        max_eval_steps=3,
    ),
    initialize_policies_from=str(paths.JAX_IMITATION_CKPT),
    initialize_q_function_from=str(paths.JAX_NASH_Q_FN_CKPT),
)

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(DEFAULT_CONFIG))


  def main(_):
    warnings.simplefilter("error", UserWarning)  # catch jax x64 warnings

    wandb.init(mode='offline')  # avoid network calls during tests

    config = flag_utils.dataclass_from_dict(
        train_nash_policy.Config, CONFIG.value)

    train_nash_policy.train(config)

  app.run(main)
