#!/usr/bin/env python
"""Evaluate a Nash Q-function."""

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

def default_config():
  config = train_nash_policy.Config()

  config.runtime.max_step = 0
  config.runtime.eval_at_start = True
  config.runtime.verbose_eval = True
  config.runtime.save_best = False

  config.data.batch_size = 256
  config.data.unroll_length = 84
  config.data.damage_ratio = 0.01
  config.data.num_workers = 1
  config.data.unroll_chunks = 4

  config.learner.num_samples = 3
  config.learner.sample_policy_dtype = DType.FP16
  config.learner.nash_policy_dtype = DType.FP16

  config.dataset.mirror = False
  config.dataset.data_dir = os.environ.get("DATA_DIR")
  config.dataset.meta_path = os.environ.get("META_PATH")

  return config

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  TOY_DATA = flags.DEFINE_bool('toy_data', False, 'Use toy data for quick testing')

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(default_config()))

  # passed to wandb.init
  WANDB = ff.DEFINE_dict(
      'wandb',
      project=ff.String('slippi-ai'),
      mode=ff.Enum('online', ['online', 'offline', 'disabled']),
      group=ff.String('nash-eval'),
      name=ff.String(None),
      notes=ff.String(None),
      dir=ff.String(None, 'directory to save logs'),
  )

  def main(_):
    config = flag_utils.dataclass_from_dict(train_nash_policy.Config, CONFIG.value)

    assert config.initialize_policies_from is not None
    imitation_state = saving.load_state_from_disk(config.initialize_policies_from)
    imitation_config = flag_utils.dataclass_from_dict(
        train_lib.Config,
        saving.upgrade_config(imitation_state['config']))
    del imitation_state

    assert config.initialize_q_function_from is not None

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
        q_fn_name = os.path.basename(config.initialize_q_function_from)
        if q_fn_name == 'latest.pkl':
          q_fn_name = os.path.basename(os.path.dirname(config.initialize_q_function_from))
        config.tag = f"{q_fn_name}_ns{config.learner.num_samples}_eval"

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
