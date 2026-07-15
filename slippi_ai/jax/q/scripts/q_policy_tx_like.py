#!/usr/bin/env python
"""Test imitation learning training loop - JAX version."""

import dataclasses
import os

os.environ["JAX_COMPILATION_CACHE_DIR"] = "./untracked/jax_cache"

from absl import app, flags
import wandb
import fancyflags as ff

import melee
from slippi_ai import flag_utils, paths
from slippi_ai.jax import saving, train_lib
from slippi_ai.jax.q import train_q_policy, train_q_fn
from slippi_ai.jax.agents import DType

NET_NAME = 'tx_like'

def default_config():
  config = train_q_policy.Config()

  config.data.batch_size = 512
  config.data.unroll_length = 84
  config.data.num_workers = 1
  config.data.balance_characters = True
  config.data.unroll_chunks = 4
  config.learner.learning_rate = 1e-4

  config.learner.num_samples = 3  # 4 total
  config.learner.num_index_samples = 16

  config.learner.sample_policy_dtype = DType.FP16
  config.learner.q_policy_dtype = DType.BF16

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
    del imitation_state

    assert config.initialize_q_function_from is not None
    q_fn_state = saving.load_state_from_disk(config.initialize_q_function_from)
    q_fn_config = flag_utils.dataclass_from_dict(
        train_q_fn.Config, q_fn_state['config'])
    del q_fn_state

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
      config.dataset.allowed_opponents = imitation_config.dataset.allowed_opponents

      if config.tag is None:
        parts = ['qp', char]

        network = imitation_config.network
        assert network['name'] == NET_NAME, f"Expected network name {NET_NAME} but got {network['name']}"
        parts.append(f"d{imitation_config.policy.delay}")

        ops = config.dataset.allowed_opponents
        if ops == char:
          parts.append('ditto')
        else:
          parts.append(f"vs_{ops}")

        n = network[NET_NAME]['num_layers']
        h = network[NET_NAME]['hidden_size']
        parts.append(f"tx{n}x{h}")

        fs = imitation_config.policy.frame_skip
        parts.append(f"rfs{fs}")

        parts.append(f"ns{config.learner.num_samples}")

        idxs = config.learner.num_index_samples
        parts.append(f"is{idxs}")

        iw = config.learner.q_policy_imitation_weight
        if iw > 0:
          parts.append(f"iw{iw:.1e}")

        ps = q_fn_config.q_function.head.epinet.prior_scale
        parts.append(f"ps{ps:.1f}")

        config.tag = "_".join(parts)

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
