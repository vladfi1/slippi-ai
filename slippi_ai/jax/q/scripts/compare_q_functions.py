#!/usr/bin/env python
"""Launch script for comparing two q-functions on policy samples - JAX version.

Samples actions from a policy and measures how much two q-functions agree on
the relative ordering of those actions (Spearman, Kendall's tau, top-1).

Example:
  python -m slippi_ai.jax.q.scripts.compare_q_functions \
      --config.sample_policy=/path/to/policy \
      --config.q_function_a=/path/to/q_fn_a \
      --config.q_function_b=/path/to/q_fn_b \
      --config.dataset.data_dir=/path/to/games \
      --config.dataset.meta_path=/path/to/meta.json
"""

import os

os.environ["JAX_COMPILATION_CACHE_DIR"] = "./untracked/jax_cache"

from absl import app, flags
import fancyflags as ff

from slippi_ai import flag_utils, paths
from slippi_ai.jax.q import compare_q_functions


def default_config():
  config = compare_q_functions.Config()
  config.data.batch_size = 256
  config.data.unroll_length = 84
  config.data.num_workers = 2
  config.num_samples = 8
  config.num_steps = 50
  config.dataset.data_dir = os.environ.get("DATA_DIR")
  config.dataset.meta_path = os.environ.get("META_PATH")
  return config


if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  TOY_DATA = flags.DEFINE_bool(
      'toy_data', False, 'Use toy data for quick testing')

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(default_config()))

  def main(_):
    config = flag_utils.dataclass_from_dict(
        compare_q_functions.Config, CONFIG.value)

    if TOY_DATA.value:
      config.dataset.data_dir = str(paths.TOY_DATA_DIR)
      config.dataset.meta_path = str(paths.TOY_META_PATH)

    compare_q_functions.compare(config)

  app.run(main)
