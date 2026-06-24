#!/usr/bin/env python
"""Test comparing two q-functions on policy samples - JAX version.

Loads the demo policy and two demo q-functions, then reports how much the two
q-functions agree on the relative ordering of actions sampled from the policy.
"""

from absl import app
import fancyflags as ff

from slippi_ai import paths, flag_utils
from slippi_ai import data as data_lib
from slippi_ai.jax.q import compare_q_functions

DEFAULT_CONFIG = compare_q_functions.Config(
    dataset=data_lib.DatasetConfig(
        data_dir=str(paths.TOY_DATA_DIR),
        meta_path=str(paths.TOY_META_PATH),
    ),
    data=data_lib.DataConfig(
        balance_characters=True,
        batch_size=2,
        unroll_length=6,
    ),
    sample_policy=str(paths.JAX_POLICY_CHECKPOINT),
    # The only standard demo q-function; comparing it against itself should
    # yield perfect ordering agreement (a sanity check on the pipeline).
    q_function_a=str(paths.JAX_Q_FN_CKPT),
    q_function_b=str(paths.JAX_Q_FN_CKPT),
    num_samples=4,
    num_steps=3,
)

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(DEFAULT_CONFIG))

  def main(_):
    config = flag_utils.dataclass_from_dict(
        compare_q_functions.Config, CONFIG.value)
    compare_q_functions.compare(config)

  app.run(main)
