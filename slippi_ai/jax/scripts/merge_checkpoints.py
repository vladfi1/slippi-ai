#!/usr/bin/env python
"""Merge a policy checkpoint and a VF checkpoint into a train_lib format checkpoint.

The output can be loaded by train_lib (imitation training) or used as the
teacher/restore path for RL training.

Example:
  python slippi_ai/jax/scripts/merge_checkpoints.py \
    --policy=experiments/my_policy/latest.pkl \
    --value_function=experiments/my_vf/latest.pkl \
    --output=experiments/merged.pkl
"""

import pickle

from absl import app, flags

from slippi_ai import flag_utils, paths
from slippi_ai.jax import saving, train_policy, train_vf

POLICY = flags.DEFINE_string(
    'policy', str(paths.JAX_POLICY_CHECKPOINT), 'Path to the policy checkpoint.')
VALUE_FUNCTION = flags.DEFINE_string(
    'value_function', str(paths.JAX_VF_CHECKPOINT), 'Path to the VF checkpoint.')
OUTPUT = flags.DEFINE_string(
    'output', str(paths.JAX_MERGED_CHECKPOINT), 'Path to write the merged checkpoint.')


def main(_):
  policy_state = saving.load_state_from_disk(POLICY.value)
  vf_state = saving.load_state_from_disk(VALUE_FUNCTION.value)

  errors = train_vf.check_compatibility(vf_state, policy_state)
  if errors:
    raise ValueError('Incompatible checkpoints:\n' + '\n'.join(errors))

  merged_state = dict(
      policy=policy_state['state']['policy'],
      policy_optimizer=policy_state['state']['policy_optimizer'],
      value_function=vf_state['state']['value_function'],
      value_optimizer=vf_state['state']['value_optimizer'],
  )

  # Start from the policy config and add the VF network config.
  merged_config = dict(policy_state['config'])
  merged_config['value_function'] = dict(
      separate_network_config=True,
      network=vf_state['config']['network'],
  )
  merged_config['version'] = saving.VERSION

  combined = dict(
      state=merged_state,
      step=0,
      config=merged_config,
      name_map=policy_state['name_map'],
      counters=dict(
          step=0,
          total_frames=0,
          train_time=0.0,
          best_eval_loss=float('inf'),
          train_epoch=0.0,
      ),
  )

  output = OUTPUT.value
  print(f'Saving merged checkpoint to {output}')
  with open(output, 'wb') as f:
    pickle.dump(combined, f)
  print('Done.')


if __name__ == '__main__':
  __spec__ = None  # https://github.com/python/cpython/issues/87115
  app.run(main)
