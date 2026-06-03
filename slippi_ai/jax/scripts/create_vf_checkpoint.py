#!/usr/bin/env python
"""Create an initial value function checkpoint in train_vf format.

Saves a freshly-initialized value function + optimizer to disk.  The
checkpoint can be used as a starting point for RL training without
requiring a prior standalone VF training run.

Example:
  python slippi_ai/jax/scripts/create_vf_checkpoint.py \
    --output=experiments/my_vf_ckpt
"""

import dataclasses
import pickle

from absl import app, flags
import fancyflags as ff
from flax import nnx

from slippi_ai import data as data_lib, flag_utils, nametags, paths
from slippi_ai.jax import saving, jax_utils
from slippi_ai.jax import value_function as vf_lib
from slippi_ai.jax import vf_learner as learner_lib
from slippi_ai.jax import train_vf, train_policy


def default_config() -> train_vf.Config:
  config = train_vf.Config()

  config.compatible_policy = str(paths.JAX_POLICY_CHECKPOINT)

  config.network['name'] = 'tx_like'
  config.network['tx_like'].update(
      hidden_size=1,
      num_layers=1,
      ffw_multiplier=4,
      recurrent_layer='lstm',
      activation='gelu',
  )

  return config


CONFIG = ff.DEFINE_dict(
    'config', **flag_utils.get_flags_from_default(default_config()))

OUTPUT = flags.DEFINE_string(
    'output', str(paths.JAX_VF_CHECKPOINT), 'Path to write the checkpoint.')
DATASET = flags.DEFINE_string(
    'dataset', str(paths.TOY_DATASET), 'Path to dataset for building name_map.')


def main(_):
  config = flag_utils.dataclass_from_dict(train_vf.Config, CONFIG.value)

  if config.compatible_policy:
    policy_state = saving.load_state_from_disk(config.compatible_policy)
    policy_config = flag_utils.dataclass_from_dict(
        train_policy.Config, policy_state['config'])
    config.make_compatible_with(policy_config)
    name_map = policy_state['name_map']
  else:
    dataset_config = data_lib.DatasetConfig(dataset_path=DATASET.value)
    train_replays, test_replays = data_lib.train_test_split(dataset_config)
    names = []
    for replay in train_replays + test_replays:
      names.append(replay.meta.p0.name)
      names.append(replay.meta.p1.name)
    name_map = nametags.name_map_from_entries(names, config.max_names)

  rngs = nnx.Rngs(config.seed)
  mesh = jax_utils.get_mesh()
  data_sharding = jax_utils.data_sharding(mesh)

  value_function = vf_lib.ValueFunction(
      rngs=rngs,
      network_config=config.network,
      num_names=config.max_names,
      embed_config=config.embed,
  )

  learner = learner_lib.VFLearner(
      value_function=value_function,
      config=config.learner,
      mesh=mesh,
      data_sharding=data_sharding,
  )

  state = jax_utils.get_module_state(learner)

  combined = dict(
      state=state,
      step=0,
      config=dataclasses.asdict(config),
      name_map=name_map,
      counters=dict(
          step=0,
          total_frames=0,
          train_time=0.0,
          best_eval_loss=float('inf'),
          train_epoch=0.0,
      ),
  )

  output = OUTPUT.value
  print(f'Saving VF checkpoint to {output}')
  with open(output, 'wb') as f:
    pickle.dump(combined, f)
  print('Done.')


if __name__ == '__main__':
  __spec__ = None  # https://github.com/python/cpython/issues/87115
  app.run(main)
