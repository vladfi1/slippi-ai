#!/usr/bin/env python
"""Create an initial policy checkpoint in train_policy format.

Saves a freshly-initialized policy + optimizer to disk.  The checkpoint
can be used as a starting point for RL training without requiring a
prior imitation-learning run.

Example:
  python slippi_ai/jax/scripts/create_policy_checkpoint.py \
    --output=experiments/my_policy_ckpt
"""

import dataclasses
import pickle

from absl import app, flags
import fancyflags as ff

from slippi_ai import data as data_lib, flag_utils, nametags, paths
from flax import nnx

from slippi_ai.jax import saving, jax_utils
from slippi_ai.jax import embed as embed_lib
from slippi_ai.jax import policy_learner as learner_lib
from slippi_ai.jax import train_policy


def default_config() -> train_policy.Config:
  config = train_policy.Config()

  config.policy.frame_skip = 3

  config.network['name'] = 'tx_like'
  config.network['tx_like'].update(
      hidden_size=1,
      num_layers=1,
      ffw_multiplier=4,
      recurrent_layer='lstm',
      activation='gelu',
  )

  config.controller_head['name'] = 'autoregressive'
  config.controller_head['autoregressive']['shared'] = True
  config.controller_head['autoregressive']['component']['lstm']['hidden_size'] = 1

  config.embed.player.with_nana = False
  config.embed.items.type = embed_lib.ItemsType.SKIP
  config.embed.controller.default.axis_spacing = 4
  config.embed.controller.default.shoulder_spacing = 2

  return config


CONFIG = ff.DEFINE_dict(
    'config', **flag_utils.get_flags_from_default(default_config()))

OUTPUT = flags.DEFINE_string(
    'output', str(paths.JAX_POLICY_CHECKPOINT), 'Path to write the checkpoint.')
DATASET = flags.DEFINE_string(
    'dataset', str(paths.TOY_DATASET), 'Path to dataset for building name_map.')


def main(_):
  config = flag_utils.dataclass_from_dict(train_policy.Config, CONFIG.value)
  config = dataclasses.replace(config, version=saving.VERSION)

  dataset_config = data_lib.DatasetConfig(dataset_path=DATASET.value)
  train_replays, test_replays = data_lib.train_test_split(dataset_config)
  names = []
  for replay in train_replays + test_replays:
    names.append(replay.meta.p0.name)
    names.append(replay.meta.p1.name)
  name_map = nametags.name_map_from_entries(names, config.max_names)

  rngs = nnx.Rngs(config.seed)

  policy = saving.policy_from_configs(
      network_config=config.network,
      controller_head_config=config.controller_head,
      embed_config=config.embed,
      policy_config=config.policy,
      max_name=config.max_names,
      rngs=rngs,
  )

  learner_config = dataclasses.replace(config.learner, use_shard_map=False)
  learner = learner_lib.PolicyLearner(
      policy=policy,
      config=learner_config,
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
  print(f'Saving policy checkpoint to {output}')
  with open(output, 'wb') as f:
    pickle.dump(combined, f)
  print('Done.')


if __name__ == '__main__':
  app.run(main)
