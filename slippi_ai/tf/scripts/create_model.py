"""Creates and saves models to use for testing."""

import dataclasses
import pickle
import os

from absl import app, flags

from slippi_ai import (
  flag_utils,
  saving,
  observations as obs_lib,
  data as data_lib,
)
from slippi_ai.tf import controller_heads, embed as embed_lib, networks, policies

def default_network_config():
  network_config = networks.default_config()
  network_config['name'] = 'tx_like'
  network_config['tx_like']['num_layers'] = 3
  network_config['tx_like']['hidden_size'] = 512
  network_config['tx_like']['ffw_multiplier'] = 2
  return network_config

def controller_head_config():
  config = controller_heads.default_config()
  config['name'] = 'autoregressive'
  config['autoregressive']['component_depth'] = 2
  config['autoregressive']['residual_size'] = 128
  return config

_field = lambda f: dataclasses.field(default_factory=f)

def policy_config():
  return policies.PolicyConfig(
      train_value_head=False,
      delay=21,
  )

@dataclasses.dataclass
class Config:
  network: dict = _field(default_network_config)
  controller_head: dict = _field(controller_head_config)
  embed: embed_lib.EmbedConfig = _field(embed_lib.EmbedConfig)
  observation: obs_lib.ObservationConfig = _field(obs_lib.ObservationConfig)
  policy: policies.PolicyConfig = _field(policy_config)
  max_names: int = 16
  dataset: data_lib.DatasetConfig = _field(data_lib.DatasetConfig)
  version: int = saving.VERSION

CONFIG = flag_utils.define_dict_from_dataclass('config', Config)

SAVE_DIR = flags.DEFINE_string('save_dir', 'test_models', 'Directory to save the model')
NAME = flags.DEFINE_string('name', None, 'Model name')

yes_no = {
  True: 'with-',
  False: 'no-',
}

def name_from_config(config: Config) -> str:
  assert config.network['name'] == 'tx_like'
  n = config.network['tx_like']['num_layers']
  h = config.network['tx_like']['hidden_size']

  assert config.controller_head['name'] == 'autoregressive'
  return '_'.join([
      f'tx_{n}x{h}',
      f'{yes_no[config.embed.player.with_nana]}nana',
      f'{config.embed.items.type.name.lower()}-items',
  ])


def main(_):
  config = flag_utils.dataclass_from_dict(Config, CONFIG.value)

  policy = saving.policy_from_config(CONFIG.value)
  policy.initialize_variables()

  name = NAME.value or name_from_config(config)

  state = {
      'config': CONFIG.value,
      'state': {'policy': policy.variables},
      'name_map': {'': 0},
  }

  save_path = os.path.join(SAVE_DIR.value, name)
  print(f'Saving model to {save_path}')
  os.makedirs(SAVE_DIR.value, exist_ok=True)
  with open(save_path, 'wb') as f:
    pickle.dump(state, f)

if __name__ == '__main__':
  app.run(main)
