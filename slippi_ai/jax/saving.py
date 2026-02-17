import pickle

import numpy as np
import jax
from flax import nnx

from slippi_ai.flag_utils import dataclass_from_dict
from slippi_ai.jax import controller_heads, embed, jax_utils, networks, policies

VERSION = 2

def add_missing_embed(network_config: dict):
  if 'embed' not in network_config:
    network_config['embed'] = {
        'name': 'simple',
        'simple': {},
    }

def upgrade_config(config: dict):
  """Upgrades a config to the latest version."""

  version = config['version']
  if not isinstance(version, int):
    raise ValueError(f"Config version must be an int, got {version}")

  if version == 1:
    add_missing_embed(config['network'])
    add_missing_embed(config['value_function']['network'])

    if 'train_value_head' in config['policy']:
      del config['policy']['train_value_head']

    version = 2

  assert version == VERSION
  config['version'] = version
  return config

def policy_from_configs(
    network_config: dict,
    controller_head_config: dict,
    embed_config: embed.EmbedConfig,
    policy_config: policies.PolicyConfig,
    max_name: int,
    rngs: nnx.Rngs,
) -> policies.Policy:
  """Build a Policy from configuration."""
  embed_controller = embed_config.controller.make_embedding()

  network = networks.build_embed_network(
      rngs=rngs,
      embed_config=embed_config,
      num_names=max_name,
      network_config=network_config,
  )

  controller_head = controller_heads.construct(
      rngs=rngs,
      input_size=network.output_size,
      embed_controller=embed_controller,
      **controller_head_config,
  )

  policy = policies.Policy(
      network=network,
      controller_head=controller_head,
      delay=policy_config.delay,
  )

  return policy

def policy_from_config_dict(config_dict: dict) -> policies.Policy:
  """Load a policy from a config dict.

  Allows any training script (RL, Q-learning, etc.) to be loadable into a
  policy, so long as the config dict has the same structure as imitation.
  """
  config_dict = upgrade_config(config_dict)

  embed_config = dataclass_from_dict(embed.EmbedConfig, config_dict['embed'])
  poliicy_config = dataclass_from_dict(policies.PolicyConfig, config_dict['policy'])
  rngs = nnx.Rngs(config_dict['seed'])

  return policy_from_configs(
      network_config=config_dict['network'],
      controller_head_config=config_dict['controller_head'],
      embed_config=embed_config,
      max_name=config_dict['max_names'],
      policy_config=poliicy_config,
      rngs=rngs,
  )

# Take into account renaming of submodules when loading state dicts.
_submodule_mappings = {
    'controller_head': '_controller_head',
}
_old_keys = ['value_head']

def upgrade_policy(params: dict):
  for old_name, new_name in _submodule_mappings.items():
    if old_name in params:
      assert new_name not in params, f"Both {old_name} and {new_name} found in params, cannot upgrade."
      params[new_name] = params.pop(old_name)

  for old_key in _old_keys:
    if old_key in params:
      del params[old_key]

def load_policy_from_state(state: dict) -> policies.Policy:
  policy = policy_from_config_dict(state['config'])

  # assign using saved params
  params = state['state']['policy']
  upgrade_policy(params)
  jax_utils.set_module_state(policy, params)

  return policy

def load_state_from_disk(path: str) -> dict:
  with open(path, 'rb') as f:
    state = pickle.load(f)

  # Params used to be stored as jax arrays. This causes issues when building
  # multiple modules from the same state and using buffer donation, as jax will
  # consider both modules "deleted" after the first one is used and donated.
  state['state'] = jax.tree.map(np.asarray, state['state'])

  return state