from flax import nnx

from slippi_ai.flag_utils import dataclass_from_dict
from slippi_ai.jax import controller_heads, jax_utils, networks, policies

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

def policy_from_config(config_dict: dict) -> policies.Policy:
  # TODO: Take config dataclasses instead of dictionaries
  config_dict = upgrade_config(config_dict)

  # Avoid circular imports
  from slippi_ai.jax.train_lib import Config

  config = dataclass_from_dict(Config, config_dict)

  rngs = nnx.Rngs(config.seed)

  network = networks.build_embed_network(
      rngs=rngs,
      embed_config=config.embed,
      num_names=config.max_names,
      network_config=config.network,
  )

  # Note: the controller embedding is constructed twice, once for the policy
  # and once for the controller head. The policy uses it to embed the previous
  # action as input, while the controller head uses it to produce the next
  # action.
  return policies.Policy(
      network=network,
      controller_head=controller_heads.construct(
          rngs=rngs,
          input_size=network.output_size,
          embed_controller=config.embed.controller.make_embedding(),
          **config.controller_head,
      ),
      **config_dict['policy'],
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
  policy = policy_from_config(state['config'])

  # assign using saved params
  params = state['state']['policy']
  upgrade_policy(params)
  jax_utils.set_module_state(policy, params)

  return policy
