import dataclasses
import pickle

import tree

from slippi_ai import observations
from slippi_ai.flag_utils import dataclass_from_dict
from slippi_ai.tf import controller_heads, embed, networks, policies

# v2: Added value function config
# v3: Added embed config
# v4: Added observation config
# v5: Added randall, fod, items to embed config

VERSION = 5

def upgrade_config(config: dict):
  """Upgrades a config to the latest version."""

  if config.get('version') is None:
    assert 'policy' not in config
    config['policy'] = dict(
      train_value_head=False,
    )
    config['version'] = 1

  if config['version'] == 1:
    if 'value_function' not in config:
      config['value_function'] = dict(
        train_separate_network=False,
      )

    config['version'] = 2

  if config['version'] == 2:
    assert 'embed' not in config
    old_embed_config = embed.EmbedConfig(
        player=embed.PlayerConfig(
            xy_scale=0.05,
            shield_scale=0.01,
            speed_scale=0.5,
            with_speeds=False,
            with_controller=False,
        ),
        controller=embed.ControllerConfig(
            axis_spacing=16,
            shoulder_spacing=4,
        ),
    )
    config['embed'] = dataclasses.asdict(old_embed_config)
    config['version'] = 3

  if config['version'] == 3:
    assert 'observation' not in config
    config['observation'] = dataclasses.asdict(
        observations.NULL_OBSERVATION_CONFIG)
    config['version'] = 4

  if config['version'] == 4:
    old_items_config = embed.ItemsConfig(type=embed.ItemsType.SKIP)
    config['embed'].update(
        with_randall=False,
        with_fod=False,
        items=dataclasses.asdict(old_items_config),
    )
    config['embed']['player'].update(
        with_nana=False,
        legacy_jumps_left=True,
    )
    config['version'] = 5

  # Everything else is handled by the defaults in train_lib.Config
  from slippi_ai.tf.train_lib import Config  # avoid circular import
  config_dc = dataclass_from_dict(Config, config)
  config = dataclasses.asdict(config_dc)

  assert config['version'] == VERSION
  return config

def policy_from_config(config: dict) -> policies.Policy:
  # TODO: Take config dataclasses instead of dictionaries
  config = upgrade_config(config)

  embed_config = dataclass_from_dict(embed.EmbedConfig, config['embed'])

  # Note: the controller embedding is constructed twice, once for the policy
  # and once for the controller head. The policy uses it to embed the previous
  # action as input, while the controller head uses it to produce the next
  # action.
  return policies.Policy(
      network=networks.build_embed_network(
          embed_config=embed_config,
          num_names=config['max_names'],
          network_config=config['network'],
      ),
      controller_head=controller_heads.construct(
          embed_controller=embed_config.controller.make_embedding(),
          **config['controller_head'],
      ),
      **config['policy'],
  )

def load_policy_from_state(state: dict) -> policies.Policy:
  policy = policy_from_config(state['config'])
  policy.initialize_variables()

  # assign using saved params
  params = state['state']['policy']
  tree.map_structure(
      lambda var, val: var.assign(val),
      policy.variables, params)

  return policy

def load_state_from_disk(path: str) -> dict:
  with open(path, 'rb') as f:
    return pickle.load(f)

def load_policy_from_disk(path: str) -> policies.Policy:
  state = load_state_from_disk(path)
  return load_policy_from_state(state)
