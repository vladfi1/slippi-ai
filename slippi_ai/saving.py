import dataclasses
import pickle

import tree

from slippi_ai import (
    embed,
    observations,
    policies,
    networks,
    controller_heads,
    embed,
    data,
)
from slippi_ai.flag_utils import dataclass_from_dict

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
  from slippi_ai.train_lib import Config  # avoid circular import
  config_dc = dataclass_from_dict(Config, config)
  config = dataclasses.asdict(config_dc)

  assert config['version'] == VERSION
  return config


def build_policy(
  controller_head_config: dict,
  network_config: dict,
  num_names: int,
  embed_controller: embed.Embedding,
  embed_game: embed.Embedding,
  **policy_kwargs,
) -> policies.Policy:
  controller_head_config = dict(
      controller_head_config,
      embed_controller=embed_controller)

  return policies.Policy(
      networks.construct_network(**network_config),
      controller_heads.construct(**controller_head_config),
      embed_game=embed_game,
      num_names=num_names,
      **policy_kwargs,
  )

def policy_from_config(config: dict) -> policies.Policy:
  # TODO: Take config dataclasses instead of dictionaries
  config = upgrade_config(config)

  return build_policy(
      controller_head_config=config['controller_head'],
      network_config=config['network'],
      num_names=config['max_names'],
      embed_controller=embed.get_controller_embedding(
          **config['embed']['controller']),
      embed_game=embed.make_game_embedding(
          player_config=config['embed']['player'],
          with_randall=config['embed']['with_randall'],
          with_fod=config['embed']['with_fod'],
          items_config=dataclass_from_dict(
              embed.ItemsConfig, config['embed']['items']),
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
