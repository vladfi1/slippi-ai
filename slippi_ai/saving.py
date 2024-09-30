import dataclasses
import pickle

from absl import logging
import tree
import tensorflow as tf

from slippi_ai import (
    data,
    embed,
    policies,
    networks,
    controller_heads,
    embed,
    s3_lib,
)

VERSION = 3

def upgrade_config(config: dict):
  """Upgrades a config to the latest version."""

  if config.get('version') is None:
    assert 'policy' not in config
    config['policy'] = dict(
      train_value_head=False,
    )
    config['version'] = 1
    logging.warning('Upgraded config to version 1')

  if config['version'] == 1:
    if 'value_function' not in config:
      config['value_function'] = dict(
        train_separate_network=False,
      )

    config['version'] = 2
    logging.warning('Upgraded config version 1 -> 2')

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
        )
    )
    config['embed'] = dataclasses.asdict(old_embed_config)
    config['version'] = 3
    logging.warning('Upgraded config version 2 -> 3')

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
          player_config=config['embed']['player']),
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

def load_state_from_s3(tag: str) -> dict:
  key = s3_lib.get_keys(tag).combined
  store = s3_lib.get_store()
  obj = store.get(key)
  return pickle.loads(obj)

def load_policy_from_s3(tag: str) -> policies.Policy:
  state = load_state_from_s3(tag)
  return load_policy_from_state(state)

def load_state_from_disk(path: str) -> dict:
  with open(path, 'rb') as f:
    return pickle.load(f)

def load_policy_from_disk(path: str) -> policies.Policy:
  state = load_state_from_disk(path)
  return load_policy_from_state(state)
