import pickle

import tree
import tensorflow as tf

from slippi_ai import (
    data,
    policies,
    networks,
    controller_heads,
    embed,
    s3_lib,
)

VERSION = 1

def upgrade_config(config: dict):
  """Upgrades a config to the latest version."""
  config = dict(config)  # config may be a Sacred ReadOnlyDict
  version = config.get('version')

  if version is None:
    assert 'policy' not in config
    config['policy'] = dict(
      train_value_head=False,
    )
    config['version'] = 1

  assert config['version'] == VERSION
  return config


def build_policy(
  controller_head_config: dict,
  network_config: dict,
  embed_controller: embed.Embedding = embed.embed_controller_discrete,
  **policy_kwargs,
) -> policies.Policy:
  controller_head_config = dict(
      controller_head_config,
      embed_controller=embed_controller)

  embed_state_action = embed.get_state_action_embedding(
      embed_game=embed.default_embed_game,
      embed_action=embed_controller,
  )

  return policies.Policy(
      networks.construct_network(**network_config),
      controller_heads.construct(**controller_head_config),
      embed_state_action=embed_state_action,
      **policy_kwargs,
  )

def policy_from_config(config: dict) -> policies.Policy:
  # TODO: set embed_controller here
  config = upgrade_config(config)
  return build_policy(
      controller_head_config=config['controller_head'],
      network_config=config['network'],
      **config['policy'],
  )

def load_policy_from_state(state: dict) -> policies.Policy:
  policy = policy_from_config(state['config'])

  # create tensorflow Variables
  dummy_state_action = policy.embed_state_action.dummy(
    [2 + policy.delay, 1])
  dummy_reward = tf.zeros([1 + policy.delay, 1], tf.float32)
  dummy_frames = data.Frames(dummy_state_action, dummy_reward)
  initial_state = policy.initial_state(1)
  policy.loss(dummy_frames, initial_state)

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
