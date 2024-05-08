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

VERSION = 2

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

  assert config['version'] == VERSION
  return config


def build_policy(
  controller_head_config: dict,
  network_config: dict,
  num_names: int,
  embed_controller: embed.Embedding = embed.embed_controller_discrete,
  **policy_kwargs,
) -> policies.Policy:
  controller_head_config = dict(
      controller_head_config,
      embed_controller=embed_controller)

  embed_state_action = embed.get_state_action_embedding(
      embed_game=embed.default_embed_game,
      embed_action=embed_controller,
      num_names=num_names,
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
      num_names=config['max_names'],
      **config['policy'],
  )

def init_policy_vars(policy: policies.Policy):
  dummy_state_action = policy.embed_state_action.dummy(
    [2 + policy.delay, 1])
  dummy_reward = tf.zeros([1 + policy.delay, 1], tf.float32)
  dummy_frames = data.Frames(dummy_state_action, dummy_reward)
  initial_state = policy.initial_state(1)

  # loss initializes value function, which isn't used during sampling
  # but is needed for setting the policy vars
  policy.imitation_loss(dummy_frames, initial_state)

def load_policy_from_state(state: dict) -> policies.Policy:
  policy = policy_from_config(state['config'])
  init_policy_vars(policy)

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
