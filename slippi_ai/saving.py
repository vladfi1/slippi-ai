import pickle
import tree

from slippi_ai import (
    policies,
    networks,
    controller_heads,
    embed,
    s3_lib,
    train_lib,
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
  max_action_repeat: int,
  network_config: dict,
  embed_controller: embed.Embedding = embed.embed_controller_discrete,
  **policy_kwargs,
) -> policies.Policy:
  embed_controller_with_repeat = embed.get_controller_embedding_with_action_repeat(
      embed_controller,
      max_action_repeat)

  controller_head_config = dict(
      controller_head_config,
      embed_controller=embed_controller_with_repeat)

  embed_state_action = embed.get_state_action_embedding(
      embed_game=embed.default_embed_game,
      embed_action=embed_controller_with_repeat,
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
      max_action_repeat=config['data']['max_action_repeat'],
      network_config=config['network'],
      **config['policy'],
  )

def build_policy_from_sacred(tag: str) -> policies.Policy:
  db = s3_lib.get_sacred_db()
  run = db.runs.find_one({'config.tag': tag}, ['config'])
  if run is None:
    raise ValueError(f"Tag {tag} not found in db.")
  return policy_from_config(run['config'])

def load_policy_from_state(state: dict) -> policies.Policy:
  policy = policy_from_config(state['config'])

  # create tensorflow Variables
  dummy_state_action = policy.embed_state_action.dummy([1, 1])
  initial_state = policy.initial_state(1)
  policy.loss(dummy_state_action, initial_state)

  # assign using saved params
  params = state['state']['policy']
  tree.map_structure(
      lambda var, val: var.assign(val),
      policy.variables, params)

  return policy

def get_state_from_s3(tag: str) -> dict:
  key = s3_lib.get_keys(tag).combined
  store = s3_lib.get_store()
  obj = store.get(key)
  return pickle.loads(obj)

def load_policy_from_s3(tag: str) -> policies.Policy:
  state = get_state_from_s3(tag)
  return load_policy_from_state(state)

def load_policy_from_disk(path: str) -> policies.Policy:
  with open(path, 'rb') as f:
    state = pickle.load(f)
  return load_policy_from_state(state)
