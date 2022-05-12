import pickle
import tree

from slippi_ai import (
    policies,
    s3_lib,
    train_lib,
)

def policy_from_config(config: dict) -> policies.Policy:
  # TODO: set embed_controller here
  return train_lib.build_policy(
      controller_head_config=config['controller_head'],
      max_action_repeat=config['data']['max_action_repeat'],
      network_config=config['network'],
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
  dummy_state_action = policy.embed_state_action.dummy([1])
  initial_state = policy.initial_state(1)
  policy.sample(dummy_state_action, initial_state)

  # assign using saved params
  params = state['state']['policy']
  tree.map_structure(
      lambda var, val: var.assign(val),
      policy.variables, params)

  return policy

def load_policy_from_s3(tag: str) -> policies.Policy:
  key = s3_lib.get_keys(tag).combined
  store = s3_lib.get_store()
  obj = store.get(key)
  state = pickle.loads(obj)
  return load_policy_from_state(state)

def load_policy_from_disk(path: str) -> policies.Policy:
  with open(path, 'rb') as f:
    state = pickle.load(f)
  return load_policy_from_state(state)
