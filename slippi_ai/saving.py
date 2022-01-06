import pickle
import numpy as np
import tree

from slippi_ai import (
    embed,
    policies,
    networks,
    controller_heads,
    s3_lib,
    data,
)

def build_policy(
  controller_head_config: dict,
  max_action_repeat: int,
  network_config: dict,
  embed_controller: embed.Embedding = embed.embed_controller_discrete,
) -> policies.Policy:
  controller_head_config = dict(
      controller_head_config,
      embed_controller=embed.get_controller_embedding_with_action_repeat(
          embed_controller,
          max_action_repeat))

  return policies.Policy(
      networks.construct_network(**network_config),
      controller_heads.construct(**controller_head_config))

def build_policy_from_sacred(tag: str) -> policies.Policy:
  db = s3_lib.get_sacred_db()
  run = db.runs.find_one({'config.tag': tag}, ['config'])
  if run is None:
    raise ValueError(f"Tag {tag} not found in db.")
  config = run['config']

  return build_policy(
      controller_head_config=config['controller_head'],
      max_action_repeat=config['data']['max_action_repeat'],
      network_config=config['network'],
  )

def get_policy_params_from_s3(tag: str):
  params_key = s3_lib.get_keys(tag).params
  store = s3_lib.get_store()
  obj = store.get(params_key)
  params = pickle.loads(obj)
  return params['policy']

embed_game = embed.make_game_embedding()
dummy_game = embed_game.dummy()
dummy_compressed_game = data.CompressedGame(
    states=dummy_game,
    counts=0,
    rewards=np.float32(0),
)
dummy_loss_batch = tree.map_structure(
    lambda x: np.full((1, 1), x),
    dummy_compressed_game)

def load_policy(tag: str) -> policies.Policy:
  policy = build_policy_from_sacred(tag)
  params = get_policy_params_from_s3(tag)

  initial_state = policy.initial_state(1)
  policy.loss(dummy_loss_batch, initial_state)  # init params
  tree.map_structure(
      lambda var, val: var.assign(val),
      policy.variables, params)

  return policy
