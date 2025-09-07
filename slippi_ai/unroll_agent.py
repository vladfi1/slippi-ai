import os
import pickle

import numpy as np
import tree

from slippi_ai import (
  data,
  policies,
  reward,
  saving,
  utils,
)

def get_frames(path: str) -> data.Frames:
  game = data.read_table(path, compressed=True)
  game_length = len(game.stage)
  rewards = reward.compute_rewards(game)
  name_codes = np.full([game_length], 0, np.int32)
  state_action = data.StateAction(game, game.p0.controller, name_codes)
  is_resetting = np.full([game_length], False)
  is_resetting[0] = True
  return data.Frames(
      state_action=state_action,
      reward=rewards,
      is_resetting=is_resetting)


def unroll(
    model_path: str,
    input_path: str,
    subsample: int,
) -> policies.DistanceOutputs:
  policy = saving.load_policy_from_disk(model_path)
  frames = get_frames(input_path)

  frames = frames._replace(
      state_action=policy.embed_state_action.from_state(
          frames.state_action))

  # Add batch dimension
  frames = utils.map_nt(lambda x: np.expand_dims(x, 1), frames)

  initial_state = policy.initial_state(1)
  outputs = policy.unroll(frames, initial_state)

  distances = utils.map_nt(
    lambda x: np.squeeze(x.numpy(), 1)[::subsample],
    outputs.distances)

  return distances

def test_or_save_outputs(
    model_path: str,
    input_path: str,
    output_path: str,
    subsample: int = 100,
    overwrite: bool = False,
):

  if overwrite or not os.path.exists(output_path):
    outputs = unroll(model_path, input_path, subsample)
    result = dict(
        subsample=subsample,
        outputs=outputs,
    )

    with open(output_path, 'wb') as f:
      pickle.dump(result, f)
    print(f'Saved outputs to {output_path}')

    return

  print(f'Loading existing outputs from {output_path}')

  with open(output_path, 'rb') as f:
    existing_outputs = pickle.load(f)
    existing_subsample = existing_outputs['subsample']
    existing_outputs = existing_outputs['outputs']

  outputs = unroll(model_path, input_path, existing_subsample)

  allclose = utils.map_nt(
      lambda x, y: np.allclose(x, y, atol=1e-5, rtol=1e-5),
      outputs, existing_outputs
  )

  for path, close in tree.flatten_with_path(allclose):
    if not close:
      raise ValueError(f'Output mismatch at {path}')

  print('Outputs match.')
