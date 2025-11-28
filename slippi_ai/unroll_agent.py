import hashlib
import os
import pickle
from typing import Optional

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
    input_path: str,
    subsample: int,
    model_path: Optional[str] = None,
    policy: Optional[policies.Policy] = None,
) -> policies.DistanceOutputs:

  if policy is None:
    if model_path is None:
      raise ValueError('Either model_path or policy must be provided.')
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

def check_arrays_with_path(path: tuple, xs: np.ndarray, ys: np.ndarray):
  isclose = np.isclose(xs, ys, atol=1e-5, rtol=1e-5)

  # Handle multi-dimensional arrays like logits.
  rank = isclose.ndim
  if rank > 1:
    isclose = np.all(isclose, axis=tuple(range(1, rank)))

  idxs = np.arange(len(xs))[~isclose]
  if len(idxs) > 0:
    idx = idxs[0].item()
    x = xs[idx]
    y = ys[idx]
    raise ValueError(f'Output mismatch at {path} ({idx}): {x} != {y}')

def check_outputs(
    outputs: policies.DistanceOutputs,
    existing_outputs: policies.DistanceOutputs,
):
  tree.map_structure_with_path(
      check_arrays_with_path, outputs, existing_outputs)

def md5(file_path: str) -> str:
  """Compute the MD5 hash of a file."""
  hasher = hashlib.md5()
  with open(file_path, 'rb') as f:
    while True:
      data = f.read(65536)
      if not data:
        break
      hasher.update(data)
  return hasher.hexdigest()


def test_or_save_outputs(
    model_path: str,
    input_dir: str,
    output_path: str,
    subsample: int = 100,
    overwrite: bool = False,
):
  """Test or save agent outputs for all input files in a directory.

  The outputs are saved to a pickle file at `output_path`. If the file
  already exists and `overwrite` is False, the existing outputs are
  compared against the newly computed outputs. If there is a mismatch,
  a ValueError is raised.

  The outputs are stored in a dictionary mapping input file names to
  their corresponding outputs, along with the model hash used to
  generate them.
  """
  inputs = os.listdir(input_dir)

  model_hash = md5(model_path)

  if os.path.exists(output_path) and not overwrite:
    with open(output_path, 'rb') as f:
      outputs = pickle.load(f)

      existing_model_hash = outputs['model_hash']
      if existing_model_hash != model_hash:
        raise ValueError(
            f'Model hash mismatch: existing {existing_model_hash} vs new {model_hash}'
        )
  else:
    outputs = dict(
        model_hash=model_hash,
    )

  # Load policy once for all inputs.
  policy = saving.load_policy_from_disk(model_path)

  updated = False
  for input_file in inputs:
    overwrite = overwrite or (input_file not in outputs)

    if not overwrite:
      results = outputs[input_file]
      subsample = results['subsample']
    else:
      subsample = subsample

    unroll_output = unroll(
        policy=policy,
        input_path=os.path.join(input_dir, input_file),
        subsample=subsample,
    )

    if overwrite:
      outputs[input_file] = dict(
          subsample=subsample,
          outputs=unroll_output,
      )
      updated = True
    else:
      check_outputs(
          unroll_output,
          results['outputs'],
      )

  if updated:
    with open(output_path, 'wb') as f:
      pickle.dump(outputs, f)
    print(f'Updated outputs saved to {output_path}')
