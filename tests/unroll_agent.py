import os
import pickle

from absl import app, flags
import numpy as np

from slippi_ai import (
  data,
  paths,
  policies,
  reward,
  saving,
  utils,
)

FLAGS = flags.FLAGS

GAME_PATH = next(paths.TOY_DATA_DIR.iterdir())
OUTPUT_PATH = paths.DATA_PATH / 'demo_unroll_output.pkl'

flags.DEFINE_string('model', str(paths.DEMO_CHECKPOINT), 'Path to the model.')
flags.DEFINE_string('input', str(GAME_PATH), 'Path to the input file.')
flags.DEFINE_string('output', str(OUTPUT_PATH), 'Path to the output file.')
flags.DEFINE_bool('overwrite', False, 'Whether to overwrite the output file.')
flags.DEFINE_integer('subsample', 100, 'Subsample the outputs across the time dimension.')

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
    subsample: int,
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

  errors = utils.check_same_structure(
      outputs, existing_outputs, equal=True)
  if errors:
    for path, message in errors:
      print(f'Error in {path}: {message}')
    raise ValueError(f'Output files do not match.')

  print('Output files match.')

def main(_):
  test_or_save_outputs(
      model_path=FLAGS.model,
      input_path=FLAGS.input,
      output_path=FLAGS.output,
      overwrite=FLAGS.overwrite,
      subsample=FLAGS.subsample,
  )

if __name__ == '__main__':
  app.run(main)
