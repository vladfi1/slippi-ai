"""
Doubles as a script for producing agent outputs and for testing them.
"""
from absl import app, flags

from slippi_ai.unroll_agent import test_or_save_outputs
from slippi_ai import paths

FLAGS = flags.FLAGS

GAME_PATH = next(paths.TOY_DATA_DIR.iterdir())
OUTPUT_PATH = paths.DATA_PATH / 'demo_unroll_output.pkl'

flags.DEFINE_string('model', str(paths.DEMO_CHECKPOINT), 'Path to the model.')
flags.DEFINE_string('input', str(GAME_PATH), 'Path to the input file.')
flags.DEFINE_string('output', str(OUTPUT_PATH), 'Path to the output file.')
flags.DEFINE_bool('overwrite', False, 'Whether to overwrite the output file.')
flags.DEFINE_integer('subsample', 100, 'Subsample the outputs across the time dimension.')

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
