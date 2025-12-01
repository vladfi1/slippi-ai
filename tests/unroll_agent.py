"""
Doubles as a script for producing agent outputs and for testing them.
"""

from absl import app, flags

from slippi_ai import unroll_agent
from slippi_ai import paths

FLAGS = flags.FLAGS

OUTPUT_PATH = paths.AGENT_OUTPUTS_DIR / 'demo.pkl'

flags.DEFINE_string('model', str(paths.DEMO_CHECKPOINT), 'Path to the model.')
flags.DEFINE_string('input', str(paths.TOY_DATA_DIR), 'Path to the input file directory.')
flags.DEFINE_string('output', str(OUTPUT_PATH), 'Path to the output file.')
flags.DEFINE_bool('overwrite', False, 'Whether to overwrite the output file.')
flags.DEFINE_integer('subsample', 100, 'Subsample the outputs across the time dimension.')

def main(_):
  unroll_agent.test_or_save_outputs(
      model_path=FLAGS.model,
      input_dir=FLAGS.input,
      output_path=FLAGS.output,
      overwrite=FLAGS.overwrite,
      subsample=FLAGS.subsample,
  )

if __name__ == '__main__':
  app.run(main)
