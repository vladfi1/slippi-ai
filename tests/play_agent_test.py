"""Tests that agents can be loaded and run with only the play dependencies.

Run this from an install with just the model platform extras (e.g.
`pip install .[jax,tf]`) to check that loading and running an agent doesn't
depend on training-only packages such as wandb. With --no_jax_tf, also checks
that jax and tensorflow aren't imported, e.g. for exported ONNX models.
"""

import os
import sys

from absl import app, flags
import numpy as np

from slippi_ai import eval_lib, paths, saving, utils
from slippi_ai.types import Game, reify_tuple_type

flags.DEFINE_list(
    'models', ['demo', 'rl_demo', 'jax_demo'],
    f'Paths, or names of checkpoints in {paths.CHECKPOINTS_DIR}, to test.')
flags.DEFINE_integer('steps', 10, 'Number of agent steps to run.')
flags.DEFINE_boolean('no_jax_tf', False, 'Check that jax and tensorflow are not imported.')

FLAGS = flags.FLAGS

TRAINING_ONLY_MODULES = ['wandb', 'pandas', 'peppi_py', 'py7zr', 'fsspec']

def run_agent(path: str, steps: int):
  state = saving.load_state_from_disk(path)
  name = next(iter(state['name_map']))
  agent = eval_lib.build_delayed_agent(
      state, console_delay=0, name=name, batch_size=1)
  game = utils.map_nt(
      lambda dtype: np.zeros([1], dtype=dtype), reify_tuple_type(Game))
  needs_reset = np.array([True])
  for _ in range(steps):
    agent.step(game, needs_reset)
    needs_reset[:] = False

def main(_):
  for model in FLAGS.models:
    path = model if os.path.exists(model) else str(paths.CHECKPOINTS_DIR / model)
    run_agent(path, FLAGS.steps)
    print(f'Ran {model}')

  forbidden = TRAINING_ONLY_MODULES
  if FLAGS.no_jax_tf:
    forbidden = forbidden + ['jax', 'tensorflow']
  loaded = [m for m in forbidden if m in sys.modules]
  assert not loaded, f'Running agents imported forbidden modules: {loaded}'

if __name__ == '__main__':
  app.run(main)
