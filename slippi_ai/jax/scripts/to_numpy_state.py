"""Convert states mistakenly saved as jax arrays to numpy."""

import pickle

from absl import flags, app
import numpy as np
import jax

PATH = flags.DEFINE_string('path', None, 'Path to the pickled model.', required=True)

def main(_):
  with open(PATH.value, 'rb') as f:
    state = pickle.load(f)

  state['state'] = jax.tree.map(np.asarray, state['state'])

  with open(PATH.value, 'wb') as f:
    pickle.dump(state, f)

if __name__ == '__main__':
  app.run(main)
