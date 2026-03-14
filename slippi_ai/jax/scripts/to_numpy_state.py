import pickle

from absl import app, flags
import numpy as np
import tree

PATH = flags.DEFINE_string('path', None, 'path to pickled model', required=True)

def main(_):
  with open(PATH.value, 'rb') as f:
    state = pickle.load(f)

  state['state'] = tree.map_structure(np.asarray, state['state'])

  with open(PATH.value, 'wb') as f:
    pickle.dump(state, f)

if __name__ == '__main__':
  app.run(main)
