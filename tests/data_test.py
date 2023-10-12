import unittest

import numpy as np

from slippi_ai import data

def get_all_values(d):
    for key, value in d.items():
        if isinstance(value, dict):
            yield from get_all_values(value)
        else:
          yield value
def get_all_keys(d):
    for key, value in d.items():
        if isinstance(value, dict):
            yield from get_all_keys(value)
        else:
          yield key

class CompressRepeatedActionsTest(unittest.TestCase):

  def test_indices_and_counts(self):
    actions = np.random.randint(2, size=100)
    repeats = data.detect_repeated_actions(actions)
    indices, counts = data.indices_and_counts(repeats,3)

    reconstruction = []
    for i, c in zip(indices, counts):
      reconstruction.extend([actions[i]] * (c + 1))
    self.assertSequenceEqual(reconstruction, actions.tolist())
  def test_compress(self):
    rewards = np.random.randint(1,2, size=605)
    import pickle
    with open('game.json','rb') as outfile:
      game = pickle.load(outfile)
    with open('embed','rb') as outfile:
      embed=pickle.load(outfile)

    compressed = data.compress_repeated_actions(game,rewards,embed,604)
    othercompressed = data.compress_repeated_actions(game,rewards,embed,0)

    
    # del compressed.states['player'][1]['y']
    # del compressed.states['player'][2]['y']
    # del othercompressed.states['player'][1]['y']
    # del othercompressed.states['player'][2]['y']
    # del compressed.states['player'][1]['action']
    # del compressed.states['player'][2]['action']
    # del othercompressed.states['player'][1]['action']
    # del othercompressed.states['player'][2]['action']
    # del compressed.states['player'][1]['jumps_left']
    # del compressed.states['player'][2]['jumps_left']
    # del othercompressed.states['player'][1]['jumps_left']
    # del othercompressed.states['player'][2]['jumps_left']
    # del compressed.states['player'][1]['on_ground']
    # del compressed.states['player'][2]['on_ground']
    # del othercompressed.states['player'][1]['on_ground']
    # del othercompressed.states['player'][2]['on_ground']
    # del compressed.states['player'][1]['controller_state']['main_stick']
    # del compressed.states['player'][2]['controller_state']['main_stick']
    # del othercompressed.states['player'][1]['controller_state']['main_stick']
    # del othercompressed.states['player'][2]['controller_state']['main_stick']
    # del compressed.states['player'][1]['controller_state']['c_stick']
    # del compressed.states['player'][2]['controller_state']['c_stick']
    # del othercompressed.states['player'][1]['controller_state']['c_stick']
    # del othercompressed.states['player'][2]['controller_state']['c_stick']

    
    for actions,otheractions,key in zip(get_all_values(compressed.states),get_all_values(othercompressed.states),get_all_keys(compressed.states)):
      reconstruction = []
      otherreconstruction = []
      for i, c in zip(actions, compressed.counts):
        reconstruction.extend([i] * (c + 1))
      for i, c in zip(otheractions, othercompressed.counts):
        otherreconstruction.extend([i] * (c + 1))
      self.assertSequenceEqual(reconstruction, otherreconstruction)

if __name__ == '__main__':
  unittest.main(failfast=True)
