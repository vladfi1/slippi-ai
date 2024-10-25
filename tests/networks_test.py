import unittest
from parameterized import parameterized

import numpy as np
import tensorflow as tf

from slippi_ai import (
    learner, networks, paths, data, tf_utils, embed
)

def assert_tensors_close(t1, t2):
  np.testing.assert_allclose(t1.numpy(), t2.numpy())

def default_network(name):
  return networks.CONSTRUCTORS[name](**networks.DEFAULT_CONFIG[name])

embed_game = embed.make_game_embedding()

def default_data_source():
  return data.toy_data_source(batch_size=1, unroll_length=8)

def get_inputs(data_source: data.DataSource):
  batch = next(data_source)[0]
  bm_state = embed_game.from_state(batch.frames.state_action.state)
  tm_state = tf.nest.map_structure(learner.swap_axes, bm_state)
  return embed_game(tm_state)

class NetworksTest(unittest.TestCase):

  @parameterized.expand(networks.CONSTRUCTORS)
  def test_unroll_vs_step(self, name='mlp'):
    network = default_network(name)
    initial_state = network.initial_state(1)
    data_source = default_data_source()

    for _ in range(5):
      inputs = get_inputs(data_source)

      unroll_outputs, unroll_final_state = network.unroll(inputs, initial_state)
      step_outputs, step_final_state = tf_utils.dynamic_rnn(network.step, inputs, initial_state)

      assert_tensors_close(unroll_outputs, step_outputs)
      tf.nest.map_structure(assert_tensors_close, unroll_final_state, step_final_state)

if __name__ == '__main__':
  unittest.main(failfast=True)
