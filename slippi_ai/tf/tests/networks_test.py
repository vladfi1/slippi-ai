import logging
import unittest
from parameterized import parameterized

import numpy as np
import tensorflow as tf

from slippi_ai import (
    data
)
from slippi_ai.tf import embed, learner, networks, tf_utils

def assert_tensors_close(t1: tf.Tensor, t2: tf.Tensor):
  # TODO: relax tolerance when running on GPU
  np.testing.assert_allclose(
      t1.numpy(), t2.numpy(),
      rtol=1e-5, atol=1e-6)

default_network_config = networks.default_config()

def default_network(name):
  return networks.CONSTRUCTORS[name](**default_network_config[name])

embed_game = embed.make_game_embedding()

def default_data_source():
  return data.toy_data_source(batch_size=1, unroll_length=8)

def get_inputs(data_source: data.DataSource):
  batch = next(data_source)[0]
  bm_state = embed_game.from_state(batch.game)
  inputs = embed_game(bm_state), batch.is_resetting
  return tf.nest.map_structure(learner.swap_axes, inputs)

class NetworksTest(unittest.TestCase):

  @parameterized.expand(networks.CONSTRUCTORS)
  def test_unroll_vs_step(self, name='mlp'):
    network = default_network(name)
    initial_state = network.initial_state(1)
    data_source = default_data_source()

    for _ in range(5):
      inputs, reset = get_inputs(data_source)

      unroll_outputs, unroll_final_state = network.unroll(
          inputs, reset, initial_state)
      step_outputs, step_final_state = tf_utils.dynamic_rnn(
          network._step_with_reset, (inputs, reset), initial_state)

      assert_tensors_close(unroll_outputs, step_outputs)
      tf.nest.map_structure(assert_tensors_close, unroll_final_state, step_final_state)

if __name__ == '__main__':
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    logging.warning("Tests may not work properly on GPU.")
  unittest.main(failfast=True)
