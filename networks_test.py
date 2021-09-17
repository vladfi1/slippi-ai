import unittest
from parameterized import parameterized

import numpy as np
import tensorflow as tf

import learner
import networks
import data
import paths
import utils
import embed
from policies import get_p1_controller

def assert_tensors_close(t1, t2):
  np.testing.assert_allclose(t1.numpy(), t2.numpy())

def default_network(name):
  return networks.CONSTRUCTORS[name](**networks.DEFAULT_CONFIG[name])

embed_controller = embed.embed_controller_discrete
max_action_repeat = 15
embed_controller_with_repeat = embed.get_controller_embedding_with_action_repeat(
    embed_controller, max_action_repeat)

def default_data_source():
  return data.DataSource(
      [paths.MULTISHINE_PATH],
      batch_size=1,
      unroll_length=8,
      embed_controller=embed_controller,
      max_action_repeat=max_action_repeat,
  )

def get_inputs(data_source: data.DataSource):
  batch = next(data_source)

  # from Learner
  bm_gamestate, restarting = batch
  tm_gamestate = tf.nest.map_structure(learner.to_time_major, bm_gamestate)

  # from Policy
  gamestate, action_repeat, rewards = tm_gamestate
  del rewards
  p1_controller = get_p1_controller(gamestate, action_repeat)
  p1_controller_embed = embed_controller_with_repeat(p1_controller)
  return (gamestate, p1_controller_embed)

class NetworksTest(unittest.TestCase):

  @parameterized.expand(networks.CONSTRUCTORS)
  def test_unroll_vs_step(self, name='mlp'):
    network = default_network(name)
    initial_state = network.initial_state(1)
    data_source = default_data_source()

    for _ in range(5):
      inputs = get_inputs(data_source)

      unroll_outputs, unroll_final_state = network.unroll(inputs, initial_state)
      step_outputs, step_final_state = utils.dynamic_rnn(network.step, inputs, initial_state)

      assert_tensors_close(unroll_outputs, step_outputs)
      tf.nest.map_structure(assert_tensors_close, unroll_final_state, step_final_state)

if __name__ == '__main__':
  unittest.main(failfast=True)
