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
  np.testing.assert_allclose(t1.numpy(), t2.numpy(), rtol=1e-5, atol=1e-5)

def default_network(name):
  return networks.CONSTRUCTORS[name](**networks.DEFAULT_CONFIG[name])

embed_controller = embed.embed_controller_discrete
max_action_repeat = 15
embed_controller_with_repeat = embed.get_controller_embedding_with_action_repeat(
    embed_controller, max_action_repeat)

def default_data_source(batch_size, unroll_length):
  return data.DataSource(
      [paths.MULTISHINE_PATH],
      batch_size=batch_size,
      unroll_length=unroll_length,
      embed_controller=embed_controller,
      max_action_repeat=max_action_repeat,
  )

def get_inputs(data_source: data.DataSource):
  batch = next(data_source)[0]

  # from Learner
  bm_gamestate = batch.game
  tm_gamestate = tf.nest.map_structure(learner.to_time_major, bm_gamestate)

  # from Policy
  gamestate, action_repeat, rewards = tm_gamestate
  del rewards
  p1_controller = get_p1_controller(gamestate, action_repeat)
  p1_controller_embed = embed_controller_with_repeat(p1_controller)
  return (gamestate, p1_controller_embed)

class NetworksTest(unittest.TestCase):

  def unroll_vs_step_helper(
      self,
      network: networks.Network,
      batch_size: int,
      sequence_length: int,
      num_rollouts=5,
  ):
    initial_state = network.initial_state(batch_size)
    data_source = default_data_source(
        batch_size=batch_size, unroll_length=sequence_length)

    for _ in range(num_rollouts):
      inputs = get_inputs(data_source)

      unroll_outputs, unroll_final_state = network.unroll(inputs, initial_state)
      step_outputs, step_final_state = utils.dynamic_rnn(
          network.step, inputs, initial_state)

      assert_tensors_close(unroll_outputs, step_outputs)
      tf.nest.map_structure(assert_tensors_close, unroll_final_state, step_final_state)

  @parameterized.expand(networks.CONSTRUCTORS)
  def test_unroll_vs_step(self, name='mlp'):
    network = default_network(name)
    self.unroll_vs_step_helper(network, batch_size=1, sequence_length=8)

  def test_transformer(self):
    def test_tfm(O, L, F, H, M, B, S):
      network = networks.TransformerWrapper(O, L, F, H, M)
      self.unroll_vs_step_helper(network, batch_size=B, sequence_length=S)

    with self.subTest('deep'):
      test_tfm(3, 20, 3, 1, 0, 1, 8)
    
    with self.subTest('serious'):
      test_tfm(O=20, L=10, F=30, H=5, M=16, B=3, S=32)

if __name__ == '__main__':
  unittest.main(failfast=True)