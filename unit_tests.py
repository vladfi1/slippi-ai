import unittest

import numpy as np
import tensorflow as tf

import embed
import utils
import transformers

def static_rnn(core, inputs, initial_state):
  unroll_length = tf.nest.flatten(inputs)[0].shape[0]

  def get_input(i):
    return tf.nest.map_structure(lambda t: t[i], inputs)

  state = initial_state
  outputs = []
  for i in range(unroll_length):
    output, state = core(get_input(i), state)
    outputs.append(output)

  outputs = tf.nest.map_structure(lambda *ts: tf.stack(ts, 0), *outputs)
  return outputs, state

def assert_tensors_close(t1, t2):
  np.testing.assert_allclose(t1.numpy(), t2.numpy())

class UtilsTest(unittest.TestCase):
  def test_dynamic_rnn(self):

    def nested_core(input_, state):
      output = tf.nest.map_structure(lambda t: t + state, input_)
      return output, state

    unroll_length = 8
    batch_size = 4
    initial_state = tf.constant(1.0)

    inputs = dict(
        a=tf.random.uniform([unroll_length, batch_size]),
        b=tf.random.uniform([unroll_length, batch_size]),
    )

    static_outputs, _ = static_rnn(nested_core, inputs, initial_state)
    dynamic_outputs, _ = utils.dynamic_rnn(nested_core, inputs, initial_state)

    tf.nest.map_structure(assert_tensors_close, static_outputs, dynamic_outputs)

class EmbedTest(unittest.TestCase):

  def test_flatten_and_unflatten(self):
    embed_game = embed.make_game_embedding()

    embed_game_struct = embed_game.map(lambda e: e)
    embed_game_flat = embed_game.flatten(embed_game_struct)
    embed_game_unflat = embed_game.unflatten(embed_game_flat)

    self.assertEqual(embed_game_unflat, embed_game_struct)

class Test_Transformers(unittest.TestCase):
  def test_attention(self):
    mhab = transformers.MultiHeadAttentionBlock(8, 512)
    # Shape grabbed from breakpoint of slippi
    test_inputs = tf.ones([64, 32, 866]) 
    result = mhab(test_inputs)
    assert result.shape == tf.TensorShape([64, 32, 512])

if __name__ == '__main__':
  unittest.main(failfast=True)
