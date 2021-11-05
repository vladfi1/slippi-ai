import unittest

import numpy as np
import tensorflow as tf

import embed
import utils
import transformers
from networks import TransformerWrapper as tw

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
  def test_attention_block(self):
    mhab = transformers.MultiHeadAttentionBlock(8, 512)
    # Shape grabbed from breakpoint of slippi
    test_inputs = tf.ones([64, 32, 866]) 
    result = mhab(test_inputs)
    assert result.shape == tf.TensorShape([64, 32, 512])

  def test_limited_attention(self):
    test_queries = tf.ones([64, 32, 32]) * 3
    test_keys = tf.ones([64, 32, 32]) * 5
    test_values = tf.ones([64, 32, 64]) * 10
    l_attention = transformers.attention(test_queries, test_keys, test_values, window_size=1)
    l_attention = transformers.attention(test_queries, test_keys, test_values, window_size=2)

  def test_attention_limited_unset_is_att(self):
    test_queries = tf.ones([64, 32, 32]) * 3
    test_keys = tf.ones([64, 32, 32]) * 5
    test_values = tf.ones([64, 32, 64]) * 10
    l_attention = transformers.limited_attention(test_queries, test_keys, test_values, -1)
    attention = transformers.attention(test_queries, test_keys, test_values)
    np.testing.assert_allclose(l_attention.numpy(), attention.numpy())

  def test_pos_encoding(self):
    pos = transformers.positional_encoding(5, 10, batch_size=32)
    assert pos.shape == tf.TensorShape([32, 5, 10])
    pos = transformers.positional_encoding(30, 512, batch_size=32)
    assert pos.shape == [32, 30, 512]
  
  def test_transformer_block(self):
    test_inputs_nice = tf.ones([64, 32, 512]) 
    tb = transformers.TransformerEncoderBlock(512)
    output = tb(test_inputs_nice)
    assert output.shape == test_inputs_nice.shape

    test_inputs_2 = tf.ones([64, 32, 866])
    tb = transformers.TransformerEncoderBlock(866)
    output_2 = tb(test_inputs_2)
    assert output_2.shape == test_inputs_2.shape

  def test_transformer_call(self):
    transformer = transformers.EncoderOnlyTransformer(512)
    test_inputs_nice = tf.ones([64, 32, 512])
    output = transformer(test_inputs_nice)
    assert output.shape == test_inputs_nice.shape

    transformer = transformers.EncoderOnlyTransformer(866)
    test_inputs_2 = tf.ones([64, 32, 866])
    output_2 = transformer(test_inputs_2)
    assert output_2.shape == test_inputs_2.shape

  def test_tranformer_wrap_unroll(self):
    transformer_wrapper = tw(128, 2, 256, 8, 32)
    test_inputs_nice = tf.ones([64, 32, 512])
    transformer_wrapper.unroll(test_inputs_nice, ())

  def test_tranformer_wrap_step(self):
    transformer_wrapper = tw(128, 2, 256, 8, 32)
    test_inputs_nice = tf.ones([64, 32, 512])
    test_hidden = tf.ones([32, 512*32])
    transformer_wrapper.step(test_inputs_nice, test_hidden)

if __name__ == '__main__':
  unittest.main(failfast=True)
