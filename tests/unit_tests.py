import unittest

import numpy as np
import tensorflow as tf

from slippi_ai import embed, utils, tf_utils

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

class TFUtilsTest(unittest.TestCase):
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
    dynamic_outputs, _ = tf_utils.dynamic_rnn(nested_core, inputs, initial_state)

    tf.nest.map_structure(assert_tensors_close, static_outputs, dynamic_outputs)

  def test_non_trainable_scope(self):
    with tf_utils.non_trainable_scope():
      assert not tf.Variable(1.0).trainable

class UtilsTest(unittest.TestCase):

  def test_peekable_queue(self):
    q = utils.PeekableQueue()
    for i in range(5):
      q.put(i)

    for i in range(5):
      n = i + 1
      self.assertListEqual(q.peek_n(n), list(range(n)))

    for i in range(5, 10):
      q.put(i)

    for i in range(10):
      n = i + 1
      self.assertListEqual(q.peek_n(n), list(range(n)))

    for i in range(10):
      self.assertEqual(q.get(), i)

    self.assertTrue(q.empty())

class EmbedTest(unittest.TestCase):

  def test_flatten_and_unflatten(self):
    embed_game = embed.make_game_embedding()

    embed_game_struct = embed_game.map(lambda e: e)
    embed_game_flat = embed_game.flatten(embed_game_struct)
    embed_game_unflat = embed_game.unflatten(embed_game_flat)

    self.assertEqual(embed_game_unflat, embed_game_struct)

if __name__ == '__main__':
  unittest.main(failfast=True)
