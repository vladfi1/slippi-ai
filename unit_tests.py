import unittest

import numpy as np
import tensorflow as tf

import utils

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

if __name__ == '__main__':
  unittest.main()