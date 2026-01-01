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

dtypes = [np.float32, np.int32, np.bool_, np.uint8]

def random_spec() -> tf_utils.ArraySpec:
  dtype = np.random.choice(dtypes)
  ndims = np.random.randint(1, 4)
  shape = tuple(np.random.randint(1, 5, size=ndims))
  return tf_utils.ArraySpec(dtype=dtype, shape=shape)

def random_signature(size: int) -> list[tf_utils.ArraySpec | None]:
  signature = []
  for _ in range(size):
    if np.random.rand() < 0.2:
      signature.append(None)
      continue

    signature.append(random_spec())

  return tuple(signature)

def random_array(spec: tf_utils.ArraySpec | None) -> np.ndarray:
  if spec is None:
    return [np.random.uniform(size=(2, 3))]

  if np.issubdtype(spec.dtype, np.floating):
    return np.random.uniform(size=spec.shape).astype(spec.dtype)
  elif np.issubdtype(spec.dtype, np.integer):
    return np.random.randint(
        0, 100, size=spec.shape, dtype=spec.dtype)
  elif spec.dtype == np.bool_:
    return np.random.randint(
        0, 2, size=spec.shape, dtype=np.int32).astype(bool)
  else:
    raise ValueError(f'Unsupported dtype: {spec.dtype}')

def sample_input(signature: list[tf_utils.ArraySpec | None]) -> list[np.ndarray]:
  return [random_array(spec) for spec in signature]

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

  def test_scan_rnn(self):

    def core(input_, state):
      return input_ + state, state + 1

    initial_state = tf.constant(0)
    inputs = tf.constant([1, 2, 3, 4])

    outputs, hidden_states = tf_utils.scan_rnn(core, inputs, initial_state)

    expected_hidden_states = tf.constant([1, 2, 3, 4])
    expected_outputs = tf.constant([1, 3, 5, 7])

    tf.nest.map_structure(
        assert_tensors_close,
        (outputs, hidden_states),
        (expected_outputs, expected_hidden_states),
    )

  def test_non_trainable_scope(self):
    with tf_utils.non_trainable_scope():
      assert not tf.Variable(1.0).trainable

  def test_move_axis(self):
    x = tf.random.uniform([4, 5, 6, 7])
    y = tf_utils.move_axis(x, 1, 3)
    self.assertEqual(y.shape, [4, 6, 7, 5])

  def test_packing_fns(self):
    for _ in range(10):
      signature = random_signature(size=20)
      inputs = sample_input(signature)

      pack_fn, unpack_fn = tf_utils.packing_fns(signature)
      packed = pack_fn(*inputs)
      unpacked = unpack_fn(*packed)

      tf.nest.map_structure(
          np.testing.assert_array_equal,
          inputs, unpacked, check_types=False)

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
