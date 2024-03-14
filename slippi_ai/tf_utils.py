"""Tensorflow utilities.

Separate from non-TF utils to reduce memory usage.
"""

import functools
import typing as tp

import numpy as np
import tensorflow as tf

def mean_and_variance(xs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
  mean = tf.reduce_mean(xs)
  variance = tf.reduce_mean(tf.square(xs - mean))
  return mean, variance

def to_numpy(x) -> np.ndarray:
  if isinstance(x, tf.Tensor):
    return x.numpy()
  return x

def dynamic_rnn(core, inputs, initial_state):
  """Dynamically unrolls an rnn core.

  In the simple case of flat inputs and outputs, executes:
  T = len(inputs)
  state[0] = initial_state
  outputs[i], state[i+1] = core(inputs[i], state[i])
  returns: outputs, state[T]

  In general inputs and outputs can be nests of tensors.

  Args:
    core: A callable with signature (input, state) -> (output, state).
      For example, a snt.LSTM.
    inputs: A nest of time-major tensors.
    initial_state: A nest of tensors.
  Returns:
    A tuple (outputs, final_state).
  """
  unroll_length = tf.shape(tf.nest.flatten(inputs)[0])[0]

  def get_input(index):
    return tf.nest.map_structure(lambda t: t[index], inputs)

  output_0, state = core(get_input(0), initial_state)

  outputs = tf.nest.map_structure(
      lambda t: tf.TensorArray(
          dtype=t.dtype, size=unroll_length, element_shape=t.shape),
      output_0)

  def write_output(index, output):
    return tf.nest.map_structure(
        lambda ta, t: ta.write(index, t),
        outputs, output)

  outputs = write_output(0, output_0)

  for i in tf.range(1, unroll_length):
    output, state = core(get_input(i), state)
    outputs = write_output(i, output)

  outputs = tf.nest.map_structure(lambda ta: ta.stack(), outputs)
  return outputs, state

def where(cond: tf.Tensor, x: tf.Tensor, y: tf.Tensor):
  """Broadcasting tf.where, with cond of shape [B]."""
  rank = len(x.shape)
  cond = tf.expand_dims(cond, list(range(1, rank)))
  return tf.where(cond, x, y)

T = tp.TypeVar('T')
P = tp.ParamSpec('P')

def run_on_cpu(fn: tp.Callable[P, T]) -> tp.Callable[P, T]:
  """Decorator to run a function on the CPU."""
  @functools.wraps(fn)
  def wrapped(*args, **kwargs):
    with tf.device('/cpu:0'):
      return fn(*args, **kwargs)
  return wrapped
