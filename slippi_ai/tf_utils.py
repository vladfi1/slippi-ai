"""Tensorflow utilities.

Separate from non-TF utils to reduce memory usage.
"""

import contextlib
import functools
import typing as tp

import numpy as np
import tensorflow as tf
import tree

def mean_and_variance(xs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
  mean = tf.reduce_mean(xs)
  variance = tf.reduce_mean(tf.square(xs - mean))
  return mean, variance

def get_stats(x):
  mean, variance = mean_and_variance(x)
  return dict(
      mean=mean,
      variance=variance,
      stddev=tf.sqrt(variance),
      min=tf.reduce_min(x),
      max=tf.reduce_max(x),
  )

def to_numpy(x) -> np.ndarray:
  if isinstance(x, tf.Tensor):
    return x.numpy()
  return x

Inputs = tree.Structure[tf.Tensor]
Outputs = tree.Structure[tf.Tensor]
RecurrentState = tree.Structure[tf.Tensor]

def dynamic_rnn(
    core: tp.Callable[[Inputs, RecurrentState], tp.Tuple[Outputs, RecurrentState]],
    inputs: Inputs,
    initial_state: RecurrentState,
) -> tp.Tuple[Outputs, RecurrentState]:
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

  input_0 = tf.nest.map_structure(lambda t: t[0], inputs)
  output_0, _ = core(input_0, initial_state)

  outputs = tf.nest.map_structure(
      lambda t: tf.TensorArray(
          dtype=t.dtype, size=unroll_length, element_shape=t.shape),
      output_0)
  del input_0, output_0

  def write_output(index, output, buffers):
    return tf.nest.map_structure(
        lambda ta, t: ta.write(index, t),
        buffers, output)

  # tf.scan also converts inputs to TensorArrays; let's copy them
  inputs = tf.nest.map_structure(
      lambda t: tf.TensorArray(
          dtype=t.dtype, size=t.shape[0],
          element_shape=t.shape[1:]).unstack(t),
      inputs)

  def get_input(index):
    return tf.nest.map_structure(lambda t: t.read(index), inputs)

  cond = lambda i, *_: i < unroll_length

  def body(index, buffers, state):
    output, state = core(get_input(index), state)
    buffers = write_output(index, output, buffers)
    return index + 1, buffers, state

  loop_vars = (0, outputs, initial_state)

  _, outputs, state = tf.while_loop(
      cond, body, loop_vars,
      parallel_iterations=1,
      maximum_iterations=unroll_length)

  # for i in tf.range(1, unroll_length):
  #   output, state = core(get_input(i), state)
  #   outputs = write_output(i, output)

  outputs = tf.nest.map_structure(lambda ta: ta.stack(), outputs)
  return outputs, state

def scan_rnn(
    core: tp.Callable[[Inputs, RecurrentState], tp.Tuple[Outputs, RecurrentState]],
    inputs: Inputs,
    initial_state: RecurrentState,
) -> tp.Tuple[Outputs, RecurrentState]:
  """Like dynamic_rnn, but returns intermediary hidden states."""
  inputs0 = tree.map_structure(lambda t: t[0], inputs)
  outputs0 = core(inputs0, initial_state)[0]
  dummy_outputs = tree.map_structure(tf.zeros_like, outputs0)

  initializer = (dummy_outputs, initial_state)

  def fn(prev_output, input):
    _, prev_state = prev_output
    return core(input, prev_state)

  return tf.scan(fn, inputs, initializer)


def where(cond: tf.Tensor, x: tf.Tensor, y: tf.Tensor):
  """Broadcasting tf.where, with cond of shape [B]."""
  rank = len(x.shape)
  cond = tf.expand_dims(cond, list(range(1, rank)))
  return tf.where(cond, x, y)

T = tp.TypeVar('T')
# P = tp.ParamSpec('P')

def run_on_cpu(fn: tp.Callable[..., T]) -> tp.Callable[..., T]:
  """Decorator to run a function on the CPU."""
  @functools.wraps(fn)
  def wrapped(*args, **kwargs):
    with tf.device('/cpu:0'):
      return fn(*args, **kwargs)
  return wrapped

def _create_non_trainable(next_creator, **kwargs) -> tf.Variable:
  kwargs = kwargs.copy()
  kwargs.update(trainable=False)
  return next_creator(**kwargs)

@contextlib.contextmanager
def non_trainable_scope():
  with tf.variable_creator_scope(_create_non_trainable):
    yield

def assert_same_variables(
    xs: tp.Sequence[tf.Variable],
    ys: tp.Sequence[tf.Variable],
) -> bool:
  if len(xs) != len(ys):
    raise ValueError('different lengths')

  xs = sorted(xs, key=lambda v: v.name)
  ys = sorted(ys, key=lambda v: v.name)

  for x, y in zip(xs, ys):
    assert x is y
