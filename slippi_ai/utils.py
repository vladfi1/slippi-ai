import time
import typing as tp

import numpy as np
import tensorflow as tf
import tree

T = tp.TypeVar('T')

def stack(*vals):
  return np.stack(vals)

def batch_nest(nests: tp.Sequence[T]) -> T:
  return tf.nest.map_structure(stack, *nests)

def to_time_major(t):
  permutation = list(range(len(t.shape)))
  permutation[0] = 1
  permutation[1] = 0
  return tf.transpose(t, permutation)

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

class Profiler:
  def __init__(self):
    self.cumtime = 0
    self.num_calls = 0

  def __enter__(self):
    self._enter_time = time.perf_counter()

  def __exit__(self, type, value, traceback):
    self.cumtime += time.perf_counter() - self._enter_time
    self.num_calls += 1

  def mean_time(self):
    return self.cumtime / self.num_calls

class Periodically:
  def __init__(self, f, interval):
    self.f = f
    self.interval = interval
    self.last_call = None

  def __call__(self, *args, **kwargs):
    now = time.time()
    if self.last_call is None or now - self.last_call > self.interval:
      self.last_call = now
      return self.f(*args, **kwargs)

def periodically(interval: int):
  def wrap(f):
    return Periodically(f, interval)
  return wrap

T = tp.TypeVar('T')

class Tracker(tp.Generic[T]):

  def __init__(self, initial: T):
    self.last = initial

  def update(self, latest: T) -> T:
    delta = latest - self.last
    self.last = latest
    return delta

class EMA:
  """Exponential moving average."""

  def __init__(self, window: float):
    self.decay = 1. / window
    self.value = None

  def update(self, value):
    if self.value is None:
      self.value = value
    else:
      self.value += self.decay * (value - self.value)

def add_batch_dims(spec: tf.TensorSpec, num_dims: int):
  return tf.TensorSpec([None] * num_dims + spec.shape.as_list(), spec.dtype)

def nested_add_batch_dims(nest, num_dims):
  return tree.map_structure(lambda spec: add_batch_dims(spec, num_dims), nest)

def with_flat_signature(fn, signature):
  def g(*flat_args):
    return fn(*tree.unflatten_as(signature, flat_args))
  return tf.function(g, input_signature=tree.flatten(signature))

def mean_and_variance(xs: tf.Tensor) -> tp.Tuple[tf.Tensor, tf.TensorSpec]:
  mean = tf.reduce_mean(xs)
  variance = tf.reduce_mean(tf.square(xs - mean))
  return mean, variance

def to_numpy(x) -> np.ndarray:
  if isinstance(x, tf.Tensor):
    return x.numpy()
  return x
