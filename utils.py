import time

import numpy as np
import tensorflow as tf
import tree

def np_array(*vals):
  return np.array(vals)

def batch_nest(nests):
  return tf.nest.map_structure(np_array, *nests)

def dynamic_rnn(core, inputs, initial_state):
  unroll_length = tf.shape(inputs)[0]
  outputs = tf.TensorArray(dtype=tf.float32, size=unroll_length)
  state = initial_state
  for i in tf.range(unroll_length):
    input_ = inputs[i]  # TODO: handle nested inputs
    output, state = core(input_, state)
    outputs = outputs.write(i, output)
  return outputs.stack(), state

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

def add_batch_dims(spec: tf.TensorSpec, num_dims: int):
  return tf.TensorSpec([None] * num_dims + spec.shape.as_list(), spec.dtype)

def nested_add_batch_dims(nest, num_dims):
  return tree.map_structure(lambda spec: add_batch_dims(spec, num_dims), nest)

def with_flat_signature(fn, signature):
  def g(*flat_args):
    return fn(*tree.unflatten_as(signature, flat_args))
  return tf.function(g, input_signature=tree.flatten(signature))
