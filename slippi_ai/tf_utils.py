"""Tensorflow utilities.

Separate from non-TF utils to reduce memory usage.
"""

import dataclasses
import math
import collections
import contextlib
import functools
import typing as tp

import numpy as np
import tensorflow as tf
import tree

from slippi_ai import utils

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
  for _ in range(rank - len(cond.shape)):
    cond = tf.expand_dims(cond, -1)
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

def tile(x: tf.Tensor, axis: int, multiple: int) -> tf.Tensor:
  multiples = [1] * len(x.shape)
  multiples[axis] = multiple
  return tf.tile(x, multiples)

def expand_tile(x: tf.Tensor, axis: int, multiple: int) -> tf.Tensor:
  return tile(tf.expand_dims(x, axis), axis, multiple)

def move_axis(x: tf.Tensor, src: int, dst: int) -> tf.Tensor:
  perm = list(range(len(x.shape)))
  perm.pop(src)
  perm.insert(dst, src)
  return tf.transpose(x, perm)


@dataclasses.dataclass
class ArraySpec:
  dtype: np.dtype
  shape: tuple[int, ...]

PackingSpec = tp.Optional[ArraySpec]
Signature = tp.Sequence[tree.StructureKV[str, PackingSpec]]

def packing_fns(
    signature: Signature,
):
  flat_signature: list[PackingSpec] = tree.flatten(signature)

  flat_slices = []
  num_skipped = 0
  packed_sizes = collections.defaultdict(lambda: 0)

  for spec in flat_signature:
    if spec is None:
      flat_slices.append(num_skipped)
      num_skipped += 1
      continue

    size = math.prod(spec.shape)
    start = packed_sizes[spec.dtype]
    end = start + size
    flat_slices.append((start, end))
    packed_sizes[spec.dtype] = end

  dtypes = list(packed_sizes.keys())
  print('dtypes:', dtypes)

  def pack_args(*args):
    flattened_args = {dtype: [] for dtype in dtypes}
    skipped = []

    # flat_inputs = tree.flatten_up_to(signature, args, check_types=False)
    flat_inputs = utils.flatten_up_to(signature, args)
    for array, spec in zip(flat_inputs, flat_signature):
      if spec is None:
        skipped.append(array)
        continue

      assert isinstance(array, np.ndarray)
      assert array.dtype == spec.dtype
      assert array.shape == spec.shape
      flattened_args[spec.dtype].append(np.reshape(array, [-1]))

    packed = []
    for dtype in dtypes:
      packed.append(np.concatenate(flattened_args[dtype], axis=0))

    return packed, skipped

  def unpack_args(packed_args, skipped):
    dtype_to_array = {
        dtype: array
        for dtype, array in zip(dtypes, packed_args)
    }
    flat_arrays = []
    for spec, slice_or_idx in zip(flat_signature, flat_slices):
      if spec is None:
        assert isinstance(slice_or_idx, int)
        flat_arrays.append(skipped[slice_or_idx])
        continue

      start, end = slice_or_idx
      array = dtype_to_array[spec.dtype]
      subarray = array[start:end]
      if isinstance(array, tf.Tensor):
        reshaped = tf.reshape(subarray, spec.shape)
      else:
        reshaped = np.reshape(subarray, spec.shape)
      flat_arrays.append(reshaped)

    return tree.unflatten_as(signature, flat_arrays)

  return pack_args, unpack_args


def packed_compile(
    fn: tp.Callable[P, T],
    signature: Signature,
    **compile_kwargs,
) -> tp.Callable[P, T]:
  """Compiles a function with packed inputs.

  Packing inputs can improve performance by reducing the number of tensors
  that TF has to handle. The tensorboard trace view shows a lot of time spent
  in `tf.constant` when there are many input tensors.
  """
  pack_args, unpack_args = packing_fns(signature)

  @tf.function(**compile_kwargs)
  def packed_fn(packed, skipped):
    return fn(*unpack_args(packed, skipped))

  def wrapped(*args):
    packed, skipped = pack_args(*args)
    return packed_fn(packed, skipped)

  return wrapped
