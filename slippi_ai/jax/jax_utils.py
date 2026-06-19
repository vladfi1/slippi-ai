"""JAX utilities."""

import collections
import functools
import logging
import math
import os
import typing as tp
import types

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS
from flax import nnx
from flax.nnx.transforms.transforms import _resolve_bound_callable

from slippi_ai import utils

Array = jax.Array

P = tp.ParamSpec('P')
T = tp.TypeVar('T')

# Multi-device utilities

DATA_AXIS = 'data'

def data_spec(axis: int, name: str = DATA_AXIS) -> PS:
  """Convenience function for creating a PartitionSpec for data parallelism."""
  axes: list[str | None] = [None] * (axis + 1)
  axes[axis] = name
  return PS(*axes)

def get_mesh(axis_name: str = DATA_AXIS) -> Mesh:
  """Create a 1D device mesh for data parallelism."""
  return Mesh(jax.devices(), (axis_name,))


def replicate_sharding(mesh: Mesh) -> NamedSharding:
  """Create a sharding that replicates data across all devices."""
  return NamedSharding(mesh, PS())


def data_sharding(mesh: Mesh, axis_name: str = DATA_AXIS) -> NamedSharding:
  """Create a sharding that splits the first axis across devices."""
  return NamedSharding(mesh, PS(axis_name))

def map_update(
    f: tp.Callable[[jax.Array], jax.Array],
    module: nnx.Module | nnx.Rngs,
):
  state = nnx.state(module)
  nnx.update(module, jax.tree.map(f, state))

def shard_module(module: nnx.Module | nnx.Rngs, sharding: NamedSharding):
  """Shard/replicate module parameters across devices in-place."""
  map_update(lambda x: jax.device_put(x, sharding), module)

def replicate_module(module: nnx.Module, mesh: Mesh):
  """Replicate module parameters across all devices in the mesh."""
  shard_module(module, replicate_sharding(mesh))

def num_devices() -> int:
  """Get the number of local devices."""
  return jax.local_device_count()

def struct_dtype(struct) -> tp.Optional[jnp.dtype]:
  leaves = jax.tree_util.tree_leaves(struct)
  floating_dtypes = set[jnp.dtype]()
  for leaf in leaves:
    assert isinstance(leaf, jax.Array), f'Expected all leaves to be jax arrays, got {type(leaf)}.'
    if jnp.issubdtype(leaf.dtype, jnp.floating):
      floating_dtypes.add(leaf.dtype)
  if not floating_dtypes:
    return None
  if len(floating_dtypes) > 1:
    raise ValueError(f'Multiple floating point dtypes found in struct: {floating_dtypes}')
  return floating_dtypes.pop()

def module_dtype(module: nnx.Module) -> jnp.dtype:
  state = nnx.state(module, nnx.Param)
  dtype = struct_dtype(state)
  if dtype is None:
    raise ValueError('No floating point parameters found in module.')
  return dtype

def cast_floats_to_dtype(struct: T, dtype: jnp.dtype) -> T:
  """Cast all floating point parameters in the module to the given dtype."""
  return jax.tree.map(
      lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
      struct)

ModT = tp.TypeVar('ModT', bound=nnx.Module)

def cast_params_to_dtype(module: ModT, dtype) -> ModT:
  """Cast nnx.Param variables to given dtype."""
  graphdef, state = nnx.split(module, nnx.Param)
  new_state = cast_floats_to_dtype(state, dtype)
  return nnx.merge(graphdef, new_state)

def with_compute_dtype(
    loss_fn: tp.Callable[tp.Concatenate[ModT, P], T],
    dtype: jnp.dtype,
) -> tp.Callable[tp.Concatenate[ModT, P], T]:
  """Wraps loss_fn(module, ...) to cast params to bf16 before computing.

  Gradients flow back as fp32 through JAX's VJP of the dtype cast, so
  master weights and optimizer state remain fp32.
  """
  @functools.wraps(loss_fn)
  def wrapped(module: ModT, *args: P.args, **kwargs: P.kwargs):
    casted_module = cast_params_to_dtype(module, dtype=dtype)
    return loss_fn(casted_module, *args, **kwargs)
  return wrapped

def with_bf16_compute(
    loss_fn: tp.Callable[tp.Concatenate[ModT, P], T],
) -> tp.Callable[tp.Concatenate[ModT, P], T]:
  return with_compute_dtype(loss_fn, dtype=jnp.bfloat16)

def stack_modules(modules: tp.Iterable[ModT], axis: int = 0) -> ModT:
  """Stack a list of modules by adding a leading axis to their parameters."""
  modules = list(modules)

  if not modules:
    raise ValueError('Cannot stack an empty list of modules.')

  def stack_fn(*params):
    return jnp.stack(params, axis=axis)

  stacked_state = jax.tree.map(stack_fn, *[nnx.state(m) for m in modules])
  graphdef = nnx.graphdef(modules[0])
  return nnx.merge(graphdef, stacked_state)

def unstack_pytree(
    pytree: T,
    axis: int = 0,
) -> list[T]:
  """Unstack a pytree along the given axis, returning a list of pytrees."""
  leaves, structure = jax.tree.flatten(pytree)
  unstacked_leaves = [jnp.unstack(x, axis=axis) for x in leaves]
  return [jax.tree.unflatten(structure, xs) for xs in zip(*unstacked_leaves)]

# Other utilities

class MultiMap(tp.Protocol):

  @tp.overload
  def __call__(self, f: tp.Callable[[tp.Any], tp.Any], x: T, /) -> T:
    ...

  @tp.overload
  def __call__(self, f: tp.Callable[[tp.Any, tp.Any], tp.Any], x: T, y: T, /) -> T:
    ...

  @tp.overload
  def __call__(self, f: tp.Callable[[tp.Any, tp.Any, tp.Any], tp.Any], x: T, y: T, z: T, /) -> T:
    ...

  def __call__(self, f: tp.Callable, *args: T) -> T:
    ...

fast_map: MultiMap = tp.cast(MultiMap, jax.tree.map)

class SliceMap(tp.Protocol):
  def __call__(self, s: slice | tuple[slice, ...], struct: T) -> T:
    ...

slice_map: SliceMap = lambda s, struct: jax.tree.map(lambda x: x[s], struct)


def mean_and_variance(xs: Array) -> tuple[Array, Array]:
  mean = jnp.mean(xs)
  variance = jnp.mean(jnp.square(xs - mean))
  return mean, variance


def get_stats(x: Array) -> dict:
  mean, variance = mean_and_variance(x)
  return dict(
      mean=mean,
      variance=variance,
      stddev=jnp.sqrt(variance),
      min=jnp.min(x),
      max=jnp.max(x),
  )


def where(cond: Array, x: Array, y: Array) -> Array:
  """Broadcasting jnp.where, with cond of shape [B]."""
  shape = jnp.broadcast_shapes(x.shape, y.shape)
  while cond.ndim < len(shape):
    cond = jnp.expand_dims(cond, -1)
  return jnp.where(cond, x, y)


def swap_axes(t, axis1=0, axis2=1):
  """Swap two axes of a tensor."""
  return jnp.swapaxes(t, axis1, axis2)

def add_n(xs: tp.Iterable[Array]) -> Array:
  xs_iter = iter(xs)
  total = next(xs_iter)
  for x in xs_iter:
    total += x
  return total

def entropy(probs: Array, axis: int = -1) -> Array:
  """Compute the entropy of a probability distribution."""
  log_probs = jnp.where(probs > 0, jnp.log(probs), 0.0)
  return -jnp.vecdot(probs, log_probs, axis=axis)

# Flax NNX

def get_module_state(module: nnx.Module, to_numpy: bool = True) -> dict:
  """Get the state of a module as a pure dict."""
  state = nnx.state(module)
  pure_state = state.to_pure_dict()
  if to_numpy:
    pure_state = jax.tree.map(np.asarray, pure_state)
  return pure_state

def set_module_state(module: nnx.Module | nnx.Optimizer, state_dict: dict):
  """Set the state of a module from a pure dict."""
  state = nnx.state(module)
  nnx.replace_by_pure_dict(state, state_dict)
  nnx.update(module, state)

def cast_module_state_to_dtype(module: nnx.Module, dtype: jnp.dtype):
  state = nnx.state(module)
  state = cast_floats_to_dtype(state, dtype)
  nnx.update(module, state)

class MLP(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      input_size: int,
      features: list[int],
      activation=jax.nn.relu,
      activate_final: bool = False,
  ):
    self.input_size = input_size

    layers = []
    in_size = input_size
    for i, out_size in enumerate(features):
      if i > 0:
        layers.append(activation)
      layer = nnx.Linear(in_size, out_size, rngs=rngs)
      layers.append(layer)
      in_size = out_size

    if activate_final:
      layers.append(activation)

    self.layers = nnx.List(layers)
    self.output_size = in_size

  def __call__(self, x: Array) -> Array:
    for layer in self.layers:
      x = layer(x)
    return x

def remat_method(
    method: tp.Callable[P, T],
    **remat_kwargs,
) -> tp.Callable[P, T]:
  """Like nnx.remat but for bound methods."""
  unbound_f, bound_self, was_bound = _resolve_bound_callable(method)

  if not was_bound:
    raise ValueError('remat_method requires a bound method.')

  return functools.partial(nnx.remat(unbound_f, **remat_kwargs), bound_self)


def eval_shape_method(
    method: tp.Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
  if not isinstance(method, types.MethodType):
    raise TypeError('eval_shape_method can only be applied to methods.')

  # TODO: handle functools.partial
  return nnx.eval_shape(method.__func__, method.__self__, *args, **kwargs)

# TODO: fix type inference
# ArrayTree = tree.StructureKV[str, Array]
# Inputs = ArrayTree
# Outputs = ArrayTree
# InputTree = tp.TypeVar('InputTree', bound=Inputs)
# OutputTree = tp.TypeVar('OutputTree', bound=Outputs)

InputTree = tp.TypeVar('InputTree')
OutputTree = tp.TypeVar('OutputTree')

RecurrentState = tp.TypeVar('RecurrentState')

ScanAxis = int | None | type[nnx.Carry]
ScanAxes = ScanAxis | tuple[ScanAxis, ...]

def scan_method(
    method: tp.Callable[P, T],
    *,
    in_axes: ScanAxes = (0, nnx.Carry),
    out_axes: ScanAxes = (0, nnx.Carry),
    **scan_kwargs,
) -> tp.Callable[P, T]:
  """Like nnx.scan but for bound methods.

  Note the swapped order of input/output and carry.
  """

  # TODO: snoop inside functools.partial args for nnx Modules
  unbound_f, bound_self, was_bound = _resolve_bound_callable(method)

  if not was_bound:
    raise ValueError('scan_method requires a bound method.')

  if not isinstance(in_axes, tuple):
    in_axes = (in_axes,)

  in_axes = (None,) + in_axes

  return functools.partial(nnx.scan(unbound_f, in_axes=in_axes, out_axes=out_axes, **scan_kwargs), bound_self)

def functionalize(f: tp.Callable[P, T]) -> tp.Callable[P, T]:
  """Hides nnx modules from nnx transformations to prevent trace errors."""
  @functools.wraps(f)
  def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
    return f(*args, **kwargs)
  return wrapped

def dynamic_rnn(
    cell_fn: tp.Callable[[InputTree, RecurrentState], tuple[OutputTree, RecurrentState]],
    inputs: InputTree,
    initial_state: RecurrentState,
) -> tuple[OutputTree, RecurrentState]:
  """Unrolls an RNN over time, returning outputs and final state.

  Args:
    cell_fn: Function (inputs, state) -> (outputs, new_state)
    inputs: Inputs with time as first axis
    initial_state: Initial recurrent state

  Returns:
    outputs: Stacked outputs over time
    final_state: Final recurrent state
  """

  cell_fn = functionalize(cell_fn)

  # Workaround for https://docs.jax.dev/en/latest/notebooks/shard_map.html#scan-vma
  _, state = jax.eval_shape(cell_fn, inputs, initial_state)
  initial_state = jax.tree.map(as_vma, initial_state, state)

  return nnx.scan(
      cell_fn,
      in_axes=(0, nnx.Carry),
      out_axes=(0, nnx.Carry),
  )(inputs, initial_state)

def dynamic_rnn_module(
    cell_fn: tp.Callable[[ModT, InputTree, RecurrentState], tuple[OutputTree, RecurrentState]],
    module: ModT,
    inputs: InputTree,
    initial_state: RecurrentState,
) -> tuple[OutputTree, RecurrentState]:
  # Workaround for https://docs.jax.dev/en/latest/notebooks/shard_map.html#scan-vma
  _, state = nnx.eval_shape(cell_fn, module, inputs, initial_state)
  initial_state = jax.tree.map(as_vma, initial_state, state)

  return nnx.scan(
      cell_fn,
      in_axes=(None, 0, nnx.Carry),
      out_axes=(0, nnx.Carry),
  )(module, inputs, initial_state)

def scan_rnn(
    cell_fn: tp.Callable[[InputTree, RecurrentState], tuple[OutputTree, RecurrentState]],
    inputs: InputTree,
    initial_state: RecurrentState,
) -> tuple[OutputTree, RecurrentState]:
  """Like dynamic_rnn but returns all intermediate hidden states.

  Args:
    cell_fn: Function (inputs, state) -> (outputs, new_state)
    inputs: Inputs with time as first axis
    initial_state: Initial recurrent state

  Returns:
    outputs: Stacked outputs over time
    hidden_states: All intermediate hidden states
  """

  unbound_f, bound_self, was_bound = _resolve_bound_callable(cell_fn)

  if not was_bound:
    raise ValueError('scan_rnn requires a bound method.')

  if was_bound:
    def unbound_output_hidden(module, x, state):
      y, state = unbound_f(module, x, state)
      return (y, state), state

    return nnx.scan(in_axes=(None, 0, nnx.Carry), out_axes=(0, nnx.Carry))(
        unbound_output_hidden)(bound_self, inputs, initial_state)[0]

  def output_hidden(x, state):
    y, state = unbound_f(x, state)
    return (y, state), state

  return nnx.scan(in_axes=(0, nnx.Carry), out_axes=(0, nnx.Carry))(
      output_hidden)(inputs, initial_state)[0]

def nonjit_while_loop(
    cond_fn: tp.Callable[[T], jax.Array],
    body_fn: tp.Callable[[T], T],
    init_val: T,
) -> T:
  """Like jax.lax.while_loop but not jitted."""
  val = init_val
  while bool(cond_fn(val)):
    val = body_fn(val)
  return val

Data = tp.TypeVar('Data')
State = tp.TypeVar('State')
GradsT = tp.TypeVar('GradsT')
AuxT = tp.TypeVar('AuxT')
Outputs = tp.TypeVarTuple('Outputs')
Loss = Array
Grads = tp.Any

def no_loss(
    loss_fn: tp.Callable[P, tuple[Loss, *Outputs]],
) -> tp.Callable[P, tuple[*Outputs]]:

  @functools.wraps(loss_fn)
  def wrapped(*args: P.args, **kwargs: P.kwargs) -> tuple[*Outputs]:
    return loss_fn(*args, **kwargs)[1:]

  return wrapped

def grad_with_aux(
    f: tp.Callable[P, tuple[Loss, AuxT]],
    argnums: int | tp.Sequence[int] = 0,
    take_mean: bool = True,
    fp32_grads: bool = True,
    loss_scale: tp.Optional[float] = None,
) -> tp.Callable[P, tuple[Grads, AuxT]]:
  """Adds type signature to nnx.grad."""

  @functools.wraps(f)  # needed to help nnx figure out the type signature
  def g(*args: P.args, **kwargs: P.kwargs):
    loss, aux = f(*args, **kwargs)
    if take_mean:
      loss = jnp.mean(loss)
    if loss_scale is not None:
      loss = loss * loss_scale
    return loss, aux

  _grad_fn = nnx.grad(g, argnums=argnums, has_aux=True)

  def grad_fn(*args: P.args, **kwargs: P.kwargs):
    grad, aux = _grad_fn(*args, **kwargs)
    if fp32_grads:
      grad = jax.tree.map(lambda x: x.astype(jnp.float32), grad)
    if loss_scale is not None:
      grad = jax.tree.map(lambda x: x / loss_scale, grad)
    return grad, aux

  return grad_fn

def grad_with_aux_tuple(
    f: tp.Callable[P, tuple[Loss, *Outputs]],
    take_mean: bool = True,
    fp32_grads: bool = True,
    loss_scale: tp.Optional[float] = None,
) -> tp.Callable[P, tuple[Grads, *Outputs]]:

  @functools.wraps(f)  # needed to help nnx figure out the type signature
  def packed_loss_fn(*args: P.args, **kwargs: P.kwargs):
    outputs = f(*args, **kwargs)
    return outputs[0], outputs[1:]

  packed_grad_fn = grad_with_aux(
      packed_loss_fn, take_mean=take_mean,
      fp32_grads=fp32_grads, loss_scale=loss_scale)

  def grad_fn(*args: P.args, **kwargs: P.kwargs):
    grad, aux = packed_grad_fn(*args, **kwargs)
    return (grad, *aux)

  return grad_fn

def pcast_module(module: ModT, axis_name: str, *, to: str) -> ModT:
  graphdef, state = nnx.split(module)
  pcasted_state = jax.lax.pcast(state, axis_name, to=to)
  return nnx.merge(graphdef, pcasted_state)

def loss_fn_with_mean(
    loss_fn: tp.Callable[P, tp.Tuple[Loss, AuxT]],
    take_pmean: bool = True,
    data_axis: tp.Optional[str] = None,
) -> tp.Callable[P, tp.Tuple[Loss, AuxT]]:

  @functools.wraps(loss_fn)
  def wrapped_loss_fn(*args: P.args, **kwargs: P.kwargs) -> tuple[Loss, AuxT]:
    loss, aux = loss_fn(*args, **kwargs)
    # First take the mean across the device-local batch.
    # loss = jnp.mean(loss, axis=0, keepdims=True)
    loss = jnp.mean(loss)

    if take_pmean:
      if data_axis is None:
        raise ValueError('data_axis must be specified when take_pmean is True.')

      loss = jax.lax.pmean(jnp.expand_dims(loss, axis=0), axis_name=data_axis)[0]

    return loss, aux

  return wrapped_loss_fn


Inputs = tp.TypeVarTuple('Inputs')

def sharded_grads(
    # Note: loss_fn should return loss of shape [B]
    loss_fn: tp.Callable[tp.Concatenate[ModT, P], tuple[Loss, *Outputs]],
    explicit_pmean: bool = True,
    data_axis: str = DATA_AXIS,
):

  def compute_grads(
      module: ModT,
      *args: P.args,
      **kwargs: P.kwargs,
  ) -> tuple[Grads, *Outputs]:
    # TODO: make sure that no nnx objects are in kwargs. If they are, nnx will
    # complain with a trace level mismatch. This is why we pass *args through
    # the loss function and its gradient.

    # TODO: merge with loss_fn_with_mean?
    def packed_loss_fn(module: ModT, *args: P.args):
      # Note: we don't pass kwargs through grad_fn because nnx can't figure out
      # how to handle them; our functions are too generic for the inspect module.
      all_outputs = loss_fn(module, *args, **kwargs)
      return all_outputs[0], all_outputs[1:]

    # If we let shard_map handle gradient communication across devices implicitly,
    # then we need to make sure the loss is averaged across devices inside loss_fn.
    sharded_loss_fn = loss_fn_with_mean(
        packed_loss_fn, take_pmean=not explicit_pmean, data_axis=data_axis)

    grad_fn = grad_with_aux(sharded_loss_fn)

    # This prevents jax from inserting an implicit psum on gradients.
    if explicit_pmean:
      module = pcast_module(module, data_axis, to='varying')

    grads, aux = grad_fn(module, *args)

    if explicit_pmean:
      grads = jax.lax.pmean(grads, axis_name=data_axis)

    return (grads, *aux)

  return compute_grads

def shard_map_grads(
    # Note: loss_fn should return loss of shape [B]
    loss_fn: tp.Callable[tp.Concatenate[ModT, Data, P], tp.Tuple[Loss, AuxT]],
    mesh: jax.sharding.Mesh,
    explicit_pmean: bool = True,
    data_axis: str = DATA_AXIS,
):
  return nnx.shard_map(
      sharded_grads(loss_fn, explicit_pmean, data_axis),
      in_specs=(PS(), PS(data_axis)),
      out_specs=(PS(), PS(data_axis)),
      mesh=mesh,
  )

# Better type hints for functools.partial and nnx.cached_partial
In1 = tp.TypeVar('In1')
In2 = tp.TypeVar('In2')
In3 = tp.TypeVar('In3')
In4 = tp.TypeVar('In4')

@tp.overload
def partial(
    func: tp.Callable[tp.Concatenate[In1, P], T],
    arg1: In1,
) -> tp.Callable[P, T]: ...

@tp.overload
def partial(
    func: tp.Callable[tp.Concatenate[In1, In2, P], T],
    arg1: In1, arg2: In2,
) -> tp.Callable[P, T]: ...

@tp.overload
def partial(
    func: tp.Callable[tp.Concatenate[In1, In2, In3, P], T],
    arg1: In1, arg2: In2, arg3: In3,
) -> tp.Callable[P, T]: ...

def partial(func, *args):  # type: ignore
  return functools.partial(func, *args)

@tp.overload
def cached_partial(
    func: tp.Callable[tp.Concatenate[In1, P], T],
    arg1: In1,
) -> tp.Callable[P, T]: ...

@tp.overload
def cached_partial(
    func: tp.Callable[tp.Concatenate[In1, In2, P], T],
    arg1: In1, arg2: In2,
) -> tp.Callable[P, T]: ...

@tp.overload
def cached_partial(
    func: tp.Callable[tp.Concatenate[In1, In2, In3, P], T],
    arg1: In1, arg2: In2, arg3: In3,
) -> tp.Callable[P, T]: ...

@tp.overload
def cached_partial(
    func: tp.Callable[tp.Concatenate[In1, In2, In3, In4, P], T],
    arg1: In1, arg2: In2, arg3: In3, arg4: In4,
) -> tp.Callable[P, T]: ...


def cached_partial(func, *args):  # type: ignore
  return nnx.cached_partial(func, *args)

F = tp.TypeVar('F')

def _typed_transform(
    transform: tp.Callable[tp.Concatenate[tp.Any, P], tp.Any],
) -> tp.Callable[tp.Concatenate[F, P], F]:
  """Adds type signature to nnx transforms."""
  return transform

jit = _typed_transform(jax.jit)
nnx_jit = _typed_transform(nnx.jit)
shard_map = _typed_transform(jax.shard_map)
annotate_function = _typed_transform(jax.profiler.annotate_function)

device_put = _typed_transform(jax.device_put)
device_get = _typed_transform(jax.device_get)

def grad0(
    f: tp.Callable[tp.Concatenate[T, P], Loss],
) -> tp.Callable[tp.Concatenate[T, P], T]:
  """Gradient of f with respect to the first argument."""
  return jax.grad(f, argnums=0)

def jacrev0(
    f: tp.Callable[tp.Concatenate[T, P], Loss],
) -> tp.Callable[tp.Concatenate[T, P], T]:
  """Jacobian of f with respect to the first argument."""
  return jax.jacrev(f, argnums=0)

def lax_map(
    f: tp.Callable[[In1], T],
    xs: In1,
    *,
    batch_size: tp.Optional[int] = None,
) -> T:
  """Like jax.lax.map but with better type signature."""
  # TODO: support multiple inputs and different axes
  return jax.lax.map(f, xs, batch_size=batch_size)

Out = tp.TypeVar('Out')

def vmap1(
    func: tp.Callable[tp.Concatenate[T, P], Out],
    in_axis: int = 0,
    out_axis: int = 0,
    static_argnames: tp.Optional[tp.Iterable[str]] = None,
    use_jit: bool = True,
):
  """Vmap across just the first argument to a function.

  This is a workaround for vmap forcing kwargs to have axis 0.
  """

  @functools.wraps(func)
  def wrapper(
      arg: T,
      *args: P.args,
      **kwargs: P.kwargs,
  ) -> Out:
    def partial_func(arg: T):
      return func(arg, *args, **kwargs)

    return jax.vmap(
        partial_func,
        in_axes=in_axis, out_axes=out_axis)(arg)

  if use_jit:
    return jit(wrapper, static_argnames=static_argnames)

  return wrapper

def multi_vmap(
    func: tp.Callable[P, Out],
    batch_rank: tp.Optional[int] = None,
    axes: tp.Optional[tp.Sequence[int]] = None,
) -> tp.Callable[P, Out]:
  """Apply vmap across the first batch_rank axes.

  First axis in axes is outermost vmap.
  """
  if batch_rank is not None and axes is not None:
    raise ValueError('Only one of batch_rank and axes can be specified.')

  if batch_rank is not None:
    axes = list(range(batch_rank))

  if axes is None:
    raise ValueError('One of batch_rank and axes must be specified.')

  if len(set(axes)) < len(axes):
    raise ValueError(f'Axes must be unique, got {axes}.')

  n = len(axes)
  modified_axes = list(axes)
  for i in range(n):
    axis = modified_axes[i]
    for j in range(i+1, n):
      if modified_axes[j] > axis:
        modified_axes[j] -= 1

  logging.info(f'multi_vmap: axes={axes}, modified_axes={modified_axes}')

  for axis in reversed(modified_axes):
    func = jax.vmap(func, in_axes=axis, out_axes=axis)

  return func

def get_vma(x) -> set[str]:
  if hasattr(x, 'vma'):
    return x.vma
  t = jax.typeof(x)
  import jax.core as jc
  if isinstance(t, jc.ShapedArray):
    return t.manual_axis_type.varying
  return set()

def as_vma(x: T, ref) -> T:
  if isinstance(ref, jax.ShapeDtypeStruct):
    if ref.manual_axis_type is None:
      return x
    vma = ref.manual_axis_type.varying

  else:
    if not hasattr(ref, 'vma') or ref.vma is None:
      return x
    vma = ref.vma

  axes = tuple(vma - get_vma(x))
  return jax.lax.pcast(x, axes, to='varying')

PackedArg = list[np.ndarray]

class ArraySpec(tp.NamedTuple):
  dtype: np.dtype
  shape: tuple[int, ...]


class ArgPacker[T]:

  def __init__(self, batch_rank: int = 0):
    self.batch_rank = batch_rank
    self.needs_init = True

  def _init(self, arg: T):
    assert self.needs_init, 'ArgPacker can only be initialized once.'
    self.needs_init = False

    self.flat_specs: list[tuple[jax.tree_util.KeyPath, ArraySpec]] = []
    self.flat_slices: list[tuple[int, int]] = []
    self.packed_sizes = collections.defaultdict(lambda: 0)
    leaves, self.structure = jax.tree.flatten_with_path(arg)

    for path, x in leaves:
      if not isinstance(x, np.ndarray):
        raise ValueError(f'All leaves must be numpy arrays, got {type(x)}.')

      spec = ArraySpec(dtype=x.dtype, shape=x.shape)
      self.flat_specs.append((path, spec))
      size = math.prod(x.shape[self.batch_rank:])
      start = self.packed_sizes[x.dtype]
      end = start + size
      self.flat_slices.append((start, end))
      self.packed_sizes[x.dtype] = end

    self.dtypes = list(self.packed_sizes.keys())
    print('dtypes:', self.dtypes)

  def pack(self, arg: T) -> PackedArg:
    if self.needs_init:
      self._init(arg)

    leaves = jax.tree.leaves(arg)
    flattened = {dtype: [] for dtype in self.packed_sizes}

    for (path, spec), x in zip(self.flat_specs, leaves):
      # if not isinstance(x, np.ndarray):
      #   raise ValueError(f'All leaves must be numpy arrays, got {type(x)} at {path}.')

      if x.dtype != spec.dtype:
        raise ValueError(f'Expected dtype {spec.dtype} at {path}, got {x.dtype}.')

      if x.shape != spec.shape:
        raise ValueError(f'Expected shape {spec.shape} at {path}, got {x.shape}.')

      flattened[x.dtype].append(np.reshape(x, (*x.shape[:self.batch_rank], -1)))

    packed = []
    for dtype in self.dtypes:
      packed.append(np.concatenate(flattened[dtype], axis=-1))

    return packed

  def unpack(self, packed_arg: list[np.ndarray] | list[jnp.ndarray]) -> T:
    dtype_to_array = {
        dtype: array
        for dtype, array in zip(self.dtypes, packed_arg)
    }

    flat_leaves = []
    for (_, spec), (start, end) in zip(self.flat_specs, self.flat_slices):
      array = dtype_to_array[spec.dtype][..., start:end]
      flat_leaves.append(array.reshape((*array.shape[:-1], *spec.shape[self.batch_rank:])))

    return jax.tree.unflatten(self.structure, flat_leaves)


def packed_nnx_jit(
    func: tp.Callable[P, T],
    pack_argnums: tp.Sequence[int],
    batch_rank: int = 0,
    **jit_kwargs,
) -> tp.Callable[P, T]:

  packers = [ArgPacker(batch_rank) for _ in pack_argnums]

  @nnx.jit(**jit_kwargs)
  def packed_func(*args, **kwargs) -> T:
    args = list(args)
    for packer, argnum in zip(packers, pack_argnums):
      args[argnum] = packer.unpack(args[argnum])

    return func(*args, **kwargs)  # type: ignore

  @functools.wraps(func)
  def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
    packed_args = list(args)
    for packer, argnum in zip(packers, pack_argnums):
      packed_args[argnum] = packer.pack(packed_args[argnum])

    return packed_func(*packed_args, **kwargs)

  return wrapped

CachedArgs = tp.TypeVarTuple('CachedArgs')

class CachedFunctionalJit(tp.Generic[*CachedArgs, P, T]):

  def __init__(
      self,
      func: tp.Callable[tp.Concatenate[tuple[*CachedArgs], P], T],
      *args: *CachedArgs,
      donate_argnums: tp.Sequence[int] = (),
      **jit_kwargs,
  ):
    self.func = func
    self.donate_argnums = donate_argnums
    self.jit_kwargs = jit_kwargs

    self.graphdefs: list[nnx.GraphDef] = []
    self.pure_states: list[dict] = []

    for arg in args:
      graphdef, state = nnx.split(arg)
      self.graphdefs.append(graphdef)
      self.pure_states.append(state.to_pure_dict())

    def pure_func(
        pure_args: list[dict], *args: P.args, **kwargs: P.kwargs) -> tuple[T, list[dict]]:
      cached_args = self._from_pure_states(pure_args)
      output = func(cached_args, *args, **kwargs)
      pure_out = [nnx.state(arg).to_pure_dict() for arg in cached_args]
      return output, pure_out

    if 0 not in donate_argnums:
      donate_argnums = (0, *donate_argnums)

    # Note: this is regular jax.jit, not nnx.jit
    self.jitted_func = jit(pure_func, donate_argnums=donate_argnums, **jit_kwargs)

  def _from_pure_states(self, pure_states: list[dict], copy: bool = False) -> tuple[*CachedArgs]:
    if len(pure_states) != len(self.graphdefs):
      raise ValueError(f'Length of pure_states {len(pure_states)} must match number of cached args {len(self.graphdefs)}.')

    return tuple(
        # Note: nnx.merge handles both States and pure dicts
        nnx.merge(graphdef, state, copy=copy)
        for graphdef, state in zip(self.graphdefs, pure_states))

  def cached_args(self) -> tuple[*CachedArgs]:
    return self._from_pure_states(self.pure_states)

  def update_pure_state(self, index: int, new_state: dict):
    self.pure_states[index] = new_state

  def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
    output, pure_out = self.jitted_func(self.pure_states, *args, **kwargs)
    self.pure_states = pure_out
    return output

@tp.overload
def cached_functional_jit(
    func: tp.Callable[tp.Concatenate[In1, P], T],
    arg1: In1,
    **jit_kwargs,
) -> CachedFunctionalJit[tuple[In1], P, T]: ...

@tp.overload
def cached_functional_jit(
    func: tp.Callable[tp.Concatenate[In1, In2, P], T],
    arg1: In1,
    arg2: In2,
    **jit_kwargs,
) -> CachedFunctionalJit[tuple[In1, In2], P, T]: ...

@tp.overload
def cached_functional_jit(
    func: tp.Callable[tp.Concatenate[In1, In2, In3, P], T],
    arg1: In1,
    arg2: In2,
    arg3: In3,
    **jit_kwargs,
) -> CachedFunctionalJit[tuple[In1, In2, In3], P, T]: ...

def cached_functional_jit(
    func: tp.Callable[..., T],
    *args,
    donate_argnums: tp.Sequence[int] = (),
    **jit_kwargs,
):
  def packed_fn(cached: tuple, *args, **kwargs):
    return func(*cached, *args, **kwargs)

  offset = len(args) - 1
  donate_argnums = tuple(i - offset for i in donate_argnums)

  return CachedFunctionalJit(
    packed_fn, *args,
    donate_argnums=donate_argnums,
    **jit_kwargs)

# Microbatching utilities.

Axis = int | None | nnx.StateAxes
AxisPrefix = Axis | tuple

def microbatch_array(x: jax.Array, axis: int, microbatch_size: int):
  new_shape = x.shape[:axis] + (-1, microbatch_size) + x.shape[axis+1:]
  return x.reshape(new_shape)

def unmicrobatch_array(x: jax.Array, axis: int):
  """Merges microbatches back into the batch dimension."""
  # NOTE: a None/StateAxes axis doesn't make sense for outputs
  assert len(x.shape) > axis + 1
  new_shape = x.shape[:axis] + (-1,) + x.shape[axis+2:]
  return x.reshape(new_shape)

def _substruct_batch_size(axis: Axis, substruct: tp.Any) -> int | None:
  if isinstance(axis, (type(None), nnx.StateAxes)):
    return None
  leaves = jax.tree.leaves(substruct)
  batch_sizes = set(leaf.shape[axis] for leaf in leaves)
  if len(batch_sizes) == 0:
    return None
  if len(batch_sizes) > 1:
    raise ValueError('Got different batch sizes: ' + str(batch_sizes))
  return batch_sizes.pop()

# We can't use jax.tree.map because it doesn't treat None as leaves, meaning
# that map(None, struct) doesn't work.
prefix_map = functools.partial(jax.tree.map, is_leaf=lambda x: x is None)

def get_batch_size(axis_prefix: AxisPrefix, struct: tp.Any) -> int | None:
  batch_sizes = jax.tree.leaves(prefix_map(
      _substruct_batch_size, axis_prefix, struct))
  batch_sizes = set(size for size in batch_sizes if size is not None)
  if len(batch_sizes) == 0:
    return None
  if len(batch_sizes) > 1:
    raise ValueError('Got different batch sizes: ' + str(batch_sizes))
  return batch_sizes.pop()

def microbatch_substruct(axis: Axis, substruct: T, microbatch_size: int) -> T:
  # TODO: properly handle StateAxes
  if axis is None:
    return substruct
  if isinstance(axis, nnx.StateAxes):
    if axis.filters == (...,) and axis.axes[0] in (None, nnx.Carry):
      return substruct
    raise NotImplementedError('Complex StateAxes not supported for microbatching')
  return jax.tree.map(
      functools.partial(microbatch_array, axis=axis, microbatch_size=microbatch_size),
      substruct)

def microbatch_struct(axis_prefix: AxisPrefix, struct: T, microbatch_size: int) -> T:
  return prefix_map(
      functools.partial(microbatch_substruct, microbatch_size=microbatch_size),
      axis_prefix, struct)

def unmicrobatch_substruct(axis: int, substruct: T) -> T:
  return jax.tree.map(
      functools.partial(unmicrobatch_array, axis=axis),
      substruct)

def unmicrobatch_struct(axis_prefix: AxisPrefix, struct: T) -> T:
  # Technically we don't need to use prefix_map here because axis_prefix should
  # not contain Nones as they don't make sense for outputs.
  return prefix_map(unmicrobatch_substruct, axis_prefix, struct)

def microbatch_fn(
    f: tp.Callable[P, T],  # operates on batched inputs
    microbatch_size: int,  # 0 means no microbatching
    input_batch_dims: int | tuple = 0,
    output_batch_dims: int | tuple = 0,
) -> tp.Callable[P, T]:
  mbs = microbatch_size

  if microbatch_size == 0:
    return f

  def microbatched(*args: P.args, **kwargs: P.kwargs) -> T:
    batch_size = get_batch_size(input_batch_dims, args)

    if batch_size is not None and batch_size % mbs != 0:
      raise ValueError(f'microbatch size {mbs} must divide batch size {batch_size}')

    microbatched_inputs = microbatch_struct(input_batch_dims, args, mbs)

    def with_kwargs(*args: P.args):
      return f(*args, **kwargs)

    scan_fn = nnx.scan(
        with_kwargs,
        in_axes=input_batch_dims,
        out_axes=output_batch_dims,
    )

    microbatched_outputs = scan_fn(*microbatched_inputs)
    return unmicrobatch_struct(output_batch_dims, microbatched_outputs)

  return microbatched

def lax_map_fn(
    f: tp.Callable[P, T],
    microbatch_size: int = 1,
    input_batch_dims: int | tuple = 0,
    output_batch_dims: int | tuple = 0,
) -> tp.Callable[P, T]:
  """Like jax.lax.map but compatible with nnx."""
  # TODO: PR to flax?
  vmapped = nnx.vmap(f, in_axes=input_batch_dims, out_axes=output_batch_dims)
  return microbatch_fn(vmapped, microbatch_size, input_batch_dims, output_batch_dims)

# TODO: replace with microbatch_fn
def microbatch_module(
    f: tp.Callable[tp.Concatenate[ModT, P], T],
    microbatch_size: int,  # 0 means no microbatching
    input_batch_dims: int | tuple = 0,
    output_batch_dims: int | tuple = 0,
) -> tp.Callable[tp.Concatenate[ModT, P], T]:
  mbs = microbatch_size

  if microbatch_size == 0:
    return f

  def microbatched(
      module: ModT, *args: P.args, **kwargs: P.kwargs,
  ) -> T:
    batch_size = get_batch_size(input_batch_dims, args)

    if batch_size is not None and batch_size % mbs != 0:
      raise ValueError(f'microbatch size {mbs} must divide batch size {batch_size}')

    microbatched_inputs = microbatch_struct(input_batch_dims, args, mbs)
    in_axes = input_batch_dims
    if isinstance(in_axes, int):
      in_axes = (in_axes,) * len(args)

    def with_kwargs(module: ModT, *args: P.args):
      return f(module, *args, **kwargs)

    scan_fn = nnx.scan(
        with_kwargs,
        in_axes=(None, *in_axes),
        out_axes=output_batch_dims,
    )

    microbatched_outputs = scan_fn(module, *microbatched_inputs)
    return unmicrobatch_struct(output_batch_dims, microbatched_outputs)

  return microbatched

def microbatched_grads(
    loss_fn: tp.Callable[tp.Concatenate[ModT, P], tuple[Loss, *Outputs]],
    microbatch_size: int,
    input_batch_dims: int | tuple = 0,  # doesn't include module
    output_batch_dims: int | tuple = 0,  # doesn't include Loss
    take_mean: bool = True,
    fp32_grads: bool = True,
    loss_scale: tp.Optional[float] = None,
):
  mbs = microbatch_size
  grad_fn = grad_with_aux_tuple(
      loss_fn, take_mean=take_mean,
      fp32_grads=fp32_grads, loss_scale=loss_scale)

  if microbatch_size == 0:
    return grad_fn

  def compute_grads(
      module: ModT, *args: P.args, **kwargs: P.kwargs,
  ) -> tuple[Grads, *Outputs]:
    batch_size = get_batch_size(input_batch_dims, args)

    if batch_size is None:
      return grad_fn(module, *args, **kwargs)

    num_microbatches, r = divmod(batch_size, mbs)
    if r != 0:
      raise ValueError(f'microbatch size {mbs} must divide batch size {batch_size}')

    logging.info(f'Split inputs into {num_microbatches} microbatches of size {mbs}')
    in_axes = input_batch_dims
    if isinstance(in_axes, int):
      in_axes = (in_axes,) * len(args)
    microbatched_inputs = microbatch_struct(input_batch_dims, args, mbs)

    output_shapes = nnx.eval_shape(grad_fn, module, *args, **kwargs)
    zero_grads = jax.tree.map(jnp.zeros_like, output_shapes[0])

    out_axes = output_batch_dims
    if isinstance(out_axes, int):
      out_axes = (out_axes,) * len(output_shapes[1:])

    @nnx.scan(
        in_axes=(None, nnx.Carry, *in_axes),
        out_axes=(nnx.Carry, *out_axes),
    )
    def scan_fn(module: ModT, grads_acc: Grads, *args: P.args):
      grads_and_outputs = grad_fn(module, *args, **kwargs)
      new_grads_acc: Grads = jax.tree.map(jnp.add, grads_and_outputs[0], grads_acc)
      return (new_grads_acc, *grads_and_outputs[1:])

    grads_and_outputs = scan_fn(module, zero_grads, *microbatched_inputs)

    grads = grads_and_outputs[0]
    if take_mean:
      grads = jax.tree.map(lambda g: g / num_microbatches, grads)

    microbatched_outputs = grads_and_outputs[1:]
    outputs = unmicrobatch_struct(output_batch_dims, microbatched_outputs)

    return (grads, *outputs)

  return compute_grads

def run_loss_fn(
    module: ModT,
    loss_fn: tp.Callable[tp.Concatenate[ModT, Data, State, P], tuple[Loss, AuxT, State, *Outputs]],
    input_batch_dims: int | tuple = 0,
    output_batch_dims: int | tuple = 0,  # doesn't include Loss
    microbatch_size: int = 0,
) -> tp.Callable[tp.Concatenate[Data, State, P], tuple[AuxT, State, *Outputs]]:
  run_fn = no_loss(loss_fn)

  if microbatch_size != 0:
    run_fn = microbatch_module(
        run_fn,
        microbatch_size=microbatch_size,
        input_batch_dims=input_batch_dims,
        output_batch_dims=output_batch_dims)

  jit_run = nnx_jit(run_fn, donate_argnums=(0, 2))
  return cached_partial(jit_run, module)

def train_fn(
    loss_fn: tp.Callable[tp.Concatenate[ModT, P], tuple[Loss, *Outputs]],
    input_batch_dims: int | tuple = 0,
    output_batch_dims: int | tuple = 0,  # doesn't include Loss
    microbatch_size: int = 0,
    fp32_grads: bool = True,
    loss_scale: tp.Optional[float] = None,
) -> tp.Callable[tp.Concatenate[ModT, nnx.Optimizer[ModT], P], tuple[*Outputs]]:

  grad_fn = microbatched_grads(
      loss_fn, microbatch_size,
      input_batch_dims=input_batch_dims,
      output_batch_dims=output_batch_dims,
      fp32_grads=fp32_grads,
      loss_scale=loss_scale,
  )

  def train(
      module: ModT, optimizer: nnx.Optimizer[ModT],
      *args: P.args, **kwargs: P.kwargs,
  ) -> tuple[*Outputs]:
    outputs = grad_fn(module, *args, **kwargs)
    optimizer.update(module, outputs[0])
    return outputs[1:]

  return train

def cached_train_fn(
    module: ModT,
    optimizer: nnx.Optimizer[ModT],
    loss_fn: tp.Callable[tp.Concatenate[ModT, Data, State, P], tuple[Loss, AuxT, State, *Outputs]],
    input_batch_dims: int | tuple = 0,
    output_batch_dims: int | tuple = 0,  # doesn't include Loss
    microbatch_size: int = 0,
) -> tp.Callable[tp.Concatenate[Data, State, P], tuple[AuxT, State, *Outputs]]:

  train = train_fn(loss_fn, input_batch_dims, output_batch_dims, microbatch_size)
  jit_train = nnx_jit(train, donate_argnums=(0, 1, 3))
  return cached_partial(jit_train, module, optimizer)

def train_fn_with_rngs(
    module: ModT,
    optimizer: nnx.Optimizer[ModT],
    rngs: nnx.Rngs,
    loss_fn: tp.Callable[tp.Concatenate[ModT, nnx.Rngs, Data, State, P], tuple[Loss, AuxT, State, *Outputs]],
    input_batch_dims: int | tuple = 0,  # doesn't include Rngs
    output_batch_dims: int | tuple = 0,  # doesn't include Loss
    microbatch_size: int = 0,
) -> tp.Callable[tp.Concatenate[Data, State, P], tuple[AuxT, State, *Outputs]]:

  grad_fn = grad_with_aux_tuple(loss_fn)

  def train(
      module: ModT, optimizer: nnx.Optimizer[ModT], rngs: nnx.Rngs,
      data: Data, state: State, *args: P.args, **kwargs: P.kwargs,
  ) -> tuple[AuxT, State, *Outputs]:
    outputs = grad_fn(module, rngs, data, state, *args, **kwargs)
    optimizer.update(module, outputs[0])
    return outputs[1:]

  jit_train = nnx.jit(train, donate_argnums=(0, 1, 2, 4))

  return cached_partial(jit_train, module, optimizer, rngs)


def data_parallel_train(
    module: ModT,
    optimizer: nnx.Optimizer[ModT],
    loss_fn: tp.Callable[tp.Concatenate[ModT, Data, State, P], tuple[Loss, AuxT, State, *Outputs]],
    mesh: jax.sharding.Mesh,
    data_axis: int = 0,
    data_axis_name: str = DATA_AXIS,
    extra_in_specs: tp.Optional[tp.Sequence[PS]] = None,
    extra_out_specs: tp.Optional[tp.Sequence[PS]] = None,
    static_argnames: tp.Optional[tp.Iterable[str]] = None,
    explicit_pmean: bool = False,
    smap_optimizer: bool = True,
    pack_data: bool = False,
) -> tp.Callable[tp.Concatenate[Data, State, P], tuple[AuxT, State, *Outputs]]:
  if data_axis_name not in mesh.axis_names:
    raise ValueError(f'Axis name {data_axis_name} not in mesh axis names {mesh.axis_names}.')

  def train(
      module: ModT, optimizer: nnx.Optimizer[ModT],
      data: Data, state: State, *args: P.args, **kwargs: P.kwargs,
  ) -> tuple[AuxT, State, *Outputs]:

    if extra_in_specs is None:
      _extra_in_specs = tuple(PS() for _ in args)
    else:
      if len(extra_in_specs) != len(args):
        raise ValueError(f'Length of extra_in_specs {len(extra_in_specs)} does not match number of inputs {len(args)}.')
      _extra_in_specs = tuple(extra_in_specs)

    if extra_out_specs is None:
      _extra_out_specs = tuple()
    else:
      _extra_out_specs = tuple(extra_out_specs)

    if not smap_optimizer:
      raise NotImplementedError('shard_map without sharding the optimizer is not implemented yet.')
      grads, (aux, new_state) = shard_map_grads(
          packed_loss_fn, mesh, explicit_pmean=explicit_pmean, data_axis=data_axis_name)(
              module, (data, state), PSpecCache(*args, **kwargs))

      optimizer.update(module, grads)

      return aux, new_state

    sharded_grads_fn = sharded_grads(
        loss_fn, explicit_pmean=explicit_pmean, data_axis=data_axis_name)

    DS = data_spec(data_axis, name=data_axis_name)

    @nnx.shard_map(
        in_specs=(PS(), PS(), DS, DS) + _extra_in_specs,
        out_specs=(DS, DS) + _extra_out_specs,
        mesh=mesh,
    )
    def update_fn(
        module: ModT,
        optimizer: nnx.Optimizer[ModT],
        data: Data,
        state: State,
        *args: P.args,
        # No kwargs because shard_map doesn't support them
    ) -> tuple[AuxT, State, *Outputs]:
      grads_and_outputs = sharded_grads_fn(module, data, state, *args, **kwargs)
      grads = grads_and_outputs[0]
      optimizer.update(module, grads)
      return grads_and_outputs[1:]

    return update_fn(module, optimizer, data, state, *args)

  jit_train = packed_nnx_jit(
      train,
      pack_argnums=(2,) if pack_data else (),
      donate_argnums=(0, 1, 3),
      static_argnames=static_argnames,
  )

  return cached_partial(jit_train, module, optimizer)

# TODO: share code with data_parallel_train?
def data_parallel_train_with_rngs(
    module: ModT,
    optimizer: nnx.Optimizer[ModT],
    rngs: nnx.Rngs,
    loss_fn: tp.Callable[tp.Concatenate[ModT, Data, State, nnx.Rngs, P], tuple[Loss, AuxT, State, *Outputs]],
    mesh: jax.sharding.Mesh,
    data_axis: int = 0,
    data_axis_name: str = DATA_AXIS,
    extra_in_specs: tp.Optional[tp.Sequence[PS]] = None,
    extra_out_specs: tp.Optional[tp.Sequence[PS]] = None,
    static_argnames: tp.Optional[tp.Iterable[str]] = None,
    explicit_pmean: bool = False,
    smap_optimizer: bool = True,
) -> tp.Callable[tp.Concatenate[Data, State, P], tuple[AuxT, State, *Outputs]]:
  if data_axis_name not in mesh.axis_names:
    raise ValueError(f'Axis name {data_axis_name} not in mesh axis names {mesh.axis_names}.')

  # Shard the Rngs across devices.
  num_shards: int = mesh.shape[data_axis_name]
  rngs = rngs.fork(split=num_shards)
  shard_module(rngs, data_sharding(mesh, data_axis_name))

  @nnx.jit(
      donate_argnums=(0, 1, 2, 4),
      static_argnames=static_argnames,
  )
  def train(
      module: ModT, optimizer: nnx.Optimizer[ModT], rngs: nnx.Rngs,
      data: Data, state: State, *args: P.args, **kwargs: P.kwargs,
  ) -> tuple[AuxT, State, *Outputs]:

    if extra_in_specs is None:
      _extra_in_specs = tuple(PS() for _ in args)
    else:
      if len(extra_in_specs) != len(args):
        raise ValueError(f'Length of extra_in_specs {len(extra_in_specs)} does not match number of inputs {len(args)}.')
      _extra_in_specs = tuple(extra_in_specs)

    if extra_out_specs is None:
      _extra_out_specs = tuple()
    else:
      _extra_out_specs = tuple(extra_out_specs)

    if not smap_optimizer:
      raise NotImplementedError('shard_map without sharding the optimizer is not implemented yet.')
      grads, (aux, new_state) = shard_map_grads(
          packed_loss_fn, mesh, explicit_pmean=explicit_pmean, data_axis=data_axis_name)(
              module, (data, state), PSpecCache(*args, **kwargs))

      optimizer.update(module, grads)

      return aux, new_state

    sharded_grads_fn = sharded_grads(
        loss_fn, explicit_pmean=explicit_pmean, data_axis=data_axis_name)

    DS = data_spec(data_axis, name=data_axis_name)

    @nnx.shard_map(
        in_specs=(PS(), PS(), DS, DS, DS) + _extra_in_specs,
        out_specs=(DS, DS) + _extra_out_specs,
        mesh=mesh,
    )
    def update_fn(
        module: ModT,
        optimizer: nnx.Optimizer[ModT],
        data: Data,
        state: State,
        rngs: nnx.Rngs,  # shape [1]
        *args: P.args,
        # No kwargs because shard_map doesn't support them
    ) -> tuple[AuxT, State, *Outputs]:
      map_update(lambda x: x[0], rngs)  # remove the extra leading axis from rngs
      grads_and_outputs = sharded_grads_fn(module, data, state, rngs, *args, **kwargs)
      map_update(lambda x: x[None], rngs)  # add back leading axis to rngs for next step
      grads = grads_and_outputs[0]
      optimizer.update(module, grads)
      return grads_and_outputs[1:]

    return update_fn(module, optimizer, data, state, rngs, *args)

  return cached_partial(train, module, optimizer, rngs)


def shard_map_loss_fn(
    module: ModT,
    loss_fn: tp.Callable[tp.Concatenate[ModT, Data, State, P], tp.Tuple[Loss, AuxT, State, *Outputs]],
    mesh: jax.sharding.Mesh,
    data_axis: int = 0,
    data_axis_name: str = DATA_AXIS,
    extra_in_specs: tp.Optional[tp.Sequence[PS]] = None,
    extra_out_specs: tp.Optional[tp.Sequence[PS]] = None,
    static_argnames: tp.Optional[tp.Iterable[str]] = None,
):
  """Shard-mapped loss function for data-parallel training."""

  if data_axis_name not in mesh.axis_names:
    raise ValueError(f'Axis name {data_axis_name} not in mesh axis names {mesh.axis_names}.')

  @nnx.jit(
      donate_argnums=(2,),
      static_argnames=static_argnames,
  )
  @functools.wraps(loss_fn)
  def loss_fn_wrapper(module: ModT, data: Data, state: State, *args: P.args, **kwargs: P.kwargs):
    if extra_in_specs is None:
      _extra_in_specs = tuple(PS() for _ in args)
    else:
      if len(extra_in_specs) != len(args):
        raise ValueError(f'Length of extra_in_specs {len(extra_in_specs)} does not match number of inputs {len(args)}.')
      _extra_in_specs = tuple(extra_in_specs)

    if extra_out_specs is None:
      _extra_out_specs = tuple()
    else:
      _extra_out_specs = tuple(extra_out_specs)

    DS = data_spec(data_axis, name=data_axis_name)

    @nnx.shard_map(
        in_specs=(PS(), DS, DS) + _extra_in_specs,
        out_specs=(DS, DS) + _extra_out_specs,
        mesh=mesh,
    )
    def sharded_loss_fn(
        module: ModT,
        data: Data,
        state: State,
        *inputs: P.args,
    ) -> tuple[AuxT, State, *Outputs]:
      loss_and_outputs = loss_fn(module, data, state, *inputs, **kwargs)
      return loss_and_outputs[1:]

    return sharded_loss_fn(module, data, state, *args)

  return cached_partial(loss_fn_wrapper, module)

def shard_map_loss_fn_with_rngs(
    module: ModT,
    rngs: nnx.Rngs,
    loss_fn: tp.Callable[tp.Concatenate[ModT, Data, State, nnx.Rngs, P], tp.Tuple[Loss, AuxT, State, *Outputs]],
    mesh: jax.sharding.Mesh,
    data_axis: int = 0,
    data_axis_name: str = DATA_AXIS,
    extra_in_specs: tp.Optional[tp.Sequence[PS]] = None,
    extra_out_specs: tp.Optional[tp.Sequence[PS]] = None,
    static_argnames: tp.Optional[tp.Iterable[str]] = None,
):
  """Shard-mapped loss function for data-parallel training."""

  if data_axis_name not in mesh.axis_names:
    raise ValueError(f'Axis name {data_axis_name} not in mesh axis names {mesh.axis_names}.')

  # Shard the Rngs across devices.
  num_shards: int = mesh.shape[data_axis_name]
  rngs = rngs.fork(split=num_shards)
  shard_module(rngs, data_sharding(mesh, data_axis_name))

  @nnx.jit(
      donate_argnums=(1, 3),
      static_argnames=static_argnames,
  )
  @functools.wraps(loss_fn)
  def loss_fn_wrapper(module: ModT, rngs: nnx.Rngs, data: Data, state: State, *args: P.args, **kwargs: P.kwargs):
    if extra_in_specs is None:
      _extra_in_specs = tuple(PS() for _ in args)
    else:
      if len(extra_in_specs) != len(args):
        raise ValueError(f'Length of extra_in_specs {len(extra_in_specs)} does not match number of inputs {len(args)}.')
      _extra_in_specs = tuple(extra_in_specs)

    if extra_out_specs is None:
      _extra_out_specs = tuple()
    else:
      _extra_out_specs = tuple(extra_out_specs)

    DS = data_spec(data_axis, name=data_axis_name)

    @nnx.shard_map(
        in_specs=(PS(), DS, DS, DS) + _extra_in_specs,
        out_specs=(DS, DS) + _extra_out_specs,
        mesh=mesh,
    )
    def sharded_loss_fn(
        module: ModT,
        data: Data,
        state: State,
        rngs: nnx.Rngs,
        *inputs: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[AuxT, State, *Outputs]:
      map_update(lambda x: x[0], rngs)  # go from [1] to []
      loss_and_outputs = loss_fn(module, data, state, rngs, *inputs, **kwargs)
      map_update(lambda x: x[None], rngs)  # go back to [1] for shard_map
      return loss_and_outputs[1:]

    return sharded_loss_fn(module, data, state, rngs, *args, **kwargs)

  return cached_partial(loss_fn_wrapper, module, rngs)

def put_into(buffer: T, x: T) -> T:
  return x

def prefetch_data(
    source: tp.Iterator[Data],
    size: int = 1,
    sharding: tp.Optional[NamedSharding] = None,
) -> tp.Iterator[Data]:
  """Prefetch data from a source iterator, optionally sharding it across devices."""
  queue = collections.deque[Data]()

  put_data = jit(put_into, donate_argnums=0, out_shardings=sharding)

  for _ in range(size):
    data = next(source)
    queue.append(device_put(data, sharding))

  for item in source:
    buffer = queue.popleft()
    yield buffer
    queue.append(put_data(buffer, item))

  yield from queue

def _get_spec(x):
  if isinstance(x, np.ndarray):
    return (x.dtype, x.shape)
  elif isinstance(x, jnp.ndarray):
    return (x.dtype, x.shape)
  raise ValueError(f'Unsupported type for _get_spec: {type(x)}')

def jit_no_retrace(f: tp.Callable[P, T]) -> tp.Callable[P, T]:
  jit_f = jit(f)

  signature = None

  def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
    nonlocal signature
    if signature is None:
      signature = jax.tree.map(_get_spec, (args, kwargs))
    else:
      new_signature = jax.tree.map(_get_spec, (args, kwargs))
      errors = utils.check_same_structure(signature, new_signature, equal=True)
      if errors:
        print(errors)
        raise ValueError(f'jit_no_retrace: got new argument types {new_signature}, expected {signature}')

    return jit_f(*args, **kwargs)

  return wrapped

# Misc

def get_process_gpu_memory_gb(target_pid: tp.Optional[int] = None) -> tp.Optional[float]:
  try:
    from pynvml import (
        nvmlInit, nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetComputeRunningProcesses,
    )
  except ImportError:
    return None

  if target_pid is None:
    target_pid = os.getpid()

  nvmlInit()
  try:
    # Get handle for the first GPU (index 0)
    handle = nvmlDeviceGetHandleByIndex(0)

    # Get list of all compute processes on this GPU
    # Note: Use nvmlDeviceGetGraphicsRunningProcesses for graphics apps
    processes = nvmlDeviceGetComputeRunningProcesses(handle)

    for proc in processes:
      if proc.pid == target_pid:
        # usedGpuMemory is returned in bytes
        return proc.usedGpuMemory / 1024**3

    return 0.0 # Process not found on GPU
  finally:
    nvmlShutdown()
