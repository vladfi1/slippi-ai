"""JAX utilities."""

import functools
import os
import typing as tp
import types

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS
from flax import nnx
from flax.nnx.transforms.transforms import _resolve_bound_callable

Array = jax.Array

P = tp.ParamSpec('P')
T = tp.TypeVar('T')

# Multi-device utilities

DATA_AXIS = 'data'

def get_mesh(axis_name: str = DATA_AXIS) -> Mesh:
  """Create a 1D device mesh for data parallelism."""
  return Mesh(jax.devices(), (axis_name,))


def replicate_sharding(mesh: Mesh) -> NamedSharding:
  """Create a sharding that replicates data across all devices."""
  return NamedSharding(mesh, PS())


def data_sharding(mesh: Mesh, axis_name: str = 'data') -> NamedSharding:
  """Create a sharding that splits the first axis across devices."""
  return NamedSharding(mesh, PS(axis_name))

def shard_pytree(pytree: T, sharding: NamedSharding) -> T:
  """Shard a pytree of arrays with the given sharding."""
  def shard_leaf(x):
    return jax.device_put(x, sharding)
  return jax.tree.map(shard_leaf, pytree)

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


# Other utilities

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
  while cond.ndim < x.ndim:
    cond = jnp.expand_dims(cond, -1)
  return jnp.where(cond, x, y)


def swap_axes(t, axis1=0, axis2=1):
  """Swap two axes of a tensor."""
  permutation = list(range(len(t.shape)))
  permutation[axis2] = axis1
  permutation[axis1] = axis2
  return jnp.transpose(t, permutation)

def add_n(xs: tp.Iterable[Array]) -> Array:
  xs_iter = iter(xs)
  total = next(xs_iter)
  for x in xs_iter:
    total += x
  return total

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

class MLP(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      input_size: int,
      features: list[int],
      activation=nnx.relu,
      activate_final: bool = False,
  ):
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

  def __call__(self, x):
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
  return scan_method(cell_fn)(inputs, initial_state)


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


Data = tp.TypeVar('Data')
State = tp.TypeVar('State')
GradsT = tp.TypeVar('GradsT')
AuxT = tp.TypeVar('AuxT')
Loss = Array
ModT = tp.TypeVar('ModT', bound=nnx.Module)
Grads = tp.Any


def grad_with_aux(
    f: tp.Callable[P, tp.Tuple[Loss, AuxT]],
    argnums: int | tp.Sequence[int] = 0,
) -> tp.Callable[P, tuple[Grads, AuxT]]:
  """Adds type signature to nnx.grad."""
  return nnx.grad(f, argnums=argnums, has_aux=True)


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
    loss = jnp.mean(loss, axis=0, keepdims=True)

    if take_pmean:
      if data_axis is None:
        raise ValueError('data_axis must be specified when take_pmean is True.')

      loss = jax.lax.pmean(loss, axis_name=data_axis)

    return loss[0], aux

  return wrapped_loss_fn


Inputs = tp.TypeVarTuple('Inputs')
Outputs = tp.TypeVarTuple('Outputs')

def sharded_grads(
    # Note: loss_fn should return loss of shape [B]
    loss_fn: tp.Callable[tp.Concatenate[ModT, P], tuple[Loss, *Outputs]],
    explicit_pmean: bool = True,
    data_axis: str = DATA_AXIS,
):

  # TODO: merge with loss_fn_with_mean?
  def packed_loss_fn(module: ModT, *args: P.args, **kwargs: P.kwargs):
    all_outputs = loss_fn(module, *args, **kwargs)
    return all_outputs[0], all_outputs[1:]

  # If we let shard_map handle gradient communication across devices implicitly,
  # then we need to make sure the loss is averaged across devices inside loss_fn.
  sharded_loss_fn = loss_fn_with_mean(
      packed_loss_fn, take_pmean=not explicit_pmean, data_axis=data_axis)

  grad_fn = grad_with_aux(sharded_loss_fn)

  def compute_grads(
      module: ModT,
      *args: P.args,
      **kwargs: P.kwargs,
  ) -> tuple[Grads, *Outputs]:
    # This prevents jax from inserting an implicit psum on gradients.
    if explicit_pmean:
      module = pcast_module(module, data_axis, to='varying')

    grads, aux = grad_fn(module, *args, **kwargs)

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

# Better type hints for nnx.cached_partial
In1 = tp.TypeVar('In1')
In2 = tp.TypeVar('In2')
In3 = tp.TypeVar('In3')

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

def cached_partial(func, *args):  # type: ignore
  return nnx.cached_partial(func, *args)

# TODO: accept kwargs?
def data_parallel_train(
    module: ModT,
    optimizer: nnx.Optimizer[ModT],
    loss_fn: tp.Callable[[ModT, Data, State, *Inputs], tuple[Loss, AuxT, State, *Outputs]],
    mesh: jax.sharding.Mesh,
    data_axis: str = DATA_AXIS,
    extra_in_specs: tp.Optional[tp.Sequence[PS]] = None,
    extra_out_specs: tp.Optional[tp.Sequence[PS]] = None,
    static_argnames: tp.Optional[tp.Iterable[str]] = None,
    explicit_pmean: bool = False,
    smap_optimizer: bool = True,
) -> tp.Callable[[Data, State, *Inputs], tuple[AuxT, State, *Outputs]]:
  if data_axis not in mesh.axis_names:
    raise ValueError(f'Axis name {data_axis} not in mesh axis names {mesh.axis_names}.')

  @nnx.jit(
      donate_argnums=(0, 1, 3),
      static_argnames=static_argnames,
  )
  def train(
      module: ModT, optimizer: nnx.Optimizer[ModT],
      data: Data, state: State, *args: *Inputs,
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
          packed_loss_fn, mesh, explicit_pmean=explicit_pmean, data_axis=data_axis)(
              module, (data, state), PSpecCache(*args, **kwargs))

      optimizer.update(module, grads)

      return aux, new_state

    sharded_grads_fn = sharded_grads(
        loss_fn, explicit_pmean=explicit_pmean, data_axis=data_axis)

    @nnx.shard_map(
        in_specs=(PS(), PS(), PS(data_axis), PS(data_axis)) + _extra_in_specs,
        out_specs=(PS(data_axis), PS(data_axis)) + _extra_out_specs,
        mesh=mesh,
    )
    def update_fn(
        module: ModT,
        optimizer: nnx.Optimizer[ModT],
        data: Data,
        state: State,
        *inputs: *Inputs,
    ) -> tuple[AuxT, State, *Outputs]:
      grads_and_outputs = sharded_grads_fn(module, data, state, *inputs)
      grads = grads_and_outputs[0]
      optimizer.update(module, grads)
      return grads_and_outputs[1:]

    return update_fn(module, optimizer, data, state, *args)

  return cached_partial(train, module, optimizer)

# TODO: share code with data_parallel_train?
def data_parallel_train_with_rngs(
    module: ModT,
    optimizer: nnx.Optimizer[ModT],
    rngs: nnx.Rngs,
    loss_fn: tp.Callable[[ModT, Data, State, nnx.Rngs, *Inputs], tuple[Loss, AuxT, State, *Outputs]],
    mesh: jax.sharding.Mesh,
    data_axis: str = DATA_AXIS,
    extra_in_specs: tp.Optional[tp.Sequence[PS]] = None,
    extra_out_specs: tp.Optional[tp.Sequence[PS]] = None,
    static_argnames: tp.Optional[tp.Iterable[str]] = None,
    explicit_pmean: bool = False,
    smap_optimizer: bool = True,
) -> tp.Callable[[Data, State, *Inputs], tuple[AuxT, State, *Outputs]]:
  if data_axis not in mesh.axis_names:
    raise ValueError(f'Axis name {data_axis} not in mesh axis names {mesh.axis_names}.')

  # Shard the Rngs across devices.
  num_shards: int = mesh.shape[data_axis]
  rngs = rngs.fork(split=num_shards)
  shard_module(rngs, data_sharding(mesh, data_axis))

  @nnx.jit(
      donate_argnums=(0, 1, 2, 4),
      static_argnames=static_argnames,
  )
  def train(
      module: ModT, optimizer: nnx.Optimizer[ModT], rngs: nnx.Rngs,
      data: Data, state: State, *args: *Inputs,
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
          packed_loss_fn, mesh, explicit_pmean=explicit_pmean, data_axis=data_axis)(
              module, (data, state), PSpecCache(*args, **kwargs))

      optimizer.update(module, grads)

      return aux, new_state

    sharded_grads_fn = sharded_grads(
        loss_fn, explicit_pmean=explicit_pmean, data_axis=data_axis)

    @nnx.shard_map(
        in_specs=(PS(), PS(), PS(data_axis), PS(data_axis), PS(data_axis)) + _extra_in_specs,
        out_specs=(PS(data_axis), PS(data_axis)) + _extra_out_specs,
        mesh=mesh,
    )
    def update_fn(
        module: ModT,
        optimizer: nnx.Optimizer[ModT],
        data: Data,
        state: State,
        rngs: nnx.Rngs,  # shape [1]
        *inputs: *Inputs,
    ) -> tuple[AuxT, State, *Outputs]:
      map_update(lambda x: x[0], rngs)  # remove the extra leading axis from rngs
      grads_and_outputs = sharded_grads_fn(module, data, state, rngs, *inputs)
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
    data_axis: str = DATA_AXIS,
    extra_in_specs: tp.Optional[tp.Sequence[PS]] = None,
    extra_out_specs: tp.Optional[tp.Sequence[PS]] = None,
    static_argnames: tp.Optional[tp.Iterable[str]] = None,
):
  """Shard-mapped loss function for data-parallel training."""

  if data_axis not in mesh.axis_names:
    raise ValueError(f'Axis name {data_axis} not in mesh axis names {mesh.axis_names}.')

  @nnx.jit(
      donate_argnums=(2,),
      static_argnames=static_argnames,
  )
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

    @nnx.shard_map(
        in_specs=(PS(), PS(data_axis), PS(data_axis)) + _extra_in_specs,
        out_specs=(PS(data_axis), PS(data_axis)) + _extra_out_specs,
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
    data_axis: str = DATA_AXIS,
    extra_in_specs: tp.Optional[tp.Sequence[PS]] = None,
    extra_out_specs: tp.Optional[tp.Sequence[PS]] = None,
    static_argnames: tp.Optional[tp.Iterable[str]] = None,
):
  """Shard-mapped loss function for data-parallel training."""

  if data_axis not in mesh.axis_names:
    raise ValueError(f'Axis name {data_axis} not in mesh axis names {mesh.axis_names}.')

  # Shard the Rngs across devices.
  num_shards: int = mesh.shape[data_axis]
  rngs = rngs.fork(split=num_shards)
  shard_module(rngs, data_sharding(mesh, data_axis))

  @nnx.jit(
      donate_argnums=(1, 3),
      static_argnames=static_argnames,
  )
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

    @nnx.shard_map(
        in_specs=(PS(), PS(data_axis), PS(data_axis), PS(data_axis)) + _extra_in_specs,
        out_specs=(PS(data_axis), PS(data_axis)) + _extra_out_specs,
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
