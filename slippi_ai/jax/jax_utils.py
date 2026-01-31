"""JAX utilities."""

import functools
import os
import typing as tp
import types

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS
from flax import nnx
from flax.nnx.transforms.transforms import _resolve_bound_callable

Array = jax.Array


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


def shard_module(module: nnx.Module, sharding: NamedSharding):
  """Shard/replicate module parameters across devices in-place."""
  _, state = nnx.split(module)
  sharded_state = jax.tree.map(
      lambda x: jax.device_put(x, sharding), state)
  nnx.update(module, sharded_state)


def replicate_module(module: nnx.Module, mesh: Mesh):
  """Replicate module parameters across all devices in the mesh."""
  shard_module(module, replicate_sharding(mesh))


def shard_pytree(pytree, sharding: NamedSharding):
  """Shard a pytree of arrays with the given sharding."""
  def shard_leaf(x):
    return jax.device_put(x, sharding)
  return jax.tree.map(shard_leaf, pytree)


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

# Flax NNX

class MLP(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      input_size: int,
      features: list[int],
      activation=nnx.relu,
      activation_final: bool = False,
  ):
    layers = []
    in_size = input_size
    for i, out_size in enumerate(features):
      if i > 0:
        layers.append(activation)
      layer = nnx.Linear(in_size, out_size, rngs=rngs)
      layers.append(layer)
      in_size = out_size

    if activation_final:
      layers.append(activation)

    self.layers = nnx.List(layers)

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

P = tp.ParamSpec('P')
T = tp.TypeVar('T')

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

def pcast_module(module: ModT, axis_name: str, *, to: str) -> ModT:
  graphdef, state = nnx.split(module)
  pcasted_state = jax.lax.pcast(state, axis_name, to=to)
  return nnx.merge(graphdef, pcasted_state)

def data_parallel_train(
    module: ModT,
    optimizer: nnx.Optimizer[ModT],
    loss_fn: tp.Callable[tp.Concatenate[ModT, Data, State, P], tp.Tuple[Loss, AuxT, State]],
    mesh: jax.sharding.Mesh,
    data_axis: str = DATA_AXIS,
    static_argnames: tp.Optional[tp.Iterable[str]] = None,
    shard_module: bool = False,
) -> tp.Callable[tp.Concatenate[Data, State, P], tuple[AuxT, State]]:
  if data_axis not in mesh.axis_names:
    raise ValueError(f'Axis name {data_axis} not in mesh axis names {mesh.axis_names}.')

  @nnx.jit(
      donate_argnums=(0, 1, 3),
      static_argnames=static_argnames,
  )
  def train(
      module: ModT, optimizer: nnx.Optimizer[ModT],
      data: Data, state: State, *args: P.args, **kwargs: P.kwargs) -> tuple[AuxT, State]:
    # shard_map the loss function over devices

    @nnx.grad(has_aux=True)
    def grad_fn(module: ModT, data: Data, state: State) -> tuple[Loss, tuple[AuxT, State]]:
      loss, aux, state = loss_fn(module, data, state, *args, **kwargs)
      # If the module is replicated, we need to average the loss across devices,
      # otherwise the gradients will be multiplied by the number of devices.
      if not shard_module:
        loss = jax.lax.pmean(loss[None], axis_name=data_axis)[0]
      return loss, (aux, state)

    # Just for type checking; nnx.grad should have better type annotations.
    grad_fn = tp.cast(tp.Callable[[ModT, Data, State], tuple[Grads, tuple[AuxT, State]]], grad_fn)

    @nnx.shard_map(
        in_specs=(PS(), PS(), PS(data_axis), PS(data_axis)),
        out_specs=(PS(data_axis), PS(data_axis)),
        mesh=mesh,
    )
    def update_fn(
        module: ModT,
        optimizer: nnx.Optimizer[ModT],
        data: Data,
        state: State,
    ) -> tuple[AuxT, State]:
      if shard_module:
        sharded_module = pcast_module(module, data_axis, to='varying')
        grads, (aux, new_state) = grad_fn(sharded_module, data, state)
        grads = jax.lax.pmean(grads, axis_name=data_axis)
      else:
        grads, (aux, new_state) = grad_fn(module, data, state)

      # Update optimizer independently on each device. This should keep the
      # variables and optimizer states in sync since grads are the same on
      # each device.
      optimizer.update(module, grads)
      return aux, new_state

    return update_fn(module, optimizer, data, state)

  return nnx.cached_partial(train, module, optimizer)

def shard_map_loss_fn(
    module: ModT,
    loss_fn: tp.Callable[tp.Concatenate[ModT, Data, State, P], tp.Tuple[Loss, AuxT, State]],
    mesh: jax.sharding.Mesh,
    data_axis: str = DATA_AXIS,
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

    @nnx.shard_map(
        in_specs=(PS(), PS(data_axis), PS(data_axis)),
        out_specs=(PS(data_axis), PS(data_axis)),
        mesh=mesh,
    )
    def sharded_loss_fn(
        module: ModT,
        data: Data,
        state: State,
    ) -> tuple[AuxT, State]:
      _, aux, state = loss_fn(module, data, state, *args, **kwargs)
      return aux, state

    return sharded_loss_fn(module, data, state)

  return nnx.cached_partial(loss_fn_wrapper, module)

# Misc

def get_process_gpu_memory_gb(target_pid: tp.Optional[int] = None) -> float:
  from pynvml import (
      nvmlInit, nvmlShutdown,
      nvmlDeviceGetHandleByIndex,
      nvmlDeviceGetComputeRunningProcesses,
  )

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
