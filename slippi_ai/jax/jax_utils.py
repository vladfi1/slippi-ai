"""JAX utilities."""

import os
from typing import Tuple
import typing as tp
import types

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS
from flax import nnx

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

def mean_and_variance(xs: Array) -> Tuple[Array, Array]:
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


Data = tp.TypeVar('Data')
GradsT = tp.TypeVar('GradsT')
AuxT = tp.TypeVar('AuxT')
Loss = Array
ModT = tp.TypeVar('ModT', bound=nnx.Module)
Grads = tp.Any


def data_parallel_train(
    module: ModT,
    optimizer: nnx.Optimizer[ModT],
    loss_fn: tp.Callable[tp.Concatenate[ModT, Data, P], tp.Tuple[Loss, AuxT]],
    mesh: jax.sharding.Mesh,
    data_axis: str = DATA_AXIS,
    static_argnames: tp.Optional[tp.Iterable[str]] = None,
) -> tp.Callable[tp.Concatenate[Data, P], AuxT]:
  if data_axis not in mesh.axis_names:
    raise ValueError(f'Axis name {data_axis} not in mesh axis names {mesh.axis_names}.')

  @nnx.jit(
      donate_argnums=(0, 1),
      static_argnames=static_argnames,
  )
  def train(
      module: ModT, optimizer: nnx.Optimizer[ModT],
      data: Data, *args: P.args, **kwargs: P.kwargs) -> AuxT:
    # pmap the loss function over devices

    @nnx.grad(has_aux=True)
    def grad_fn(module: ModT, data: Data) -> tp.Tuple[Loss, AuxT]:
      return loss_fn(module, data, *args, **kwargs)

    grad_fn = tp.cast(tp.Callable[[ModT, Data], tuple[AuxT, Grads]], grad_fn)

    @nnx.shard_map(
        in_specs=(PS(), PS(), PS(data_axis)),
        out_specs=PS(data_axis),
        mesh=mesh,
    )
    def update_fn(
        module: ModT,
        optimizer: nnx.Optimizer[ModT],
        data: Data,
    ):
      grads, aux = grad_fn(module, data)
      grads = jax.lax.pmean(grads, axis_name=data_axis)
      optimizer.update(module, grads)
      return aux

    return update_fn(module, optimizer, data)

  return nnx.cached_partial(train, module, optimizer)

def shard_map_loss_fn(
    module: ModT,
    loss_fn: tp.Callable[tp.Concatenate[ModT, Data, P], tp.Tuple[Loss, AuxT]],
    mesh: jax.sharding.Mesh,
    data_axis: str = DATA_AXIS,
    static_argnames: tp.Optional[tp.Iterable[str]] = None,
):
  """Shard-mapped loss function for data-parallel training."""

  if data_axis not in mesh.axis_names:
    raise ValueError(f'Axis name {data_axis} not in mesh axis names {mesh.axis_names}.')

  @nnx.jit(static_argnames=static_argnames)
  def loss_fn_wrapper(module: ModT, data: Data, *args: P.args, **kwargs: P.kwargs):

    @nnx.shard_map(
        in_specs=(PS(), PS(data_axis)),
        out_specs=PS(data_axis),
        mesh=mesh,
    )
    def sharded_loss_fn(
        module: ModT,
        data: Data,
    ) -> AuxT:
      _, aux = loss_fn(module, data, *args, **kwargs)
      return aux

    return sharded_loss_fn(module, data)

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
