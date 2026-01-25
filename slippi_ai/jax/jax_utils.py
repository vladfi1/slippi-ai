"""JAX utilities."""

from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx

Array = jax.Array


# Multi-device utilities

def get_mesh(axis_name: str = 'data') -> Mesh:
  """Create a 1D device mesh for data parallelism."""
  return Mesh(jax.devices(), (axis_name,))


def replicate_sharding(mesh: Mesh) -> NamedSharding:
  """Create a sharding that replicates data across all devices."""
  return NamedSharding(mesh, P())


def data_sharding(mesh: Mesh, axis_name: str = 'data') -> NamedSharding:
  """Create a sharding that splits the first axis across devices."""
  return NamedSharding(mesh, P(axis_name))


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
