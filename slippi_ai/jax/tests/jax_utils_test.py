"""Tests for jax_utils.shard_map_grads."""

import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import unittest

import jax
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx
import numpy as np

import sys
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jax_utils import (
    shard_map_grads, DATA_AXIS, replicate_module,
    shard_pytree, data_sharding,
)


def _make_mesh(num_devices: int = 2) -> Mesh:
  devices = jax.devices('cpu')[:num_devices]
  return Mesh(devices, (DATA_AXIS,))


def _loss_fn(module: nnx.Linear, data: jax.Array) -> tuple[jax.Array, dict]:
  y = module(data)
  # Per-example losses so aux has a batch dimension for shard_map out_specs.
  per_example_loss = jnp.mean(y ** 2, axis=-1)  # shape [batch]
  return per_example_loss, dict(loss=per_example_loss)


def _single_device_grads(module: nnx.Linear, data: jax.Array):
  def loss_fn(module: nnx.Linear, data: jax.Array):
    loss, aux = _loss_fn(module, data)
    # Take mean across batch for single-device loss.
    return jnp.mean(loss), aux

  grad_fn = nnx.grad(loss_fn, has_aux=True)
  return grad_fn(module, data)


class ShardMapGradsTest(unittest.TestCase):

  def _assert_states_close(self, grads_a, grads_b):
    for leaf_a, leaf_b in zip(
        jax.tree.leaves(grads_a),
        jax.tree.leaves(grads_b),
    ):
      np.testing.assert_allclose(
          np.array(leaf_a), np.array(leaf_b),
          rtol=1e-5, atol=1e-6,
      )

  def test_explicit_pmean(self):
    """shard_map_grads with explicit_pmean should match single-device grads."""
    mesh = _make_mesh(2)
    module = nnx.Linear(3, 5, rngs=nnx.Rngs(0))
    data = jax.random.normal(jax.random.PRNGKey(42), (8, 3))

    ref_grads, ref_aux = _single_device_grads(module, data)

    replicate_module(module, mesh)
    sharded_data = shard_pytree(data, data_sharding(mesh))
    shard_grads, shard_aux = shard_map_grads(
        _loss_fn, mesh, explicit_pmean=True)(module, sharded_data)

    self._assert_states_close(ref_grads, shard_grads)
    np.testing.assert_allclose(
        np.array(ref_aux['loss']),
        np.array(shard_aux['loss']),
        rtol=1e-5, atol=1e-6,
    )

  def test_implicit_pmean(self):
    """explicit_pmean=False should also match single-device grads."""
    mesh = _make_mesh(2)
    module = nnx.Linear(3, 5, rngs=nnx.Rngs(0))
    data = jax.random.normal(jax.random.PRNGKey(42), (8, 3))

    ref_grads, _ = _single_device_grads(module, data)

    replicate_module(module, mesh)
    sharded_data = shard_pytree(data, data_sharding(mesh))
    shard_grads, _ = shard_map_grads(
        _loss_fn, mesh, explicit_pmean=False)(module, sharded_data)

    self._assert_states_close(ref_grads, shard_grads)

  def test_different_batch_sizes(self):
    """Grads should be consistent regardless of per-device batch size."""
    mesh = _make_mesh(2)
    module = nnx.Linear(4, 3, rngs=nnx.Rngs(1))
    key = jax.random.PRNGKey(7)

    for batch_size in [4, 16]:
      with self.subTest(batch_size=batch_size):
        data = jax.random.normal(key, (batch_size, 4))
        ref_grads, _ = _single_device_grads(module, data)

        replicate_module(module, mesh)
        sharded_data = shard_pytree(data, data_sharding(mesh))
        shard_grads, _ = shard_map_grads(
            _loss_fn, mesh, explicit_pmean=True)(module, sharded_data)

        self._assert_states_close(ref_grads, shard_grads)

  def test_four_devices(self):
    """Test with 4 devices to verify correctness beyond 2."""
    mesh = _make_mesh(4)
    module = nnx.Linear(6, 2, rngs=nnx.Rngs(3))
    data = jax.random.normal(jax.random.PRNGKey(99), (16, 6))

    ref_grads, ref_aux = _single_device_grads(module, data)

    replicate_module(module, mesh)
    sharded_data = shard_pytree(data, data_sharding(mesh))
    shard_grads, shard_aux = shard_map_grads(
        _loss_fn, mesh, explicit_pmean=True)(module, sharded_data)

    self._assert_states_close(ref_grads, shard_grads)
    np.testing.assert_allclose(
        np.array(ref_aux['loss']), np.array(shard_aux['loss']),
        rtol=1e-5, atol=1e-6,
    )


if __name__ == '__main__':
  unittest.main()
