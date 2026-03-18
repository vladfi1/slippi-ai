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

from jax_utils import (
    shard_map_grads, DATA_AXIS, replicate_module,
    device_put, data_sharding, ArgPacker,
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
    sharded_data = device_put(data, data_sharding(mesh))
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
    sharded_data = device_put(data, data_sharding(mesh))
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
        sharded_data = device_put(data, data_sharding(mesh))
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
    sharded_data = device_put(data, data_sharding(mesh))
    shard_grads, shard_aux = shard_map_grads(
        _loss_fn, mesh, explicit_pmean=True)(module, sharded_data)

    self._assert_states_close(ref_grads, shard_grads)
    np.testing.assert_allclose(
        np.array(ref_aux['loss']), np.array(shard_aux['loss']),
        rtol=1e-5, atol=1e-6,
    )


class ArgPackerTest(unittest.TestCase):

  def _assert_pytree_equal(self, a, b):
    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    self.assertEqual(len(leaves_a), len(leaves_b))
    for la, lb in zip(leaves_a, leaves_b):
      np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))

  def test_single_array_roundtrip(self):
    packer = ArgPacker()
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    packed = packer.pack(x)
    self.assertEqual(len(packed), 1)  # one dtype
    self._assert_pytree_equal(packer.unpack(packed), x)

  def test_multiple_arrays_same_dtype(self):
    packer = ArgPacker()
    arg = [np.ones((2, 3), dtype=np.float32), np.zeros((2, 5), dtype=np.float32)]
    packed = packer.pack(arg)
    self.assertEqual(len(packed), 1)  # only one dtype
    self._assert_pytree_equal(packer.unpack(packed), arg)

  def test_multiple_dtypes(self):
    packer = ArgPacker()
    arg = {
        'f': np.ones((4, 3), dtype=np.float32),
        'i': np.arange(8, dtype=np.int32).reshape(4, 2),
    }
    packed = packer.pack(arg)
    self.assertEqual(len(packed), 2)
    result = packer.unpack(packed)
    self._assert_pytree_equal(result, arg)

  def test_batch_rank_2(self):
    packer = ArgPacker(batch_rank=2)
    arg = [
        np.ones((2, 3, 4), dtype=np.float32),
        np.zeros((2, 3, 5), dtype=np.float32),
    ]
    packed = packer.pack(arg)
    self.assertEqual(packed[0].shape, (2, 3, 9))
    self._assert_pytree_equal(packer.unpack(packed), arg)

  def test_nested_namedtuple(self):
    import collections
    Point = collections.namedtuple('Point', ['x', 'y'])
    packer = ArgPacker()
    arg = Point(
        x=np.array([1.0, 2.0], dtype=np.float32),
        y=np.array([3.0, 4.0], dtype=np.float32),
    )
    packed = packer.pack(arg)
    result = packer.unpack(packed)
    self.assertIsInstance(result, Point)
    self._assert_pytree_equal(result, arg)

  def test_unpack_jax_arrays(self):
    """unpack should work with jax arrays (e.g. after jit)."""
    packer = ArgPacker()
    arg = np.arange(6, dtype=np.float32).reshape(2, 3)
    packed = packer.pack(arg)
    jax_packed = [jnp.array(p) for p in packed]
    result = packer.unpack(jax_packed)
    self._assert_pytree_equal(result, arg)

  def test_pack_multiple_times(self):
    """pack can be called multiple times after initialization."""
    packer = ArgPacker()
    a = np.ones((3, 4), dtype=np.float32)
    b = np.zeros((3, 4), dtype=np.float32)
    packer.pack(a)
    packed_b = packer.pack(b)
    self._assert_pytree_equal(packer.unpack(packed_b), b)

  def test_non_numpy_raises(self):
    packer = ArgPacker()
    with self.assertRaises(ValueError):
      packer.pack(jnp.ones((2, 3)))

  def test_lazy_init(self):
    packer = ArgPacker()
    self.assertTrue(packer.needs_init)
    packer.pack(np.ones((2, 3), dtype=np.float32))
    self.assertFalse(packer.needs_init)

  def test_preserves_values(self):
    """Packed/unpacked values should be numerically identical."""
    rng = np.random.default_rng(0)
    packer = ArgPacker()
    arg = {
        'a': rng.standard_normal((5, 3)).astype(np.float32),
        'b': rng.integers(0, 100, (5, 7)).astype(np.int32),
        'c': rng.standard_normal((5, 2)).astype(np.float32),
    }
    packed = packer.pack(arg)
    result = packer.unpack(packed)
    self._assert_pytree_equal(result, arg)


if __name__ == '__main__':
  unittest.main()
