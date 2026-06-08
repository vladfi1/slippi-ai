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

from slippi_ai.jax.jax_utils import (
    shard_map_grads, DATA_AXIS, replicate_module,
    device_put, data_sharding, ArgPacker,
    microbatch_module, microbatched_grads, grad_with_aux_tuple,
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


def _simple_loss_fn(module: nnx.Linear, data: jax.Array) -> tuple[jax.Array, dict]:
  """Loss fn returning per-example losses and a dict aux."""
  y = module(data)
  per_example = jnp.sum(y ** 2, axis=-1)  # shape [B]
  return per_example, dict(loss=per_example)


def _reference_grads(module: nnx.Linear, data: jax.Array):
  """Return (grads, aux_dict) using grad_with_aux_tuple as the baseline.

  grad_with_aux_tuple splats the aux tuple, so the return is (grads, *aux).
  _simple_loss_fn has one aux element (a dict), giving (grads, dict).
  """
  grad_fn = grad_with_aux_tuple(_simple_loss_fn)
  return grad_fn(module, data)


class MicrobatchFnTest(unittest.TestCase):
  """microbatch_fn(f, mbs)(module, data) must equal f(module, data)."""

  def _assert_close(self, a, b):
    for la, lb in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
      np.testing.assert_allclose(np.array(la), np.array(lb), rtol=1e-5, atol=1e-6)

  def _forward_fn(self, module: nnx.Linear, data: jax.Array):
    return module(data)

  def test_matches_direct_call(self):
    """microbatch_fn output should equal direct f(module, data)."""
    module = nnx.Linear(4, 3, rngs=nnx.Rngs(0))
    data = jax.random.normal(jax.random.PRNGKey(0), (8, 4))

    ref = self._forward_fn(module, data)
    mb = microbatch_module(self._forward_fn, microbatch_size=2)(module, data)

    self._assert_close(ref, mb)

  def test_batch_size_equals_microbatch_size(self):
    """microbatch_size == batch_size means one microbatch; should match directly."""
    module = nnx.Linear(3, 2, rngs=nnx.Rngs(1))
    data = jax.random.normal(jax.random.PRNGKey(1), (4, 3))

    ref = self._forward_fn(module, data)
    mb = microbatch_module(self._forward_fn, microbatch_size=4)(module, data)

    self._assert_close(ref, mb)

  def test_various_microbatch_sizes(self):
    """Any divisor microbatch size should produce the same output."""
    module = nnx.Linear(5, 3, rngs=nnx.Rngs(2))
    data = jax.random.normal(jax.random.PRNGKey(2), (12, 5))
    ref = self._forward_fn(module, data)

    for mbs in [1, 2, 3, 4, 6, 12]:
      with self.subTest(mbs=mbs):
        mb = microbatch_module(self._forward_fn, microbatch_size=mbs)(module, data)
        self._assert_close(ref, mb)

  def test_preserves_output_shape(self):
    """Output shape should equal the un-batched output shape."""
    module = nnx.Linear(4, 3, rngs=nnx.Rngs(3))
    data = jax.random.normal(jax.random.PRNGKey(3), (6, 4))
    ref = self._forward_fn(module, data)
    mb = microbatch_module(self._forward_fn, microbatch_size=2)(module, data)
    self.assertEqual(mb.shape, ref.shape)

  def test_multi_output(self):
    """Works when f returns a tuple of outputs."""
    module = nnx.Linear(4, 3, rngs=nnx.Rngs(4))
    data = jax.random.normal(jax.random.PRNGKey(4), (8, 4))

    def f(module, x):
      y = module(x)
      return y, jnp.sum(y, axis=-1)

    ref_y, ref_s = f(module, data)
    mb_y, mb_s = microbatch_module(f, microbatch_size=4)(module, data)

    self._assert_close(ref_y, mb_y)
    self._assert_close(ref_s, mb_s)

  def test_non_zero_axis(self):
    """in_axes/out_axes can specify a non-zero batch axis."""
    module = nnx.Linear(3, 2, rngs=nnx.Rngs(5))
    # time-major: shape [T, B, F]; batch along axis 1
    data = jax.random.normal(jax.random.PRNGKey(5), (6, 4, 3))

    def f(module, x):
      # x: [T, B, F] → apply Linear over last dim
      return jax.vmap(jax.vmap(module))(x)

    ref = f(module, data)
    mb = microbatch_module(f, microbatch_size=2, input_batch_dims=1, output_batch_dims=1)(module, data)
    self._assert_close(ref, mb)


class MicrobatchedGradsTest(unittest.TestCase):

  def _assert_grads_close(self, grads_a, grads_b):
    for la, lb in zip(jax.tree.leaves(grads_a), jax.tree.leaves(grads_b)):
      np.testing.assert_allclose(
          np.array(la), np.array(lb), rtol=1e-5, atol=1e-5)

  def test_matches_reference_grads(self):
    """microbatched_grads should match grad_with_aux_tuple."""
    module = nnx.Linear(4, 3, rngs=nnx.Rngs(0))
    data = jax.random.normal(jax.random.PRNGKey(10), (8, 4))

    ref_grads, _ref_aux = _reference_grads(module, data)
    mb_grads, _mb_aux = microbatched_grads(_simple_loss_fn, microbatch_size=4)(module, data)

    self._assert_grads_close(ref_grads, mb_grads)

  def test_microbatch_size_one(self):
    """microbatch_size=1 (maximum splitting) should still match reference."""
    module = nnx.Linear(3, 2, rngs=nnx.Rngs(1))
    data = jax.random.normal(jax.random.PRNGKey(11), (4, 3))

    ref_grads, _ref_aux = _reference_grads(module, data)
    mb_grads, _mb_aux = microbatched_grads(_simple_loss_fn, microbatch_size=1)(module, data)

    self._assert_grads_close(ref_grads, mb_grads)

  def test_microbatch_size_equals_batch(self):
    """microbatch_size == batch_size means one microbatch; should match exactly."""
    module = nnx.Linear(5, 2, rngs=nnx.Rngs(2))
    data = jax.random.normal(jax.random.PRNGKey(12), (6, 5))

    ref_grads, _ref_aux = _reference_grads(module, data)
    mb_grads, _mb_aux = microbatched_grads(_simple_loss_fn, microbatch_size=6)(module, data)

    self._assert_grads_close(ref_grads, mb_grads)

  def test_various_microbatch_sizes(self):
    """All valid microbatch sizes should produce the same grads."""
    module = nnx.Linear(4, 3, rngs=nnx.Rngs(3))
    data = jax.random.normal(jax.random.PRNGKey(13), (12, 4))

    ref_grads, _ref_aux = _reference_grads(module, data)

    for mbs in [1, 2, 3, 4, 6, 12]:
      with self.subTest(mbs=mbs):
        mb_grads, _mb_aux = microbatched_grads(_simple_loss_fn, microbatch_size=mbs)(module, data)
        self._assert_grads_close(ref_grads, mb_grads)

  def test_aux_outputs_concatenated(self):
    """Per-example aux values should be stacked across microbatches."""
    module = nnx.Linear(3, 2, rngs=nnx.Rngs(4))
    data = jax.random.normal(jax.random.PRNGKey(14), (8, 3))

    # Reference: full-batch aux (per-example losses, shape [8])
    _ref_grads, ref_aux = _reference_grads(module, data)
    # Microbatched: aux should still cover all 8 examples
    _mb_grads, mb_aux = microbatched_grads(_simple_loss_fn, microbatch_size=2)(module, data)

    np.testing.assert_allclose(
        np.array(ref_aux['loss']), np.array(mb_aux['loss']), rtol=1e-5, atol=1e-5)

  def test_bad_microbatch_size_raises(self):
    """Non-divisor microbatch_size should raise ValueError."""
    module = nnx.Linear(3, 2, rngs=nnx.Rngs(5))
    data = jax.random.normal(jax.random.PRNGKey(15), (7, 3))

    with self.assertRaises(ValueError):
      microbatched_grads(_simple_loss_fn, microbatch_size=4)(module, data)

  def test_grads_are_fp32(self):
    """fp32_grads=True (default) should return float32 gradients."""
    module = nnx.Linear(4, 2, rngs=nnx.Rngs(6))
    data = jax.random.normal(jax.random.PRNGKey(16), (8, 4))

    mb_grads, _ = microbatched_grads(_simple_loss_fn, microbatch_size=2)(module, data)
    for leaf in jax.tree.leaves(mb_grads):
      self.assertEqual(leaf.dtype, jnp.float32)


if __name__ == '__main__':
  unittest.main()
