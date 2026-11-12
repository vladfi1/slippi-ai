"""The compacted item MLP (item_sum) must match the full one.

Builds a small tx_like policy with the enhanced embed and compares the
unrolled output and every parameter gradient with SLIPPI_ITEM_COMPACT on and
off, with few items (the compacted branch) and with more items than the
buffer holds (the full fallback).
"""
# Shapes use the usual math names (T, B).
# pylint: disable=invalid-name
import os
import unittest
from unittest import mock

from parameterized import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from slippi_ai.jax import embed as embed_lib
from slippi_ai.jax import networks


def _small_policy(dtype):
  config = networks.default_config()
  config['name'] = 'tx_like'
  config['tx_like'].update(hidden_size=64, num_layers=1, ffw_multiplier=2)
  config['embed']['name'] = 'enhanced'
  config['embed']['enhanced'].update(hidden_size=32)
  net = networks.build_embed_network(
      rngs=nnx.Rngs(0), embed_config=embed_lib.EmbedConfig(), num_names=8,
      frame_skip=1, network_config=config)
  graphdef, state = nnx.split(net)
  state = jax.tree.map(lambda p: p.astype(dtype), state)
  return graphdef, state


def _random_inputs(embedding, shape, items_per_frame, seed):
  rng = np.random.default_rng(seed)

  def leaf(e, x):
    x = np.asarray(x)
    if isinstance(e, embed_lib.OneHotEmbedding):
      return rng.integers(0, e.size, size=shape).astype(x.dtype)
    if x.dtype == np.bool_:
      return rng.random(shape) < 0.5
    return (rng.standard_normal(shape) * 3).astype(x.dtype)

  sa = embedding.map(leaf, embedding.dummy(shape))
  items = sa.state.items
  p = items_per_frame / len(items)
  items = type(items)(*[
      item._replace(exists=rng.random(shape) < p) for item in items])
  return sa._replace(state=sa.state._replace(items=items))


def _rel(a, b):
  a = np.asarray(a, np.float32)
  b = np.asarray(b, np.float32)
  return np.abs(a - b).max() / max(np.abs(b).max(), 1e-6)


class ItemSumTest(unittest.TestCase):

  def _run(self, dtype, items_per_frame, compact, shard=False):
    graphdef, state = _small_policy(dtype)
    net = nnx.merge(graphdef, state)
    embedding = net._embed_module._embed_state_action  # pylint: disable=protected-access
    T, B = 6, 8
    sa = _random_inputs(embedding, (T, B), items_per_frame, seed=1)
    reset = jnp.zeros((T, B), bool).at[2, 1].set(True)
    init = net.initial_state(B, nnx.Rngs(0))
    g = jax.random.normal(jax.random.PRNGKey(1), (T, B, net.output_size))

    with mock.patch.dict(os.environ, SLIPPI_ITEM_COMPACT='1' if compact else '0'):
      def loss(state, sa, reset, init, g):
        out, _ = nnx.merge(graphdef, state).unroll(sa, reset, init)
        return jnp.sum(out.astype(jnp.float32) * g), out
      jaxpr = str(jax.make_jaxpr(lambda *a: loss(*a)[0])(state, sa, reset, init, g))
      self.assertEqual('cond[' in jaxpr, compact)
      def fn(*args):
        (_, out), grads = jax.value_and_grad(loss, has_aux=True)(*args)
        return out, grads
      if shard:
        # As in training: parameters replicated, data sharded; the parameter
        # gradients must come back replicated.
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ('data',))
        P = jax.sharding.PartitionSpec
        fn = jax.shard_map(
            fn, mesh=mesh,
            in_specs=(P(), P(None, 'data'), P(None, 'data'), P('data'), P(None, 'data')),
            out_specs=(P(None, 'data'), P()))
      out, grads = jax.jit(fn)(state, sa, reset, init, g)
      # Finish before mock.patch.dict restores os.environ: setenv racing
      # XLA's worker threads' getenv can segfault.
      jax.block_until_ready((out, grads))
    return out, grads

  def _compare(self, dtype, items_per_frame, tol, shard=False):
    out0, grads0 = self._run(dtype, items_per_frame, compact=False)
    out1, grads1 = self._run(dtype, items_per_frame, compact=True, shard=shard)
    self.assertLess(_rel(out1, out0), tol)
    flat0 = jax.tree_util.tree_leaves_with_path(grads0)
    flat1 = jax.tree.leaves(grads1)
    self.assertEqual(len(flat0), len(flat1))
    item_grads = 0
    for (path, b), a in zip(flat0, flat1):
      self.assertLess(_rel(a, b), tol, jax.tree_util.keystr(path))
      item_grads += '_item_mlp' in jax.tree_util.keystr(path) and bool(np.any(np.asarray(b) != 0))
    self.assertGreater(item_grads, 0)

  @parameterized.expand([
      ('compact_f32', jnp.float32, 0.7, 1e-5),
      ('compact_bf16', jnp.bfloat16, 0.7, 2e-2),
      ('fallback_f32', jnp.float32, 6.0, 1e-5),  # more than 2 items per frame
  ])
  def test_matches_full(self, _, dtype, items_per_frame, tol):
    with jax.default_matmul_precision('highest'):
      self._compare(dtype, items_per_frame, tol)

  def test_under_shard_map(self):
    with jax.default_matmul_precision('highest'):
      self._compare(jnp.float32, 0.7, 1e-5, shard=True)


if __name__ == '__main__':
  unittest.main()
