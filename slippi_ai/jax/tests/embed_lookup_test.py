"""The one-hot lookup must match concatenate-then-Linear (embed_lookup).

Builds small bf16 tx_like policies with the simple and the enhanced embed and
compares the unrolled output and every parameter gradient with the lookup on
and off, on random inputs (for the simple embed including out-of-range one-hot
classes). On GPU this runs the real kernel; without a GPU it runs in Pallas
interpret mode.
"""
# Shapes use the usual math names (T, B, H).
# pylint: disable=invalid-name
import os
import typing as tp
import unittest
from unittest import mock

from parameterized import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from slippi_ai.jax import embed as embed_lib
from slippi_ai.jax import embed_lookup, networks

ON_GPU = jax.default_backend() == 'gpu'
INTERPRET = {} if ON_GPU else {'SLIPPI_PALLAS_INTERPRET': '1'}


# Enhanced embed options: the defaults, and those of the newest public models.
ENHANCED = {
    'enhanced': {},
    'enhanced_v6': dict(use_learned_char=False, use_learned_action=False, use_self_nana=False),
}


def _small_policy(hidden_size: int, embed: str = 'simple', options: tp.Optional[dict] = None):
  config = networks.default_config()
  config['name'] = 'tx_like'
  config['tx_like'].update(hidden_size=hidden_size, num_layers=1, ffw_multiplier=2)
  if embed == 'simple':
    config['embed']['name'] = 'simple'
  else:
    config['embed']['name'] = 'enhanced'
    config['embed']['enhanced'].update(
        hidden_size=32, **(ENHANCED[embed] if options is None else options))
  net = networks.build_embed_network(
      rngs=nnx.Rngs(0), embed_config=embed_lib.EmbedConfig(), num_names=8,
      frame_skip=1, network_config=config)
  graphdef, state = nnx.split(net)
  state = jax.tree.map(lambda p: p.astype(jnp.bfloat16), state)
  return graphdef, state


def _random_inputs(embedding, shape, seed, out_of_range=3):
  rng = np.random.default_rng(seed)

  def leaf(e, x):
    x = np.asarray(x)
    if isinstance(e, embed_lib.OneHotEmbedding):
      # Includes classes past the end (all-zero one-hot). The enhanced embed's
      # learned tables give NaN for those, so its tests use in-range classes.
      return rng.integers(0, e.size + out_of_range, size=shape).astype(x.dtype)
    if x.dtype == np.bool_:
      return rng.random(shape) < 0.5
    return (rng.standard_normal(shape) * 3).astype(x.dtype)

  return embedding.map(leaf, embedding.dummy(shape))


def _rel(a, b):
  a = np.asarray(a, np.float32)
  b = np.asarray(b, np.float32)
  return np.abs(a - b).max() / max(np.abs(a).max(), 1e-6)


class EmbedLookupTest(unittest.TestCase):

  def test_sparse_embed_reassembles(self):
    graphdef, state = _small_policy(128)
    embedding = nnx.merge(graphdef, state)._embed_module._embed_state_action  # pylint: disable=protected-access
    sa = _random_inputs(embedding, (5, 3), seed=0)
    sparse = embed_lookup.embed(embedding, sa)
    np.testing.assert_array_equal(np.asarray(sparse.to_dense()), np.asarray(embedding(sa)))
    self.assertGreater(int((sparse.idx < 0).sum()), 0)  # out-of-range classes

  @parameterized.expand(
      list(ENHANCED.items())
      + [('flat_items', dict(use_item_sum=False)), ('hybrid', dict(hybrid_embed=True))])
  def test_enhanced_parts_reassemble(self, _, options):
    graphdef, state = _small_policy(128, 'enhanced', options)
    module = nnx.merge(graphdef, jax.tree.map(lambda p: p.astype(jnp.float32), state))._embed_module  # pylint: disable=protected-access
    sa = _random_inputs(module._embed_state_action, (5, 3), seed=0, out_of_range=0)  # pylint: disable=protected-access
    sparse = embed_lookup.from_parts(module.parts(sa))
    self.assertEqual(sparse.layout.size, module.output_size)
    np.testing.assert_array_equal(np.asarray(sparse.to_dense()), np.asarray(module(sa)))

  @parameterized.expand(
      [((5, 3), 128, embed) for embed in ('simple', *ENHANCED)]
      + ([((8, 16), 256, embed) for embed in ('simple', *ENHANCED)] if ON_GPU else []))
  def test_unroll_matches_plain(self, shape, hidden_size, embed):
    graphdef, state = _small_policy(hidden_size, embed)
    net = nnx.merge(graphdef, state)
    embedding = net._embed_module._embed_state_action  # pylint: disable=protected-access
    sa = _random_inputs(embedding, shape, seed=1, out_of_range=3 if embed == 'simple' else 0)
    reset = jnp.zeros(shape, bool).at[2, 1].set(True)
    init = net.initial_state(shape[1], nnx.Rngs(0))
    g = jax.random.normal(jax.random.PRNGKey(1), (*shape, net.output_size))

    def run(lookup):
      env = dict(INTERPRET, SLIPPI_EMBED_LOOKUP='1' if lookup else '0')
      with mock.patch.dict(os.environ, env):
        # Fresh functions per run: the switch is read at trace time.
        def loss(state):
          out, _ = nnx.merge(graphdef, state).unroll(sa, reset, init)
          return jnp.sum(out.astype(jnp.float32) * g), out
        jaxpr = str(jax.make_jaxpr(lambda s: loss(s)[0])(state))
        self.assertEqual('embed_lookup' in jaxpr, lookup)  # the pallas_call's name
        (_, out), grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(state)
        # Finish before mock.patch.dict restores os.environ: setenv racing
        # XLA's worker threads' getenv can segfault.
        return jax.block_until_ready((out, grads))

    out0, grads0 = run(False)
    out1, grads1 = run(True)
    self.assertLess(_rel(out0, out1), 2e-2)
    flat0 = jax.tree_util.tree_leaves_with_path(grads0)
    flat1 = jax.tree.leaves(grads1)
    self.assertEqual(len(flat0), len(flat1))
    for (path, a), b in zip(flat0, flat1):
      self.assertLess(_rel(a, b), 2e-2, jax.tree_util.keystr(path))

  @unittest.skipUnless(ON_GPU, 'Pallas interpret mode does not support check_vma')
  def test_linear_under_shard_map(self):
    """The trainer runs under shard_map: weights replicated, data sharded."""
    graphdef, state = _small_policy(128)
    embedding = nnx.merge(graphdef, state)._embed_module._embed_state_action  # pylint: disable=protected-access
    T, B, H = 6, 8, 128  # 48 rows: a padded partial tile
    x = embed_lookup.embed(embedding, _random_inputs(embedding, (T, B), seed=3))
    x = x.astype(jnp.bfloat16)
    keys = jax.random.split(jax.random.PRNGKey(4), 3)
    kernel = (jax.random.normal(keys[0], (x.layout.size, H)) * 0.1).astype(jnp.bfloat16)
    bias = jax.random.normal(keys[1], (H,)).astype(jnp.bfloat16)
    g = jax.random.normal(keys[2], (T, B, H))

    def grads(fn):
      def loss(kernel, bias, dense, idx, g):
        y = fn(embed_lookup.SparseEmbedded(dense, idx, x.layout), kernel, bias)
        return jnp.sum(y.astype(jnp.float32) * g), y
      def f(*args):
        (_, y), dparams = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(*args)
        return y, dparams
      return f

    lookup = embed_lookup.linear
    plain = lambda xs, k, b: xs.to_dense() @ k + b
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ('data',))
    P = jax.sharding.PartitionSpec
    data = P(None, 'data')
    sharded = jax.jit(jax.shard_map(
        grads(lookup), mesh=mesh, in_specs=(P(), P(), data, data, data),
        out_specs=(data, P())))
    y1, (dk1, db1) = sharded(kernel, bias, x.dense, x.idx, g)
    y0, (dk0, db0) = jax.jit(grads(plain))(kernel, bias, x.dense, x.idx, g)
    for a, b in ((y0, y1), (dk0, dk1), (db0, db1)):
      self.assertLess(_rel(a, b), 2e-2)


if __name__ == '__main__':
  unittest.main()
