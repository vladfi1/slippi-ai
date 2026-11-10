"""The fused residual feed-forward block must match the plain XLA path.

Compares a bf16 ResBlock (LayerNorm, Linear, GELU, Linear, residual) with and
without ffw_pallas: the output and every gradient. On GPU this runs the real
kernels, including at the tx_like trainer's widths; without a GPU it runs a
small block in Pallas interpret mode.
"""
# Shapes use the usual math names (D, F).
# pylint: disable=invalid-name
import os
import unittest
from unittest import mock

from parameterized import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from slippi_ai.jax import ffw_pallas, networks

ON_GPU = jax.default_backend() == 'gpu'

if ON_GPU:
  CASES = [
      ((84, 64), 1024, 4096),  # the trainer's widths, fewer rows
      ((8, 128), 256, 512),
  ]
else:
  CASES = [((2, 128), 128, 256)]


class FfwPallasTest(unittest.TestCase):

  @parameterized.expand(CASES)
  def test_resblock_matches_xla(self, lead, D, F):
    env = {} if ON_GPU else {'SLIPPI_PALLAS_INTERPRET': '1'}
    with mock.patch.dict(os.environ, env):
      self.assertTrue(ffw_pallas.supported(int(np.prod(lead)), D, F, jnp.bfloat16))

    block = networks.ResBlock(
        nnx.Rngs(0), D, F, activation=lambda x: nnx.gelu(x, approximate=True),
        fused_gelu=True)
    # Non-trivial parameters: the output projection initialises to zero.
    keys = jax.random.split(jax.random.PRNGKey(1), 5)
    block.linear2.kernel[...] = jax.random.normal(keys[0], (F, D)) / np.sqrt(F)
    block.linear1.bias[...] = jax.random.normal(keys[1], (F,)) * 0.1
    block.linear2.bias[...] = jax.random.normal(keys[2], (D,)) * 0.1
    block.layernorm.scale[...] = 1 + jax.random.normal(keys[3], (D,)) * 0.2
    block.layernorm.bias[...] = jax.random.normal(keys[4], (D,)) * 0.2
    graphdef, state = nnx.split(block)
    state = jax.tree.map(lambda p: p.astype(jnp.bfloat16), state)
    # A residual stream with a non-zero mean and scale.
    x = (jax.random.normal(jax.random.PRNGKey(3), (*lead, D)) * 2 + 0.5).astype(jnp.bfloat16)
    g = jax.random.normal(jax.random.PRNGKey(4), (*lead, D)).astype(jnp.float32)

    def run(fused):
      # Fresh functions per run: the kernel switch is read at trace time, so
      # a cached trace from the other run must not be reused.
      def forward(state, x):
        return nnx.merge(graphdef, state)(x)

      def loss(state, x):
        return jnp.sum(forward(state, x).astype(jnp.float32) * g)

      env = {'SLIPPI_FFW_PALLAS': '1' if fused else '0'}
      if not ON_GPU:
        env['SLIPPI_PALLAS_INTERPRET'] = '1'
      with mock.patch.dict(os.environ, env):
        uses_kernels = 'pallas_call' in str(jax.make_jaxpr(forward)(state, x))
        self.assertEqual(uses_kernels, fused)
        y = jax.jit(forward)(state, x)
        dstate, dx = jax.jit(jax.grad(loss, argnums=(0, 1)))(state, x)
      # The block's own contribution to the input gradient, without the
      # identity path's g, which would otherwise dominate the comparison.
      return [y, *jax.tree.leaves(dstate), dx.astype(jnp.float32) - g]

    for i, (want, got) in enumerate(zip(run(False), run(True))):
      want = np.asarray(want, np.float32)
      got = np.asarray(got, np.float32)
      rel = np.abs(want - got).max() / max(np.abs(want).max(), 1e-6)
      self.assertLess(rel, 3e-2, f'output {i}: relative error {rel:.2e}')


if __name__ == '__main__':
  unittest.main()
