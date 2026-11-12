"""lstm_pallas's kernels must match a frame-by-frame lax.scan reference.

Checks outputs, final state and every gradient (gx, Wh, b, h0, c0 and the
residual input) with resets, a non-zero initial state and cotangents on the
final state. On GPU this runs the real kernels at the imitation trainer's
shape and at shapes that take the planner's other tilings; without a GPU it
runs small shapes in Pallas interpret mode (a single program, no barriers),
which checks the kernels' math but not their tiling.
"""
# Shapes and tile sizes use the usual math names (T, B, H, BM, cT, ...).
# pylint: disable=invalid-name
import os
import unittest
from unittest import mock

from parameterized import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from slippi_ai.jax import lstm_pallas

ON_GPU = jax.default_backend() == 'gpu'

if ON_GPU:
  CASES = [
      (84, 512, 1024, jnp.bfloat16, False),  # the imitation trainer's shape
      (84, 512, 1024, jnp.bfloat16, True),   # ... as a residual layer
      (37, 64, 1024, jnp.bfloat16, False),   # long-unroll batch
      (21, 48, 256, jnp.bfloat16, True),     # rows that only tile by 16
      (12, 32, 128, jnp.float32, False),
  ]
else:
  CASES = [
      (6, 8, 16, jnp.float32, False),
      (6, 8, 16, jnp.float32, True),
  ]


def _reference(gx, wh, b, h0, c0, reset):
  def step(carry, xs):
    c, h = carry
    gx_t, r = xs
    c = jnp.where(r[:, None], 0, c)
    h = jnp.where(r[:, None], 0, h)
    z = gx_t.astype(jnp.float32) + b.astype(jnp.float32) + jnp.dot(
        h.astype(jnp.float32), wh.astype(jnp.float32), precision='highest')
    i, f, g, o = jnp.split(z, 4, axis=-1)
    c2 = jax.nn.sigmoid(f) * c.astype(jnp.float32) + jax.nn.sigmoid(i) * jnp.tanh(g)
    h2 = jax.nn.sigmoid(o) * jnp.tanh(c2)
    return (c2.astype(c.dtype), h2.astype(h.dtype)), h2.astype(h.dtype)

  (cT, hT), ys = jax.lax.scan(step, (c0, h0), (gx, reset))
  return ys, (cT, hT)


class LstmPallasTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    if not ON_GPU:
      patcher = mock.patch.dict(os.environ, {'SLIPPI_PALLAS_INTERPRET': '1'})
      patcher.start()
      self.addCleanup(patcher.stop)

  @parameterized.expand(CASES)
  def test_matches_scan(self, T, B, H, dtype, residual):
    self.assertTrue(lstm_pallas.supported(B, H))
    ks = jax.random.split(jax.random.PRNGKey(0), 10)
    gx = (jax.random.normal(ks[0], (T, B, 4 * H)) * 0.5).astype(dtype)
    wh = (jax.random.normal(ks[1], (H, 4 * H)) / np.sqrt(H)).astype(dtype)
    h0 = jnp.tanh(jax.random.normal(ks[2], (B, H))).astype(dtype)
    c0 = jax.random.normal(ks[3], (B, H)).astype(dtype)
    reset = (jax.random.uniform(ks[4], (T, B)) < 0.05).at[3, 0].set(True)
    dy = jax.random.normal(ks[5], (T, B, H)).astype(dtype)
    dcT = jax.random.normal(ks[6], (B, H))
    dhT = jax.random.normal(ks[7], (B, H))
    b = (jax.random.normal(ks[8], (4 * H,)) * 0.5).astype(dtype)
    # The residual input; small, so that rounding h + r to bf16 stays well
    # below the tolerance on h. Unused when residual=False.
    r = (jax.random.normal(ks[9], (T, B, H)) * (0.25 if residual else 0.0)).astype(dtype)

    def kernel(gx, wh, b, h0, c0, r):
      reset_i = reset.astype(jnp.int32)
      if residual:
        return lstm_pallas.lstm_recurrence_residual(gx, wh, b, h0, c0, reset_i, r)
      return lstm_pallas.lstm_recurrence(gx, wh, b, h0, c0, reset_i)

    def reference(gx, wh, b, h0, c0, r):
      ys, final = _reference(gx, wh, b, h0, c0, reset)
      return (r + ys if residual else ys), final

    def run(fn):
      def loss(*args):
        ys, (cT, hT) = fn(*args)
        return (jnp.sum(ys.astype(jnp.float32) * dy.astype(jnp.float32))
                + jnp.sum(cT.astype(jnp.float32) * dcT)
                + jnp.sum(hT.astype(jnp.float32) * dhT))
      args = (gx, wh, b, h0, c0, r)
      ys, (cT, hT) = jax.jit(fn)(*args)
      grads = jax.jit(jax.grad(loss, argnums=tuple(range(6))))(*args)
      if residual:  # compare the recurrent part of the output
        ys = ys.astype(jnp.float32) - r.astype(jnp.float32)
      return [ys, cT, hT, *grads]

    tol = 3e-2 if dtype == jnp.bfloat16 else 1e-4
    names = ['ys', 'cT', 'hT', 'dgx', 'dwh', 'db', 'dh0', 'dc0', 'dr']
    for name, want, got in zip(names, run(reference), run(kernel)):
      want = np.asarray(want, np.float32)
      got = np.asarray(got, np.float32)
      rel = np.abs(want - got).max() / max(np.abs(want).max(), 1e-6)
      self.assertLess(rel, tol, f'{name}: relative error {rel:.2e}')


if __name__ == '__main__':
  unittest.main()
