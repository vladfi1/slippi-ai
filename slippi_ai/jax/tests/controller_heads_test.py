"""The autoregressive head's one-action path must match its scan path.

With one action per network step, AutoRegressive skips the last component's
network step (nothing reads it). Its distances, logits and parameter gradients
must equal the first action of the general (scanned) path.
"""
# Shapes use the usual math names (T, B, H).
# pylint: disable=invalid-name
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from slippi_ai.jax import controller_heads
from slippi_ai.jax import embed as embed_lib


def _controllers(embedding, shape, seed):
  rng = np.random.default_rng(seed)

  def leaf(e, x):
    return rng.integers(0, e.size, size=shape).astype(np.asarray(x).dtype)

  return embedding.map(leaf, embedding.dummy(shape))


class AutoRegressiveTest(unittest.TestCase):

  def test_one_action_matches_scan(self):
    config = controller_heads.default_config()
    config['name'] = 'autoregressive'
    component = config['autoregressive']['component']
    component['name'] = 'lstm'
    component['lstm']['hidden_size'] = 32
    embed_config = embed_lib.EmbedConfig()
    embed_config.controller.type = embed_lib.ControllerType.CUSTOM_V1
    embed_controller = embed_config.make_controller_embedding()
    head = controller_heads.construct(
        rngs=nnx.Rngs(0), input_size=48, frame_skip=1,
        embed_controller=embed_controller, **config)
    self.assertIsInstance(head, controller_heads.AutoRegressive)
    graphdef, state = nnx.split(head)

    T, B = 4, 3
    inputs = jax.random.normal(jax.random.PRNGKey(0), (T, B, 48))
    prev = [_controllers(embed_controller, (T, B), seed) for seed in (1, 2)]
    target = [_controllers(embed_controller, (T, B), seed) for seed in (3, 4)]

    def run(num_actions):
      def loss(state):
        outs = nnx.merge(graphdef, state).distance_outputs(
            inputs, prev[:num_actions], target[:num_actions])
        return sum(jnp.sum(d) for d in jax.tree.leaves(outs[0].distance)), outs[0]
      jaxpr = str(jax.make_jaxpr(lambda s: loss(s)[0])(state))
      self.assertEqual('scan' in jaxpr or 'while' in jaxpr, num_actions > 1)
      (_, out), grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(state)
      return out, grads

    out1, grads1 = run(1)  # the one-action path
    out2, grads2 = run(2)  # the scan; only the first action enters the loss
    for a, b in zip(jax.tree.leaves(out2), jax.tree.leaves(out1)):
      np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)
    paths = jax.tree_util.tree_leaves_with_path(grads2)
    leaves1 = jax.tree.leaves(grads1)
    self.assertEqual(len(paths), len(leaves1))
    nonzero = 0
    for (path, a), b in zip(paths, leaves1):
      np.testing.assert_allclose(
          np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6,
          err_msg=jax.tree_util.keystr(path))
      nonzero += bool(np.any(np.asarray(a) != 0))
    self.assertGreater(nonzero, 0)


if __name__ == '__main__':
  unittest.main()
