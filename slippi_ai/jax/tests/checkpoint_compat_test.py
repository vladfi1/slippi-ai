"""Checks that policies from different code generations still load and run.

Covers checkpoints produced before the frame-skip refactor (residual
autoregressive controller head, no frame_skip config) as well as current ones.
"""

import unittest

import jax
import numpy as np

from slippi_ai import paths
from slippi_ai.jax import saving, agents, controller_heads

POLICY_CHECKPOINTS = {
    'legacy_imitation': paths.JAX_IMITATION_CHECKPOINT_LEGACY,
    'legacy_policy': paths.JAX_POLICY_CHECKPOINT_LEGACY,
    'imitation': paths.JAX_IMITATION_CHECKPOINT,
    'policy': paths.JAX_POLICY_CHECKPOINT,
}

BATCH_SIZE = 2
NUM_STEPS = 7  # not a multiple of any frame skip, to exercise buffering


class CheckpointCompatTest(unittest.TestCase):

  def _run_agent(self, path, batch_steps: int):
    state = saving.load_state_from_disk(str(path))
    policy = saving.load_policy_from_state(state)

    agent = agents.BasicAgent(
        policy, batch_size=BATCH_SIZE, name_code=0, compile=True)
    agent.warmup()

    dummy_game = policy.network.dummy((BATCH_SIZE,)).state
    needs_reset = np.full([BATCH_SIZE], False)

    outputs = []
    if batch_steps == 0:
      for _ in range(NUM_STEPS):
        outputs.append(agent.step(dummy_game, needs_reset))
    else:
      states = [(dummy_game, needs_reset)] * batch_steps
      for _ in range(NUM_STEPS):
        outputs.extend(agent.multi_step(states))

    for so in outputs:
      controller = policy.controller_head.decode_controller(so.controller_state)
      leaves = jax.tree.leaves(controller)
      self.assertTrue(leaves)
      for leaf in leaves:
        self.assertEqual(np.asarray(leaf).shape[:1], (BATCH_SIZE,))

    return policy, outputs

  def test_step(self):
    for name, path in POLICY_CHECKPOINTS.items():
      with self.subTest(name):
        policy, outputs = self._run_agent(path, batch_steps=0)
        self.assertEqual(len(outputs), NUM_STEPS)

  def test_multi_step(self):
    for name, path in POLICY_CHECKPOINTS.items():
      with self.subTest(name):
        policy, outputs = self._run_agent(path, batch_steps=3)
        self.assertEqual(len(outputs), NUM_STEPS * 3)

  def test_legacy_head_type(self):
    state = saving.load_state_from_disk(str(paths.JAX_POLICY_CHECKPOINT_LEGACY))
    policy = saving.load_policy_from_state(state)
    self.assertIsInstance(
        policy.controller_head, controller_heads.ResidualAutoRegressive)
    self.assertEqual(policy.frame_skip, 1)

    state = saving.load_state_from_disk(str(paths.JAX_POLICY_CHECKPOINT))
    policy = saving.load_policy_from_state(state)
    self.assertIsInstance(policy.controller_head, controller_heads.AutoRegressive)

  def test_multi_step_matches_step(self):
    """With frame skip, batched and unbatched stepping must agree."""
    state = saving.load_state_from_disk(str(paths.JAX_POLICY_CHECKPOINT))
    policy = saving.load_policy_from_state(state)
    self.assertGreater(policy.frame_skip, 1)

    dummy_game = policy.network.dummy((BATCH_SIZE,)).state
    needs_reset = np.full([BATCH_SIZE], False)

    def run(batch_steps: int):
      agent = agents.BasicAgent(
          policy, batch_size=BATCH_SIZE, name_code=0, compile=True, seed=0)
      outputs = []
      if batch_steps == 0:
        for _ in range(NUM_STEPS):
          outputs.append(agent.step(dummy_game, needs_reset))
      else:
        states = [(dummy_game, needs_reset)] * batch_steps
        while len(outputs) < NUM_STEPS:
          outputs.extend(agent.multi_step(states))
      return [jax.tree.map(np.asarray, o.logits) for o in outputs[:NUM_STEPS]]

    single = run(0)
    batched = run(2)
    # Later components and steps condition on sampled actions, whose random
    # streams differ between the two paths, so only the first component of the
    # first step is deterministic.
    embedding = policy.controller_head.controller_embedding
    x = next(embedding.flatten(single[0]))
    y = next(embedding.flatten(batched[0]))
    np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
  unittest.main()
