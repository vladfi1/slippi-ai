"""Tests for the action-chain helpers in slippi_ai.jax.nash.utils."""

import unittest

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from slippi_ai import data, paths
from slippi_ai.jax import saving
from slippi_ai.jax.nash import utils as nash_utils


def _index_frames(num_states: int, batch_size: int, frame_skip: int):
  """Frames whose every leaf holds its own time index."""
  time = np.arange(num_states, dtype=np.int32)
  time = np.broadcast_to(time[:, np.newaxis], (num_states, batch_size))
  state_action = data.StateAction(
      state=time,
      action=[time + 100 * i for i in range(frame_skip)],
      name=time,
      rating=(),
  )
  return data.Frames(
      state_action=state_action,
      is_resetting=np.zeros((num_states, batch_size), dtype=bool),
      reward=time[:-1].astype(np.float32),
  )


class ChainAlignmentTest(unittest.TestCase):

  def test_time_slices(self):
    x = np.arange(10)
    slices = nash_utils.time_slices(x, 4, range(1, 3))
    np.testing.assert_array_equal(slices[0], [1, 2, 3, 4])
    np.testing.assert_array_equal(slices[1], [2, 3, 4, 5])

  def test_context_and_taken_chain(self):
    unroll_length, skip_delay, batch_size, frame_skip = 6, 2, 3, 2
    num_valid = unroll_length - skip_delay
    # T + 1 states/actions after the delayed slice.
    frames = _index_frames(unroll_length + 1, batch_size, frame_skip)
    t = np.broadcast_to(
        np.arange(num_valid)[:, np.newaxis], (num_valid, batch_size))

    outputs = np.arange(unroll_length)
    hidden = {'h': np.arange(unroll_length) + 1000}
    context = nash_utils.chain_context(outputs, hidden, frames, skip_delay)

    # Everything starts at index t - Ds, i.e. at t' for valid index t = t' + Ds.
    np.testing.assert_array_equal(context.outputs, np.arange(num_valid))
    np.testing.assert_array_equal(context.hidden_states['h'], np.arange(num_valid) + 1000)
    for i, action in enumerate(context.prev_action):
      np.testing.assert_array_equal(action, t + 100 * i)

    # The re-run inputs are the real inputs at indices [t - Ds + 1, t].
    self.assertEqual(len(context.inputs), skip_delay)
    self.assertEqual(len(context.resets), skip_delay)
    for k, state_action in enumerate(context.inputs):
      np.testing.assert_array_equal(state_action.state, t + k + 1)
      np.testing.assert_array_equal(state_action.name, t + k + 1)
      self.assertEqual(state_action.rating, ())

    # The chain taken replaces the actions at [t - Ds + 1, t + 1].
    chain = nash_utils.taken_chain(frames, skip_delay)
    self.assertEqual(len(chain), skip_delay + 1)
    for k, action in enumerate(chain):
      self.assertEqual(len(action), frame_skip)
      for i, controller in enumerate(action):
        np.testing.assert_array_equal(controller, t + k + 1 + 100 * i)

    flat = nash_utils.flatten_chain(chain)
    self.assertEqual(len(flat), (skip_delay + 1) * frame_skip)

  def test_zero_delay(self):
    frames = _index_frames(5, 1, 1)
    context = nash_utils.chain_context(np.arange(4), None, frames, 0)
    self.assertEqual(context.inputs, [])
    np.testing.assert_array_equal(context.prev_action[0], frames.state_action.action[0][:-1])
    chain = nash_utils.taken_chain(frames, 0)
    np.testing.assert_array_equal(chain[0][0], frames.state_action.action[0][1:])


class ChainRerunTest(unittest.TestCase):
  """Re-running a policy over the chain it actually took must reproduce the
  main unroll, so the chain's log-prob is the sum of the per-step ones."""

  def test_taken_chain_log_prob_matches_unroll(self):
    skip_delay = 2
    unroll_length, batch_size = 7, 3
    state = saving.load_state_from_disk(str(paths.JAX_POLICY_CHECKPOINT))
    policy = saving.load_policy_from_state(state)
    frame_skip = policy.frame_skip

    # Dummy encoded inputs with random floats so that steps differ.
    num_states = unroll_length + 1
    state_action = policy.network.dummy((num_states, batch_size))
    key = jax.random.key(0)

    def randomize(x):
      nonlocal key
      if not hasattr(x, 'dtype') or not jnp.issubdtype(x.dtype, jnp.floating):
        return x
      key, subkey = jax.random.split(key)
      return jax.random.normal(subkey, x.shape, x.dtype)

    state_action = jax.tree.map(randomize, state_action)
    controllers = [
        policy.controller_head.dummy_controller([num_states, batch_size])
        for _ in range(frame_skip)]
    state_action = state_action._replace(action=controllers)
    frames = data.Frames(
        state_action=state_action,
        is_resetting=jnp.zeros((num_states, batch_size), dtype=bool),
        reward=jnp.zeros((num_states - 1, batch_size), dtype=jnp.float32),
    )

    initial_state = policy.initial_state(batch_size, nnx.Rngs(0))
    outputs = policy.scan_with_outputs(frames, initial_state)
    context = nash_utils.chain_context(
        outputs.outputs, outputs.hidden_states, frames, skip_delay)
    chain = nash_utils.taken_chain(frames, skip_delay)

    log_prob = nash_utils.chain_log_prob(policy, context, chain)  # [T', B]
    num_valid = unroll_length - skip_delay
    self.assertEqual(log_prob.shape, (num_valid, batch_size))

    # imitation_loss[j] is the (frame-skip mean) distance of the real action
    # at index j + 1; the chain at valid index t covers indices [t - Ds, t].
    per_step = -outputs.imitation_loss  # [T, B]
    expected = sum(
        per_step[k:k + num_valid] for k in range(skip_delay + 1))
    np.testing.assert_allclose(
        np.asarray(log_prob), np.asarray(expected), rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
  unittest.main()
