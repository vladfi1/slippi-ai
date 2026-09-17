"""Tests for the action-chain helpers in slippi_ai.jax.nash.utils."""

import pickle
import unittest

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from slippi_ai import data, flag_utils, paths
from slippi_ai.jax import jax_utils, saving
from slippi_ai.jax.nash import train_q_fn
from slippi_ai.jax.nash import q_function as q_lib
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

  def test_chunk_layout(self):
    delay, frame_skip, unroll_length, batch_size = 4, 2, 5, 3
    layout = nash_utils.ChunkLayout(delay, frame_skip)
    self.assertEqual(layout.skip_delay, 2)
    self.assertEqual(layout.extra_frames, frame_skip + 2 * delay)
    with self.assertRaises(ValueError):
      nash_utils.ChunkLayout(3, 2)

    # A chunk of U + 2 Ds + 1 states, as the data source provides.
    chunk = _index_frames(
        unroll_length + 2 * layout.skip_delay + 1, batch_size, frame_skip)
    frames = layout.delayed_frames(chunk)
    self.assertEqual(layout.num_valid(frames), unroll_length)
    # States [0, U + Ds], rewards [0, U + Ds - 1]; the game covers [Ds, U + Ds).
    num_steps = unroll_length + layout.skip_delay
    self.assertEqual(frames.reward.shape[0], num_steps)
    np.testing.assert_array_equal(
        layout.game_slice(frames.reward)[:, 0],
        np.arange(layout.skip_delay, num_steps))
    np.testing.assert_array_equal(
        layout.game_slice(frames.reward[np.newaxis], axis=1)[0, :, 0],
        np.arange(layout.skip_delay, num_steps))
    # The next chunk starts at index U, so the state after index U - 1 is
    # carried.
    hidden = {'h': np.arange(num_steps)}
    self.assertEqual(layout.carried_state(hidden, frames)['h'], unroll_length - 1)

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

  def test_q_function_taken_chain_matches_scan(self):
    """The q-function's re-run over the chain actually taken reproduces the
    action_net initial states of its main scan (at every skip-delay)."""
    unroll_length, batch_size = 7, 3
    with open(paths.JAX_NASH_Q_FN_CKPT, 'rb') as f:
      q_fn_state = pickle.load(f)
    q_fn_config = flag_utils.dataclass_from_dict(
        train_q_fn.Config, q_fn_state['config'])
    q_function = q_lib.build_q_function(nnx.Rngs(0), q_fn_config.q_function)
    jax_utils.set_module_state(
        q_function, jax.tree.map(jnp.asarray, q_fn_state['state']['q_function']))
    frame_skip = q_function.frame_skip

    num_states = unroll_length + 1
    shape = (num_states, batch_size, 2)
    state_action = q_function.core_net.dummy(shape)
    key = jax.random.key(0)

    def randomize(x):
      nonlocal key
      if not hasattr(x, 'dtype') or not jnp.issubdtype(x.dtype, jnp.floating):
        return x
      key, subkey = jax.random.split(key)
      return jax.random.normal(subkey, x.shape, x.dtype)

    state_action = jax.tree.map(randomize, state_action)
    state_action = state_action._replace(
        action=[q_function.embed_action.dummy(shape) for _ in range(frame_skip)])
    frames = data.Frames(
        state_action=state_action,
        is_resetting=jnp.zeros(shape, dtype=bool),
        reward=jnp.zeros((unroll_length, batch_size, 2), dtype=jnp.float32),
    )

    initial_state = q_function.initial_state(batch_size, nnx.Rngs(0))
    core = q_function.scan_core(frames, initial_state, rngs=nnx.Rngs(1))

    for skip_delay in [0, 2]:
      context = nash_utils.chain_context(
          core.core_outputs, core.core_states, frames, skip_delay)
      chain = nash_utils.taken_chain(frames, skip_delay)
      action_init = q_function.chain_action_init_state(context, chain)
      expected = jax.tree.map(lambda x: x[skip_delay:], core.action_init_state)
      jax.tree.map(
          lambda a, b: np.testing.assert_allclose(
              np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-4),
          action_init, expected)


if __name__ == '__main__':
  unittest.main()
