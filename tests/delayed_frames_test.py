"""Tests for data.delayed_frames alignment."""

import unittest

import numpy as np

from slippi_ai import data


def _make_frames(num_states: int, batch_size: int, frame_skip: int, with_rating: bool):
  """Frames whose every leaf holds its own time index."""
  time = np.arange(num_states, dtype=np.int32)
  time = np.broadcast_to(time[:, np.newaxis], (num_states, batch_size))

  state_action = data.StateAction(
      state=time,  # any nest works; a bare array keeps the test simple
      action=[time + 100 * i for i in range(frame_skip)],
      name=time,
      rating=time.astype(np.float32) if with_rating else (),
  )
  return data.Frames(
      state_action=state_action,
      is_resetting=np.zeros((num_states, batch_size), dtype=bool),
      reward=time[:-1].astype(np.float32),
  )


class DelayedFramesTest(unittest.TestCase):

  def test_zero_delay_is_identity(self):
    frames = _make_frames(8, 2, frame_skip=1, with_rating=True)
    self.assertIs(data.delayed_frames(frames, 0), frames)

  def test_alignment(self):
    unroll_length, skip_delay, batch_size, frame_skip = 5, 2, 3, 2
    frames = _make_frames(
        unroll_length + skip_delay + 1, batch_size, frame_skip, with_rating=True)
    delayed = data.delayed_frames(frames, skip_delay)

    num_states = unroll_length + 1
    self.assertEqual(delayed.is_resetting.shape, (num_states, batch_size))
    self.assertEqual(delayed.reward.shape, (unroll_length, batch_size))

    t = np.broadcast_to(np.arange(num_states)[:, np.newaxis], (num_states, batch_size))
    # States, names and ratings keep their original index.
    np.testing.assert_array_equal(delayed.state_action.state, t)
    np.testing.assert_array_equal(delayed.state_action.name, t)
    np.testing.assert_array_equal(delayed.state_action.rating, t)
    # Actions are shifted forward by the delay, for every frame-skip slot.
    self.assertEqual(len(delayed.state_action.action), frame_skip)
    for i, action in enumerate(delayed.state_action.action):
      np.testing.assert_array_equal(action, t + skip_delay + 100 * i)
    # Rewards follow the delayed actions.
    np.testing.assert_array_equal(delayed.reward, t[:-1] + skip_delay)

  def test_missing_rating(self):
    frames = _make_frames(6, 1, frame_skip=1, with_rating=False)
    delayed = data.delayed_frames(frames, 2)
    self.assertEqual(delayed.state_action.rating, ())
    self.assertEqual(delayed.reward.shape, (3, 1))


if __name__ == '__main__':
  unittest.main()
