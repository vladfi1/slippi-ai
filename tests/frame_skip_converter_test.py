"""Tests for FrameSkipConverter, get_delayed_frames and get_frames with delay."""

import unittest
from unittest import mock

import numpy as np

from slippi_ai import evaluators
from slippi_ai.controller_heads import SampleOutputs
from slippi_ai.jax.rl import learner as learner_lib

DUMMY = -1


def _index_outputs(frames: np.ndarray) -> SampleOutputs:
  return SampleOutputs(controller_state=frames, logits=frames + 1000)


def _make_trajectory(first_frame: int, num_frames: int, delay: int, batch_size: int):
  """A per-frame Trajectory whose every leaf holds its own frame index."""
  frames = np.arange(first_frame, first_frame + num_frames + 1, dtype=np.int32)
  frames = np.broadcast_to(frames[:, np.newaxis], (num_frames + 1, batch_size))
  last = first_frame + num_frames
  return evaluators.Trajectory(
      states=frames,  # any nest works; a bare array keeps the test simple
      name=frames,
      rating=frames.astype(np.float32),
      actions=_index_outputs(frames),
      rewards=frames[1:].astype(np.float32),  # unused; recomputed by converter
      is_resetting=np.zeros((num_frames + 1, batch_size), dtype=bool),
      initial_state=(),
      # The actor's queued actions for frames [U+1, U+D].
      delayed_actions=[
          _index_outputs(np.full([batch_size], last + 1 + d, dtype=np.int32))
          for d in range(delay)
      ],
  )


def _make_converter(frame_skip, skip_delay, batch_size, **kwargs):
  dummy = _index_outputs(np.full([batch_size], DUMMY, dtype=np.int32))
  return learner_lib.FrameSkipConverter(
      frame_skip=frame_skip,
      batch_shape=(batch_size,),
      dummy_sample_outputs=dummy,
      reward_config=learner_lib.reward_lib.RewardConfig.default(),
      skip_delay=skip_delay,
      **kwargs,
  )


# Reward for the transition into frame f is f.
def _fake_rewards(states, **kwargs):
  return states[1:].astype(np.float32)


class FrameSkipConverterTest(unittest.TestCase):

  def _convert_rollouts(
      self, frame_skip, skip_delay, num_frames, num_rollouts, batch_size,
      **kwargs):
    converter = _make_converter(frame_skip, skip_delay, batch_size, **kwargs)
    with mock.patch.object(
        learner_lib.reward_lib, 'compute_rewards', _fake_rewards):
      return [
          converter.convert(_make_trajectory(
              k * num_frames, num_frames, skip_delay * frame_skip, batch_size))
          for k in range(num_rollouts)
      ]

  def _check_alignment(self, frame_skip, skip_delay, num_frames, batch_size):
    steps = num_frames // frame_skip  # T
    fs_trajectories = self._convert_rollouts(
        frame_skip, skip_delay, num_frames, num_rollouts=3, batch_size=batch_size)

    for k, fs_trajectory in enumerate(fs_trajectories):
      self.assertEqual(fs_trajectory.is_resetting.shape, (steps + 1, batch_size))
      self.assertEqual(fs_trajectory.rewards.shape, (steps, batch_size))
      self.assertEqual(len(fs_trajectory.actions), frame_skip)

      fs_steps = np.arange(k * steps, (k + 1) * steps + 1)
      np.testing.assert_array_equal(fs_trajectory.states[:, 0], fs_steps * frame_skip)

      # Slot i of step g holds the per-frame action g * FS - FS + 1 + i, for
      # skip_delay steps beyond the last state (the actor's queued actions).
      action_steps = np.arange(k * steps, (k + 1) * steps + skip_delay + 1)
      for i, action in enumerate(fs_trajectory.actions):
        action_frames = action_steps * frame_skip - frame_skip + 1 + i
        # Before frame 0 the prev-actions are dummies.
        action_frames = np.where(action_frames < 0, DUMMY, action_frames)
        np.testing.assert_array_equal(action.controller_state[:, 0], action_frames)
        # Dummy outputs carry DUMMY + 1000 as their logits.
        np.testing.assert_array_equal(action.logits[:, 0], action_frames + 1000)

      # Reward for step g sums the transitions into frames (g * FS, (g + 1) * FS].
      expected_rewards = np.stack([
          np.arange(g * frame_skip + 1, (g + 1) * frame_skip + 1).sum()
          for g in fs_steps[:-1]
      ]).astype(np.float32)
      np.testing.assert_array_equal(fs_trajectory.rewards[:, 0], expected_rewards)

      frames = learner_lib.get_delayed_frames(fs_trajectory, skip_delay)
      # States are untouched; actions are those of skip_delay steps later.
      np.testing.assert_array_equal(frames.state_action.state, fs_trajectory.states)
      np.testing.assert_array_equal(frames.is_resetting, fs_trajectory.is_resetting)
      for i, action in enumerate(frames.state_action.action):
        expected = fs_trajectory.actions[i].controller_state[skip_delay:]
        np.testing.assert_array_equal(action.controller_state, expected)
        np.testing.assert_array_equal(action.logits, fs_trajectory.actions[i].logits[skip_delay:])
      # Only the rewards following the paired actions remain.
      np.testing.assert_array_equal(frames.reward, fs_trajectory.rewards[skip_delay:])
      self.assertEqual(frames.reward.shape, (steps - skip_delay, batch_size))

      # Unless the prefix rewards are kept.
      frames = learner_lib.get_delayed_frames(
          fs_trajectory, skip_delay, keep_prefix_rewards=True)
      np.testing.assert_array_equal(frames.reward, fs_trajectory.rewards)
      for i, action in enumerate(frames.state_action.action):
        expected = fs_trajectory.actions[i].controller_state[skip_delay:]
        np.testing.assert_array_equal(action.controller_state, expected)

      # get_frames keeps the actions actually taken, dropping the queued ones.
      vf_frames = learner_lib.get_frames(fs_trajectory)
      np.testing.assert_array_equal(vf_frames.state_action.state, fs_trajectory.states)
      for i, action in enumerate(vf_frames.state_action.action):
        expected = fs_trajectory.actions[i].controller_state[:steps + 1]
        np.testing.assert_array_equal(action, expected)
      np.testing.assert_array_equal(vf_frames.reward, fs_trajectory.rewards)

  def test_no_delay(self):
    self._check_alignment(frame_skip=2, skip_delay=0, num_frames=6, batch_size=1)

  def test_delay(self):
    self._check_alignment(frame_skip=2, skip_delay=2, num_frames=6, batch_size=2)

  def test_delay_with_frame_skip_3(self):
    self._check_alignment(frame_skip=3, skip_delay=1, num_frames=9, batch_size=1)

  def test_discount(self):
    frame_skip, discount = 3, 0.5
    (fs_trajectory,) = self._convert_rollouts(
        frame_skip, 0, num_frames=6, num_rollouts=1, batch_size=1,
        discount=discount)
    # Reward for step g discounts the transitions into frames
    # (g * FS, (g + 1) * FS] by their offset within the step.
    discounts = discount ** np.arange(frame_skip)
    expected = np.stack([
        (np.arange(g * frame_skip + 1, (g + 1) * frame_skip + 1) * discounts).sum()
        for g in range(2)
    ]).astype(np.float32)
    np.testing.assert_allclose(fs_trajectory.rewards[:, 0], expected)

  def test_overlap(self):
    frame_skip, skip_delay, num_frames, batch_size = 2, 1, 8, 2
    overlap = 2
    steps = num_frames // frame_skip  # U
    plain = self._convert_rollouts(
        frame_skip, skip_delay, num_frames, num_rollouts=3, batch_size=batch_size)
    overlapped = self._convert_rollouts(
        frame_skip, skip_delay, num_frames, num_rollouts=3, batch_size=batch_size,
        overlap_steps=overlap)

    # The first trajectory has nothing to prepend.
    np.testing.assert_array_equal(overlapped[0].states, plain[0].states)
    np.testing.assert_array_equal(overlapped[0].rewards, plain[0].rewards)

    for prev, plain_fs, fs in zip(plain[:-1], plain[1:], overlapped[1:]):
      # Steps [U - O, U) of the previous rollout, then the new rollout.
      self.assertEqual(fs.is_resetting.shape, (steps + overlap + 1, batch_size))
      self.assertEqual(fs.rewards.shape, (steps + overlap, batch_size))
      for old, new, got in [
          (prev.states, plain_fs.states, fs.states),
          (prev.name, plain_fs.name, fs.name),
          (prev.rating, plain_fs.rating, fs.rating),
          (prev.is_resetting, plain_fs.is_resetting, fs.is_resetting),
          (prev.rewards, plain_fs.rewards, fs.rewards),
      ]:
        expected = np.concatenate([old[steps - overlap:steps], new], axis=0)
        np.testing.assert_array_equal(got, expected)
      for old, new, got in zip(prev.actions, plain_fs.actions, fs.actions):
        expected = np.concatenate(
            [old.controller_state[steps - overlap:steps], new.controller_state],
            axis=0)
        np.testing.assert_array_equal(got.controller_state, expected)
        # The action steps still run skip_delay past the last state.
        self.assertEqual(
            got.controller_state.shape,
            (steps + overlap + skip_delay + 1, batch_size))
      # The actor's state belongs to the start of the new rollout.
      self.assertEqual(fs.initial_state, plain_fs.initial_state)

      # The overlapped trajectory is one continuous sequence of steps.
      frames = learner_lib.get_delayed_frames(fs, skip_delay)
      states = frames.state_action.state[:, 0]
      np.testing.assert_array_equal(np.diff(states), frame_skip)

  def test_batch_shape(self):
    frame_skip, num_frames, b1, b2 = 2, 4, 2, 3
    dummy = _index_outputs(np.full([b1, b2], DUMMY, dtype=np.int32))
    converter = learner_lib.FrameSkipConverter(
        frame_skip=frame_skip,
        batch_shape=(b1, b2),
        dummy_sample_outputs=dummy,
        reward_config=learner_lib.reward_lib.RewardConfig.default(),
    )
    trajectory = _make_trajectory(0, num_frames, 0, b1)
    trajectory = trajectory._replace(**{
        k: np.broadcast_to(v[..., np.newaxis], v.shape + (b2,))
        for k, v in trajectory._asdict().items()
        if k not in ['initial_state', 'delayed_actions', 'actions']
    }, actions=_index_outputs(
        np.broadcast_to(
            trajectory.actions.controller_state[..., np.newaxis],
            (num_frames + 1, b1, b2))))
    with mock.patch.object(
        learner_lib.reward_lib, 'compute_rewards', _fake_rewards):
      fs_trajectory = converter.convert(trajectory)
    steps = num_frames // frame_skip
    self.assertEqual(fs_trajectory.is_resetting.shape, (steps + 1, b1, b2))
    self.assertEqual(fs_trajectory.rewards.shape, (steps, b1, b2))
    self.assertEqual(fs_trajectory.states.shape, (steps + 1, b1, b2))
    for action in fs_trajectory.actions:
      self.assertEqual(action.controller_state.shape, (steps + 1, b1, b2))

  def test_rejects_mismatched_delayed_actions(self):
    converter = _make_converter(frame_skip=2, skip_delay=1, batch_size=1)
    trajectory = _make_trajectory(0, 4, delay=0, batch_size=1)
    with self.assertRaises(ValueError):
      converter.convert(trajectory)

  def test_delayed_frames_rejects_wrong_skip_delay(self):
    (fs_trajectory,) = self._convert_rollouts(
        2, 1, num_frames=4, num_rollouts=1, batch_size=1)
    with self.assertRaises(ValueError):
      learner_lib.get_delayed_frames(fs_trajectory, 0)


if __name__ == '__main__':
  unittest.main()
