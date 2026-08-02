import functools
import json
import unittest

import tree
import numpy as np

from slippi_ai import data, paths, observations, utils
from slippi_ai.types import Game

@functools.cache
def load_toy_game() -> Game:
  with open(paths.TOY_META_PATH) as f:
    meta_rows: list[dict] = json.load(f)

  replay_meta = data.ReplayMeta.from_metadata(meta_rows[0])
  replay_path = paths.TOY_DATA_DIR / replay_meta.slp_md5
  return data.read_table(str(replay_path), compressed=True)

def toy_game_slice(length: int, start: int = 0) -> Game:
  """A writable time-slice of the toy game."""
  game = load_toy_game()
  return utils.map_nt(lambda x: x[start:start + length].copy(), game)

# The toy game's only tech animation is already NEUTRAL_TECH, which is what the
# mask replaces tech actions with; masking it is a no-op. Tests that care about
# masking must supply their own actions.
def random_actions(
    rng: np.random.Generator,
    length: int,
    max_run: int = 2 * observations.DEFAULT_TECH_MASK_WINDOW,
) -> np.ndarray:
  """Random action sequence with many (maskable) tech animations."""
  # Arbitrary non-tech actions.
  non_tech = np.array([14, 20, 44], dtype=observations.ActionDType)
  choices = np.concatenate([non_tech, observations._TECH_ACTIONS_NP])

  actions = np.zeros([length], dtype=observations.ActionDType)
  index = 0
  while index < length:
    run = rng.integers(1, max_run + 1)
    actions[index:index + run] = rng.choice(choices)
    index += run
  return actions

def stack_envs(games: list[Game]) -> Game:
  """Stacks time-major games into a (time, env)-major game."""
  return utils.map_nt(lambda *xs: np.stack(xs, axis=1), *games)

def run_filter_batched(
    filter: observations.ObservationFilter,
    game: Game,
    needs_reset: np.ndarray,
):
  """Filters an (time, env)-major game frame by frame, in place."""
  for t in range(len(game.stage)):
    filter.reset_batched(needs_reset[t])
    # Indexing a leading axis gives a view, so the filter's in-place writes
    # land back in `game`. This is how the rollout workers use it.
    filter.filter_batched(utils.map_nt(lambda x: x[t], game))

# Name needs to start with _ to not be pickup by pytest as a test case.
def _test_filter_time(filter: observations.ObservationFilter):
  """Test that time-batched and sequential filtering gives the same result."""
  game = load_toy_game()
  filter.reset()
  batch_filtered_game = filter.filter_time(game)
  filter.reset()
  filtered_games = [filter.filter(game) for game in utils.unstack_nest(game)]
  assert utils.unstack_nest(batch_filtered_game) == filtered_games

class AnimationFilterTest(unittest.TestCase):

  def test_filter_time(self):
    filter = observations.AnimationFilter()
    _test_filter_time(filter)

  def test_filter_time_masks(self):
    """filter_time and filter agree on actions that actually get masked."""
    rng = np.random.default_rng(0)
    game = toy_game_slice(500)
    game.p1.action[:] = random_actions(rng, 500)

    filter = observations.AnimationFilter()
    filter.reset()
    time_filtered = filter.filter_time(game)
    filter.reset()
    frame_filtered = [filter.filter(g) for g in utils.unstack_nest(game)]

    # Sanity check that the test data exercises masking at all.
    assert not np.array_equal(time_filtered.p1.action, game.p1.action)
    np.testing.assert_array_equal(
        time_filtered.p1.action,
        [g.p1.action for g in frame_filtered])

  def test_mask_window(self):
    """All three filters mask exactly tech_mask_window frames."""
    window = observations.DEFAULT_TECH_MASK_WINDOW
    prefix, run = 3, 2 * window
    actions = np.full([prefix + run + 3], 14, dtype=observations.ActionDType)
    actions[prefix:prefix + run] = observations.F_TECH

    expected = actions.copy()
    expected[prefix:prefix + window] = observations.N_TECH

    game = toy_game_slice(len(actions))
    game.p1.action[:] = actions

    filter = observations.AnimationFilter(tech_mask_window=window)
    filter.reset()
    np.testing.assert_array_equal(
        filter.filter_time(game).p1.action, expected)

    filter.reset()
    np.testing.assert_array_equal(
        [filter.filter(g).p1.action for g in utils.unstack_nest(game)],
        expected)

    batched_game = stack_envs([game])
    needs_reset = np.zeros([len(actions), 1], dtype=bool)
    needs_reset[0] = True
    run_filter_batched(
        observations.AnimationFilter(shape=(1,), tech_mask_window=window),
        batched_game, needs_reset)
    np.testing.assert_array_equal(batched_game.p1.action[:, 0], expected)

  def test_null_filter(self):
    filter = observations.build_observation_filter(
        observations.NULL_OBSERVATION_CONFIG)
    game = load_toy_game()
    filtered_game = filter.filter_time(game)
    assert filtered_game == game

class BatchedAnimationFilterTest(unittest.TestCase):
  """filter_batched is used by the RL actors; the learner sees its output."""

  BATCH_SIZE = 4
  LENGTH = 300

  def make_games(self, rng: np.random.Generator) -> list[Game]:
    games = []
    for _ in range(self.BATCH_SIZE):
      game = toy_game_slice(self.LENGTH, start=int(rng.integers(0, 1000)))
      # Both players get tech animations; only the opponent's may be masked.
      game.p0.action[:] = random_actions(rng, self.LENGTH)
      game.p1.action[:] = random_actions(rng, self.LENGTH)
      games.append(game)
    return games

  def test_matches_unbatched(self):
    """Each env must be filtered as if it were run through `filter` alone."""
    rng = np.random.default_rng(1)
    games = self.make_games(rng)

    needs_reset = rng.random([self.LENGTH, self.BATCH_SIZE]) < 0.02
    needs_reset[0] = True

    # stack_envs copies, so `games` keeps the unfiltered actions.
    batched_game = stack_envs(games)
    run_filter_batched(
        observations.AnimationFilter(shape=(self.BATCH_SIZE,)),
        batched_game, needs_reset)

    for i, game in enumerate(games):
      filter = observations.AnimationFilter()
      expected = []
      for t, frame in enumerate(utils.unstack_nest(game)):
        if needs_reset[t, i]:
          filter.reset()
        expected.append(filter.filter(frame).p1.action)

      np.testing.assert_array_equal(
          batched_game.p1.action[:, i], expected, err_msg=f'env {i}')

  def test_only_masks_opponent_action(self):
    """The filter must leave everything but the opponent's action alone."""
    rng = np.random.default_rng(2)
    batched_game = stack_envs(self.make_games(rng))
    original = utils.map_nt(np.copy, batched_game)

    needs_reset = np.zeros([self.LENGTH, self.BATCH_SIZE], dtype=bool)
    needs_reset[0] = True
    run_filter_batched(
        observations.AnimationFilter(shape=(self.BATCH_SIZE,)),
        batched_game, needs_reset)

    # Sanity check that the test data exercises masking at all.
    assert not np.array_equal(batched_game.p1.action, original.p1.action)

    def check_unmodified(path: tuple[str], before: np.ndarray, after: np.ndarray):
      if path == ('p1', 'action'):
        return
      np.testing.assert_array_equal(before, after, err_msg=str(path))

    tree.map_structure_with_path(check_unmodified, original, batched_game)


class FrameSkipFilterTest(unittest.TestCase):

  def test_filter_time(self):
    filter = observations.FrameSkipFilter(skip=4)
    _test_filter_time(filter)

  def test_control_preservation(self):
    filter = observations.FrameSkipFilter(skip=4)
    game = load_toy_game()
    filtered_game = filter.filter_time(game)

    # Check that the controller is preserved for all frames
    utils.map_nt(
        np.testing.assert_array_equal,
        game.p0.controller, filtered_game.p0.controller)

  def test_frame_skipping(self):
    skip = 4
    filter = observations.FrameSkipFilter(skip=skip)
    game = load_toy_game()
    filtered_game = filter.filter_time(game)

    for index in range(100):
      reference_index = index - index % skip

      def maybe_check_arrays_equal(path: tuple[str], arr1: np.ndarray, arr2: np.ndarray):
        if path[:2] == ('p0', 'controller'):
          return
        assert arr1[index] == arr2[reference_index], f"Mismatch at path {path} for index {index}"

      tree.map_structure_with_path(maybe_check_arrays_equal, filtered_game, game)

if __name__ == '__main__':
  unittest.main()
