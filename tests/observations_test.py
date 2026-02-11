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

  def test_null_filter(self):
    filter = observations.build_observation_filter(
        observations.NULL_OBSERVATION_CONFIG)
    game = load_toy_game()
    filtered_game = filter.filter_time(game)
    assert filtered_game == game

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
