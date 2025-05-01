import json
import unittest
from parameterized import parameterized

from slippi_ai import data, paths, observations, utils
from slippi_ai.types import Game

def load_toy_game() -> Game:
  with open(paths.TOY_META_PATH) as f:
    meta_rows: list[dict] = json.load(f)

  replay_meta = data.ReplayMeta.from_metadata(meta_rows[0])
  replay_path = paths.TOY_DATA_DIR / replay_meta.slp_md5
  return data.read_table(replay_path, compressed=True)

def test_filter_time(filter: observations.ObservationFilter):
  """Test that time-batched and sequential filtering gives the same result."""
  game = load_toy_game()
  filter.reset()
  batch_filtered_game = filter.filter_time(game)
  filter.reset()
  filtered_games = [filter.filter(game) for game in utils.unstack_nest(game)]
  assert utils.unstack_nest(batch_filtered_game) == filtered_games

class AnimationFilterTest(unittest.TestCase):

  @parameterized.expand([(0,), (7,)])
  def test_filter_time(self, num_frames: int):
    filter = observations.TechAnimationFilter(num_frames)
    test_filter_time(filter)

if __name__ == '__main__':
  unittest.main()
