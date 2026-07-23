"""Compute the player rating distribution of a dataset.

Player identities come from parsed.sqlite under the replay root, and ratings
are looked up by normalized player name in ratings.json.

Usage:
  python slippi_db/scripts/rating_distribution.py --root=$REPLAYS
"""

import collections
import os

from absl import app, flags

from slippi_ai import nametags
from slippi_db import player_ratings

ROOT = flags.DEFINE_string(
    'root', os.environ.get('REPLAYS'),
    'Replay root directory containing parsed.sqlite.')
RATINGS = flags.DEFINE_string(
    'ratings', None,
    'Path to ratings.json. Defaults to <root>/ratings.json.')
BUCKET_SIZE = flags.DEFINE_integer('bucket_size', 250, 'Rating bucket size.')

def main(_):
  root = ROOT.value
  if not root:
    raise app.UsageError('Set --root or $REPLAYS.')

  ratings = player_ratings.load_ratings(
      RATINGS.value or player_ratings.ratings_path(root))

  rows = player_ratings.rows_from_sqlite(root)

  bucket_size = BUCKET_SIZE.value
  bucket_counts = collections.Counter()  # bucket -> player-game count
  bucket_players = collections.defaultdict(set)  # bucket -> player names
  unrated_count = 0
  unrated_players = set()

  for row in rows:
    for player in row.get('players', []):
      name = nametags.name_from_metadata(player, row['raw'])
      name = nametags.normalize_name(name)
      rating = ratings.get(name)
      if rating is None:
        unrated_count += 1
        unrated_players.add(name)
      else:
        bucket = int(rating // bucket_size)
        bucket_counts[bucket] += 1
        bucket_players[bucket].add(name)

  total = sum(bucket_counts.values())
  if total == 0:
    print('No rated players found.')
    return

  print(f'Rating distribution (bucket size {bucket_size}):')
  for bucket in range(min(bucket_counts), max(bucket_counts) + 1):
    count = bucket_counts[bucket]
    num_players = len(bucket_players[bucket])
    lo, hi = bucket * bucket_size, (bucket + 1) * bucket_size
    print(f'[{lo:5d}, {hi:5d}): {count:9d} player-games ({100 * count / total:6.2f}%)'
          f'  {num_players:6d} players')

  num_players = len(set.union(*bucket_players.values()))
  print(f'Rated:   {total} player-games, {num_players} players')
  print(f'Unrated: {unrated_count} player-games, {len(unrated_players)} players')

if __name__ == '__main__':
  app.run(main)
