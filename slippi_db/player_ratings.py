"""Utilities for working with player ratings.

Ratings live in ratings.json, mapping normalized player names (see
slippi_ai.nametags) to slippi ranked rating ordinals. The file is produced by
scripts/fetch_player_ratings.py from a TinyDB cache of slippi API responses.
"""

import json
import os
import sqlite3
from typing import Iterator, Optional

from slippi_db import parse_local

# Ratings for the anonymized names in Fizzi's ranked dumps.
FIXED_RATINGS = {
    'Platinum Player': 1750,
    'Diamond Player': 2000,
    'Master Player': 2150,
}

def ratings_path(root: str) -> str:
  return os.path.join(root, 'ratings.json')

def load_ratings(path: str, with_fixed: bool = True) -> dict[str, float]:
  with open(path) as f:
    ratings = json.load(f)
  if with_fixed:
    ratings.update(FIXED_RATINGS)
  return ratings

def rows_from_sqlite(
    root: str,
    require_data_ok: bool = True,
) -> Iterator[dict]:
  db_path = os.path.join(root, 'parsed.sqlite')
  conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
  conn.row_factory = sqlite3.Row
  query = 'SELECT * FROM replays WHERE valid = 1 AND is_training = 1'
  if require_data_ok:
    query += ' AND data_ok = 1'
  for row in conn.execute(query):
    yield parse_local.sqlite_row_to_dict(dict(row))
  conn.close()

def is_connect_code(name: str) -> bool:
  return '#' in name

# Slippi hides ranks until five placement sets have been played; ratings with
# fewer updates than this are unreliable, and accounts that have never played
# ranked report the default rating ordinal of 1100.
MIN_RATING_UPDATES = 5

def rating_from_entry(entry: dict, min_updates: int = MIN_RATING_UPDATES) -> Optional[float]:
  """Max rating over a cache entry's seasons, or None if effectively unrated.

  Legacy entries predating per-season data only have 'max_rating', which
  can't be filtered by update count; it is returned as-is.
  """
  if 'seasons' not in entry:
    return entry.get('max_rating')
  seasons = entry['seasons'] or []
  ratings = [
      s['ratingOrdinal'] for s in seasons
      if (s.get('ratingUpdateCount') or 0) >= min_updates
  ]
  return max(ratings, default=None)
