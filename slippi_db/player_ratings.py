"""Utilities for working with player ratings.

Ratings live in ratings.json, mapping normalized player names (see
slippi_ai.nametags) to slippi ranked rating ordinals. The file is produced by
scripts/fetch_player_ratings.py from a TinyDB cache of slippi API responses.
"""

import json
import os
import sqlite3
from typing import Iterator, Optional

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
  """Yields minimal replay rows: name, raw, slp_md5, and player identities.

  Selects only the columns needed to identify players; materializing full
  rows from the ~40-column table is an order of magnitude slower.
  """
  columns = ['name', 'raw', 'slp_md5']
  for prefix in ('p0', 'p1'):
    columns.extend(f'{prefix}_{c}' for c in (
        'port', 'name_tag', 'netplay_name', 'netplay_code', 'character'))

  query = (f'SELECT {", ".join(columns)} FROM replays'
           ' WHERE valid = 1 AND is_training = 1')
  if require_data_ok:
    query += ' AND data_ok = 1'

  db_path = os.path.join(root, 'parsed.sqlite')
  conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
  for name, raw, slp_md5, *player_fields in conn.execute(query):
    players = []
    for i in range(2):
      port, name_tag, netplay_name, netplay_code, character = (
          player_fields[5 * i:5 * (i + 1)])
      if port is None:
        continue
      netplay = None
      if netplay_name is not None or netplay_code is not None:
        netplay = dict(name=netplay_name, code=netplay_code)
      players.append(dict(
          name_tag=name_tag, netplay=netplay, character=character))
    yield dict(name=name, raw=raw, slp_md5=slp_md5, players=players)
  conn.close()

def is_connect_code(name: str) -> bool:
  return '#' in name

# Slippi hides ranks until five placement sets have been played; ratings with
# fewer updates than this are unreliable, and accounts that have never played
# ranked report the default rating ordinal of 1100.
MIN_RATING_UPDATES = 5

# Thresholds for treating a (code, suid) majority pairing as trustworthy:
# per-replay suid metadata is occasionally corrupted (swapped between player
# slots), and many dumps record an empty suid.
MIN_ALIAS_GAMES = 10
MIN_ALIAS_FRACTION = 0.9

def code_alias_groups(
    root: str,
    min_games: int = MIN_ALIAS_GAMES,
    min_fraction: float = MIN_ALIAS_FRACTION,
) -> list[list[str]]:
  """Groups of connect codes that belong to the same slippi account.

  Renaming a slippi account changes its connect code but not its suid, so
  codes sharing a suid share the account's ranked rating. Each code is
  assigned to the suid it appears with most often, if that pairing occurs in
  at least min_games replays and a min_fraction majority of the code's
  replays.
  """
  db_path = os.path.join(root, 'parsed.sqlite')
  conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
  pair_counts: dict[str, dict[str, int]] = {}  # code -> suid -> games
  for prefix in ('p0', 'p1'):
    query = (
        f"SELECT REPLACE({prefix}_netplay_code, '＃', '#'),"
        f" {prefix}_netplay_suid, COUNT(*) FROM replays"
        f" WHERE valid = 1 AND is_training = 1"
        f" AND {prefix}_netplay_code != ''"
        f" AND {prefix}_netplay_suid IS NOT NULL"
        f" AND {prefix}_netplay_suid != ''"
        f" GROUP BY 1, 2")
    for code, suid, count in conn.execute(query):
      suids = pair_counts.setdefault(code, {})
      suids[suid] = suids.get(suid, 0) + count
  conn.close()

  suid_to_codes: dict[str, list[str]] = {}
  for code, suids in pair_counts.items():
    suid, top = max(suids.items(), key=lambda kv: kv[1])
    if top >= min_games and top / sum(suids.values()) >= min_fraction:
      suid_to_codes.setdefault(suid, []).append(code)

  return [codes for codes in suid_to_codes.values() if len(codes) > 1]

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
