"""Fetch player ratings from the slippi API and write ratings.json.

Collects connect codes from the dataset, fetches each player's ranked history
(all seasons, with rating update counts) into a TinyDB cache, and writes
ratings.json mapping normalized player names to their max rating ordinal.

Accounts whose seasons all have fewer than --min_rating_updates rating updates
are considered unrated and excluded: the slippi API reports the default rating
ordinal (1100) even for accounts that have never played a ranked game.

Legacy cache entries (from the player_ratings notebook) only stored a
max_rating with no update counts; they are refetched by default.

Connect codes that demonstrably share a slippi account (via the suid metadata
in parsed.sqlite; see player_ratings.code_alias_groups) share the account's
rating. In particular a renamed account's old code, which no longer resolves
on the slippi API, inherits the rating of its current code.

Usage:
  python slippi_db/scripts/fetch_player_ratings.py --root=$REPLAYS
"""

import collections
import datetime
import json
import logging
import os

from absl import app, flags
import tinydb
import tqdm

from slippi.slippi_api import SlippiRankedAPI, logger as slippi_logger
from slippi_ai import nametags
from slippi_db import player_ratings

ROOT = flags.DEFINE_string(
    'root', os.environ.get('REPLAYS'),
    'Replay root directory containing parsed.sqlite.')
DB = flags.DEFINE_string(
    'db', None,
    'TinyDB cache of slippi API responses. Defaults to <root>/ranks.json.')
OUTPUT = flags.DEFINE_string(
    'output', None,
    'Path to write ratings.json. Defaults to <root>/ratings.json.')
MIN_RATING_UPDATES = flags.DEFINE_integer(
    'min_rating_updates', player_ratings.MIN_RATING_UPDATES,
    'Ignore seasons with fewer rating updates than this.')
FETCH_LIMIT = flags.DEFINE_integer(
    'fetch_limit', 0, 'Max number of API fetches this run (0 = unlimited).')
FETCH = flags.DEFINE_boolean(
    'fetch', True, 'With --nofetch, only rewrite ratings.json from the cache.')
DRY_RUN = flags.DEFINE_boolean(
    'dry_run', False,
    'Report what would be fetched and exit without fetching or writing.')
OVERWRITE = flags.DEFINE_boolean(
    'overwrite', False, 'Refetch codes already present in the cache.')

MAX_CONSECUTIVE_FAILURES = 5

SEASON_KEYS = ['ratingOrdinal', 'ratingUpdateCount', 'wins', 'losses']

def fetch_seasons(api: SlippiRankedAPI, code: str) -> list[dict] | None:
  """Fetch all ranked seasons for a code; None if the user doesn't exist."""
  response = api.get_player_data_throttled(code)
  user = response['data']['getUser']
  if user is None:
    return None

  seasons = list(user['rankedNetplayProfileHistory'] or [])
  current = user['rankedNetplayProfile']
  if current is not None:
    seasons.append(current)

  return [{key: season.get(key) for key in SEASON_KEYS}
          for season in seasons if season is not None]

def collect_code_counts(root: str) -> collections.Counter:
  """Count per-game player occurrences of each connect code in the dataset."""
  rows = player_ratings.rows_from_sqlite(root, require_data_ok=False)
  counts = collections.Counter()
  for row in rows:
    for player in row.get('players', []):
      name = nametags.name_from_metadata(player, row['raw'])
      if player_ratings.is_connect_code(name):
        counts[name] += 1
  return counts

def write_ratings(
    entries: list[dict],
    output: str,
    min_updates: int,
    alias_groups: list[list[str]],
):
  """Write normalized name -> max rating, preserving manual entries."""
  existing = {}
  if os.path.isfile(output):
    existing = player_ratings.load_ratings(output, with_fixed=False)

  ratings: dict[str, float] = {}
  derived_names = set()
  for entry in entries:
    name = nametags.normalize_name(entry['code'])
    derived_names.add(name)
    rating = player_ratings.rating_from_entry(entry, min_updates)
    if rating is not None:
      ratings[name] = max(rating, ratings.get(name, rating))

  # Codes that share a slippi account share its rating, e.g. when the account
  # was renamed and the old code no longer resolves on the slippi API.
  aliased = 0
  for group in alias_groups:
    names = {nametags.normalize_name(code) for code in group}
    group_rating = max(
        (ratings[name] for name in names if name in ratings), default=None)
    if group_rating is None:
      continue
    for name in names:
      if name not in ratings:
        aliased += 1
      derived_names.add(name)
      ratings[name] = group_rating
  print(f'Assigned ratings to {aliased} unrated codes via suid aliases.')

  # Keep entries that didn't come from a cached code (e.g. manual additions),
  # but drop names whose codes are now known to be unrated.
  preserved = {
      name: rating for name, rating in existing.items()
      if name not in derived_names and name not in ratings
  }
  ratings.update(preserved)

  with open(output, 'w') as f:
    json.dump(ratings, f)
  print(f'Wrote {len(ratings)} ratings to {output} '
        f'({len(preserved)} preserved from existing file).')

def main(_):
  root = ROOT.value
  if not root:
    raise app.UsageError('Set --root or $REPLAYS.')

  output = OUTPUT.value or player_ratings.ratings_path(root)

  print('Collecting connect codes from the dataset...')
  code_counts = collect_code_counts(root)
  alias_groups = player_ratings.code_alias_groups(root)
  print(f'Found {len(code_counts)} distinct connect codes,'
        f' {len(alias_groups)} multi-code accounts.')

  db = tinydb.TinyDB(DB.value or os.path.join(root, 'ranks.json'))
  code_query = tinydb.Query().code
  entries = {}  # last write wins, deduplicating legacy entries
  for entry in db.all():
    entries[entry['code']] = entry

  def needs_fetch(code: str) -> bool:
    if OVERWRITE.value:
      return True
    entry = entries.get(code)
    # Legacy entries have no per-season data and can't be filtered by
    # rating update count.
    return entry is None or 'seasons' not in entry

  pending = [code for code in code_counts if needs_fetch(code)]
  # Fetch the most common players first so that partial runs (e.g. with
  # --fetch_limit) fix the largest contributors to the distribution.
  pending.sort(key=lambda code: code_counts[code], reverse=True)
  total_pending = len(pending)
  if FETCH_LIMIT.value:
    pending = pending[:FETCH_LIMIT.value]

  if DRY_RUN.value:
    missing = sum(1 for code in pending if code not in entries)
    eta = datetime.timedelta(seconds=len(pending))  # API limit is 1 req/sec.
    print(f'Would fetch {len(pending)} codes'
          f' ({missing} missing, {len(pending) - missing} legacy'
          f'{f", limited from {total_pending} pending" if len(pending) < total_pending else ""}),'
          f' taking ~{eta} at 1 request/sec.')
    db.close()
    return

  if not FETCH.value:
    pending = []
  print(f'Fetching {len(pending)} codes.')

  slippi_logger.setLevel(logging.WARNING)
  api = SlippiRankedAPI()
  failures = 0

  try:
    for code in tqdm.tqdm(pending, unit='code'):
      try:
        seasons = fetch_seasons(api, code)
      except Exception as e:
        failures += 1
        tqdm.tqdm.write(f'Failed to fetch {code}: {e}')
        if failures >= MAX_CONSECUTIVE_FAILURES:
          print(f'{failures} consecutive failures, is the API down? Aborting.')
          break
        continue
      failures = 0

      entry = dict(
          code=code,
          date=datetime.date.today().isoformat(),
          seasons=seasons,
      )
      db.upsert(entry, code_query == code)
      entries[code] = entry
  except KeyboardInterrupt:
    print('Interrupted; writing ratings from cache so far.')
  finally:
    db.close()
    write_ratings(
        list(entries.values()), output, MIN_RATING_UPDATES.value, alias_groups)

  legacy = sum(1 for e in entries.values() if 'seasons' not in e)
  if legacy:
    print(f'{legacy} legacy cache entries remain without per-season data; '
          'rerun to refetch them.')

if __name__ == '__main__':
  app.run(main)
