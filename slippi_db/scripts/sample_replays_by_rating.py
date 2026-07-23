"""Randomly sample .slp replays per rating bucket into a zip archive.

Training replays are bucketed by game rating (an aggregate of the players'
ratings; games with any unrated player are skipped), N are sampled per bucket,
and the .slp files are extracted from the raw archives into a zip organized as
<lo>-<hi>/<idx>_<original_name>.slp. A manifest.json at the zip root records
each replay's source, players, and ratings.

Usage:
  python slippi_db/scripts/sample_replays_by_rating.py \
      --root=$REPLAYS --output=samples.zip
"""

import collections
import json
import os
import random
import zipfile

from absl import app, flags
import melee
import tqdm

from slippi_ai import nametags
from slippi_db import player_ratings, utils

ROOT = flags.DEFINE_string(
    'root', os.environ.get('REPLAYS'),
    'Replay root directory containing parsed.sqlite and Raw/.')
RATINGS = flags.DEFINE_string(
    'ratings', None,
    'Path to ratings.json. Defaults to <root>/ratings.json.')
OUTPUT = flags.DEFINE_string('output', None, 'Output zip path.', required=True)
BUCKET_SIZE = flags.DEFINE_integer('bucket_size', 250, 'Rating bucket size.')
SAMPLES_PER_BUCKET = flags.DEFINE_integer(
    'samples_per_bucket', 10, 'Number of replays to sample per bucket.')
BUCKET_BY = flags.DEFINE_enum(
    'bucket_by', 'min', ['min', 'mean', 'max'],
    "How to aggregate the players' ratings into a game rating.")
SEED = flags.DEFINE_integer('seed', 0, 'Random seed for sampling.')

AGGREGATORS = {
    'min': min,
    'max': max,
    'mean': lambda ratings: sum(ratings) / len(ratings),
}

def main(_):
  root = ROOT.value
  if not root:
    raise app.UsageError('Set --root or $REPLAYS.')

  ratings = player_ratings.load_ratings(
      RATINGS.value or player_ratings.ratings_path(root))
  aggregate = AGGREGATORS[BUCKET_BY.value]
  bucket_size = BUCKET_SIZE.value

  buckets = collections.defaultdict(list)
  num_games = 0
  num_unrated = 0
  for row in player_ratings.rows_from_sqlite(root):
    num_games += 1
    players = []
    for player in row.get('players', []):
      name = nametags.normalize_name(
          nametags.name_from_metadata(player, row['raw']))
      players.append(dict(
          name=name,
          rating=ratings.get(name),
          character=melee.Character(player['character']).name,
      ))

    player_ratings_ = [p['rating'] for p in players]
    if not player_ratings_ or None in player_ratings_:
      num_unrated += 1
      continue

    rating = aggregate(player_ratings_)
    buckets[int(rating // bucket_size)].append(dict(
        name=row['name'],
        raw=row['raw'],
        md5=row['slp_md5'],
        rating=rating,
        players=players,
    ))

  print(f'{num_games} games, {num_unrated} skipped with unrated players.')

  rng = random.Random(SEED.value)
  sampled = []
  for bucket in sorted(buckets):
    rows = buckets[bucket]
    chosen = rng.sample(rows, min(SAMPLES_PER_BUCKET.value, len(rows)))
    lo, hi = bucket * bucket_size, (bucket + 1) * bucket_size
    print(f'[{lo}, {hi}): sampled {len(chosen)} of {len(rows)}')
    for i, row in enumerate(chosen):
      row['path'] = f'{lo}-{hi}/{i:02d}_{os.path.basename(row["name"])}'
      sampled.append(row)

  # Group by raw archive so each source zip is only opened once.
  by_raw = collections.defaultdict(list)
  for row in sampled:
    by_raw[row['raw']].append(row)

  with zipfile.ZipFile(OUTPUT.value, 'w', zipfile.ZIP_DEFLATED) as out:
    for raw, rows in tqdm.tqdm(by_raw.items(), unit='archive'):
      src_path = os.path.join(root, 'Raw', raw)
      with zipfile.ZipFile(src_path) as src:
        members = set(src.namelist())
      for row in rows:
        # The stored file may be compressed as .slp.gz or .slpz.
        base = row['name'].removesuffix('.slp')
        candidates = [base + suffix for suffix in utils.VALID_SUFFIXES]
        member = next(c for c in candidates if c in members)
        slp = utils.SlpZipFile(src_path, member).read()
        out.writestr(row['path'], slp)
    out.writestr('manifest.json', json.dumps(sampled, indent=2))

  print(f'Wrote {len(sampled)} replays to {OUTPUT.value}.')

if __name__ == '__main__':
  app.run(main)
