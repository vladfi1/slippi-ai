"""The final step of dataset creation.

python slippi_db/scripts/make_local_dataset.py --root=Root
"""

import collections
import os
import pickle
import json
import tarfile
import tqdm
from typing import Optional

from absl import app, flags

from slippi_ai import nametags

ROOT = flags.DEFINE_string('root', None, 'root directory', required=True)
WINNER_ONLY = flags.DEFINE_boolean(
  'winner_only', True, 'only keep games that have a winner')

MAKE_TAR = flags.DEFINE_boolean('tar', False, 'Create dataset tar archive')

MIN_DAMAGE = 100

def total_damage(row: dict) -> Optional[int]:
  total = 0
  for player in row['players']:
    damage = player.get('damage_taken')
    if damage is None:
      return None
    total += damage
  return total

def check_replay(row: dict, winner_only: bool = True) -> Optional[str]:
  if not row['valid']:
    return 'invalid'

  if not row['is_training']:
    return row['not_training_reason']

  if row['raw'].startswith('Phillip/'):
    model: str = row['name'].split('/')[0]
    if model.startswith('basic-') or 'imitation' in model:
      return 'vs weak phillip'

    # Only train on replays vs good players.
    for player in row['players']:
      # One of the players is always `Phillip AI` who is "known".
      name = nametags.name_from_metadata(player)
      if not nametags.is_known_player(name):
        return 'unknown player vs phillip'

  damage = total_damage(row)
  if damage is not None:
    if damage < MIN_DAMAGE:
      return 'insufficient damage dealt'
  elif winner_only and row.get('winner') is None:
    return 'no winner'

  return None

def main(_):
  with open(os.path.join(ROOT.value, 'parsed.pkl'), 'rb') as f:
    rows = pickle.load(f)

  # keep only training replays
  reasons = collections.Counter()
  match_ids = set()
  valid = []

  for row in tqdm.tqdm(rows, smoothing=0, unit='slp'):
    reason = check_replay(row, winner_only=WINNER_ONLY.value)
    if reason is not None:
      reasons[reason] += 1
      continue

    match = row.get('match')
    if match is not None and match['id']:
      match_id = (match['id'], match['game'], match['tiebreaker'])

      if match_id in match_ids:
        reasons['duplicate match ID'] += 1
        continue

      match_ids.add(match_id)

    valid.append(row)

  for reason, count in reasons.most_common():
    print(f'Filtered {100 * count / len(rows):.2f}% due to "{reason}"')

  print(f"Found {len(valid)}/{len(rows)} training replays.")
  del rows

  # fix numpy floats which json can't handle
  for row in valid:
    for player in row['players']:
      damage = player.get('damage_taken')
      if damage is not None:
        player['damage_taken'] = float(damage)

  make_tar = MAKE_TAR.value

  if make_tar:
    tar = tarfile.open(os.path.join(ROOT.value, 'training.tar'), 'w')

  missing = collections.Counter()

  for row in tqdm.tqdm(valid, smoothing=0, unit='slp'):
    md5 = row['slp_md5']
    parsed_path = os.path.join(ROOT.value, 'Parsed', md5)
    if not os.path.isfile(parsed_path):
      missing[row['raw']] += 1

    if make_tar:
      tar.add(parsed_path, arcname='games/' + md5)

  print(f"Missing: {missing}")

  # write metadata
  meta_path = os.path.join(ROOT.value, 'meta.json')
  with open(meta_path, 'w') as f:
    json.dump(valid, f, indent=2)

  if make_tar:
    tar.add(meta_path, arcname='meta.json')
    tar.close()

if __name__ == '__main__':
  app.run(main)
