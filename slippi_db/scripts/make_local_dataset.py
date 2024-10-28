import collections
import os
import pickle
import json
import tarfile
import tempfile
import tqdm

from absl import app, flags

ROOT = flags.DEFINE_string('root', None, 'root directory', required=True)
WINNER_ONLY = flags.DEFINE_boolean(
  'winner_only', True, 'only keep games that have a winner')

MAKE_TAR = flags.DEFINE_boolean('tar', False, 'Create dataset tar archive')

def is_valid_replay(row: dict):
  if not row.get('is_training'):
    return False

  return True

def main(_):
  with open(os.path.join(ROOT.value, 'parsed.pkl'), 'rb') as f:
    rows = pickle.load(f)

  # keep only training replays
  rows = [row for row in rows if is_valid_replay(row)]
  print(f"Found {len(rows)} training replays.")

  # keep only games with a winner
  # TODO: this throws away games that have a salty runback

  if WINNER_ONLY.value:
    rows = [row for row in rows if row.get('winner') is not None]
    print(f"Filtered to {len(rows)} games with a winner.")

  make_tar = MAKE_TAR.value

  if make_tar:
    tar = tarfile.open(os.path.join(ROOT.value, 'training.tar'), 'w')

  missing = collections.Counter()

  for row in tqdm.tqdm(rows, smoothing=0, unit='slp'):
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
    json.dump(rows, f, indent=2)

  if make_tar:
    tar.add(meta_path, arcname='meta.json')
    tar.close()

if __name__ == '__main__':
  app.run(main)
