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

SKIP_TAR = flags.DEFINE_boolean('skip_tar', False, 'Only create meta.json')

def is_valid_replay(row: dict):
  if not row.get('is_training'):
    return False

  # Mango has stated he doesn't want AI trained on his replays
  if 'Mango' in row['raw']:
    return False

  return True

def main(_):
  with open(os.path.join(ROOT.value, 'parsed.pkl'), 'rb') as f:
    rows = pickle.load(f)

  # keep only training replays
  rows = [row for row in rows if is_valid_replay(row)]
  print(f"Found {len(rows)} training replays.")

  # keep only games with a winner
  # TODO: this throws away games that have a salty runback; instead we should

  if WINNER_ONLY.value:
    rows = [row for row in rows if row.get('winner') is not None]
    print(f"Filtered to {len(rows)} games with a winner.")

  use_tar = not SKIP_TAR.value

  if use_tar:
    tar = tarfile.open(os.path.join(ROOT.value, 'training.tar'), 'w')

  for row in tqdm.tqdm(rows, smoothing=0, unit='slp'):
    md5 = row['slp_md5']
    parsed_path = os.path.join(ROOT.value, 'Parsed', md5)
    assert os.path.isfile(parsed_path), row

    if use_tar:
      tar.add(parsed_path, arcname='games/' + md5)

  with tempfile.TemporaryDirectory() as tmpdir:
    # write metadata
    meta_path = os.path.join(tmpdir, 'meta.json')
    with open(meta_path, 'w') as f:
      json.dump(rows, f, indent=2)

    if use_tar:
      tar.add(meta_path, arcname='meta.json')

  if use_tar:
    tar.close()

if __name__ == '__main__':
  app.run(main)
