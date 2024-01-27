import os
import pickle
import json
import tarfile
import tempfile

from absl import app, flags

ROOT = flags.DEFINE_string('root', None, 'root directory', required=True)
WINNER_ONLY = flags.DEFINE_boolean(
  'winner_only', True, 'only keep games that have a winner')

def main(_):
  with open(os.path.join(ROOT.value, 'parsed.pkl'), 'rb') as f:
    rows = pickle.load(f)

  # keep only training replays
  rows = [row for row in rows if row.get('is_training')]
  print(f"Found {len(rows)} training replays.")

  # keep only games with a winner
  if WINNER_ONLY.value:
    rows = [row for row in rows if row.get('winner') is not None]
    print(f"Filtered to {len(rows)} games with a winner.")

  tar = tarfile.open(os.path.join(ROOT.value, 'training.tar'), 'w')

  for row in rows:
    md5 = row['slp_md5']
    parsed_path = os.path.join(ROOT.value, 'Parsed', md5)
    assert os.path.isfile(parsed_path)
    tar.add(parsed_path, arcname='games/' + md5)

  with tempfile.TemporaryDirectory() as tmpdir:
    # write metadata
    meta_path = os.path.join(tmpdir, 'meta.json')
    with open(meta_path, 'w') as f:
      json.dump(rows, f, indent=2)
    tar.add(meta_path, arcname='meta.json')

  tar.close()

if __name__ == '__main__':
  app.run(main)
