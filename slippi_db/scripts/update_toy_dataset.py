import json
import os
import shutil

from absl import app, flags


flags.DEFINE_string('dataset', None, 'Source directory', required=True)
flags.DEFINE_string('toy_path', 'slippi_ai/data/toy_dataset', 'Path to toy dataset.')
flags.DEFINE_integer('num_games', 1, 'Number of games to include in the toy dataset.')

def main(_):
  meta_path = os.path.join(flags.FLAGS.dataset, 'meta.json')

  with open(meta_path, 'r') as f:
    meta = json.load(f)

  found = []

  for row in meta:
    if row['valid'] and row['is_training']:
      found.append(row)

    if len(found) >= flags.FLAGS.num_games:
      break

  if len(found) < flags.FLAGS.num_games:
    raise ValueError(f'Found only {len(found)} valid games, wanted {flags.FLAGS.num_games}.')

  if not os.path.exists(flags.FLAGS.toy_path):
    os.makedirs(flags.FLAGS.toy_path)

  with open(os.path.join(flags.FLAGS.toy_path, 'meta.json'), 'w') as f:
    json.dump(found, f, indent=2)

  input_games_dir = os.path.join(flags.FLAGS.dataset, 'Parsed')
  output_games_dir = os.path.join(flags.FLAGS.toy_path, 'games')

  shutil.rmtree(output_games_dir, ignore_errors=True)
  os.makedirs(output_games_dir)

  for row in found:
    input_game = os.path.join(input_games_dir, row['slp_md5'])
    output_game = os.path.join(output_games_dir, row['slp_md5'])

    shutil.copy(input_game, output_game)

if __name__ == '__main__':
  app.run(main)
