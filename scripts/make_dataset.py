import os
import shutil
import pickle
import multiprocessing
import zlib

from absl import app
from absl import flags

import melee
import embed
import stats
import paths
import utils

FLAGS = flags.FLAGS
flags.DEFINE_integer('cores', 1, 'number of cores')
flags.DEFINE_boolean('compress', True, 'Compress with zlib.')
flags.DEFINE_enum('subset', None, stats.SUBSETS, 'Subset of full dataset.')

flags.DEFINE_string('src_dir', paths.DATASET_PATH, 'Folder with slippi replays.')
flags.DEFINE_string('dst_dir', paths.COMPRESSED_PATH, 'Where to create the dataset.')

def read_gamestates(replay_path):
  print("Reading from ", replay_path)
  console = melee.Console(is_dolphin=False,
                          allow_old_version=True,
                          path=replay_path)
  console.connect()

  gamestate = console.step()
  port_map = dict(zip(gamestate.player.keys(), [1, 2]))

  def fix_state(s):
    s.player = {port_map[p]: v for p, v in s.player.items()}

  while gamestate:
    fix_state(gamestate)
    yield gamestate
    gamestate = console.step()

# TODO: enable speeds?
embed_game = embed.make_game_embedding()

def game_to_numpy(replay_path):
  states = read_gamestates(replay_path)
  states = map(embed_game.from_state, states)
  return utils.batch_nest(states)

def slp_to_pkl(src_dir, dst_dir, name, compress=False):
  src = os.path.join(src_dir, name)
  assert os.path.isfile(src)
  dst = os.path.join(dst_dir, name + '.pkl')
  if os.path.isfile(dst): return
  obj = game_to_numpy(src)
  obj_bytes = pickle.dumps(obj)
  if compress:
    obj_bytes = zlib.compress(obj_bytes)
  with open(dst, 'wb') as f:
    f.write(obj_bytes)

def batch_slp_to_pkl(src_dir, dst_dir, names, compress=False, cores=1):
  os.makedirs(dst_dir, exist_ok=True)

  dst_files = set(os.listdir(dst_dir))
  def is_new(name):
    return (name + '.pkl') not in dst_files
  names = list(filter(is_new, names))
  print(f"Converting {len(names)} replays.")

  # to see error messages
  if cores == 1:
    for name in names:
      try:
        slp_to_pkl(src_dir, dst_dir, name, compress)
      except Exception:
        print('Bad replay file', name)
    return

  with multiprocessing.Pool(cores) as pool:
    results = []
    for name in names:
      results.append(pool.apply_async(
          slp_to_pkl, [src_dir, dst_dir, name, compress]))
    for r in results:
      r.wait()

def main(_):
  if FLAGS.subset:
    subset = stats.get_subset(FLAGS.subset)
  else:
    subset = os.listdir(FLAGS.src_dir)
    subset = set(subset) - stats.BAD_NAMES

  batch_slp_to_pkl(
      FLAGS.src_dir,
      FLAGS.dst_dir,
      subset,
      FLAGS.compress,
      FLAGS.cores)

if __name__ == '__main__':
  app.run(main)
