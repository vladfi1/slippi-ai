import os
import shutil
import pickle
import multiprocessing

from absl import app
from absl import flags

import melee
import embed
import stats
import paths
import utils

FLAGS = flags.FLAGS
flags.DEFINE_integer('cores', 1, 'number of cores')

def get_fox_ditto_names():
  table = stats.table
  table = table[table.css_character_0 == melee.Character.FOX.value]
  table = table[table.css_character_1 == melee.Character.FOX.value]
  return table.filename

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

embed_game = embed.make_game_embedding()

def game_to_numpy(replay_path):
  states = read_gamestates(replay_path)
  states = map(embed_game.from_state, states)
  return utils.batch_nest(states)

def slp_to_pkl(src_dir, dst_dir, name):
  src = src_dir + name
  dst = dst_dir + name + '.pkl'
  if os.path.isfile(dst): return
  obj = game_to_numpy(src)
  with open(dst, 'wb') as f:
    pickle.dump(obj, f)

def batch_slp_to_pkl(src_dir, dst_dir, names, cores=1):
  with multiprocessing.Pool(cores) as pool:
    results = []
    for name in names:
      results.append(pool.apply_async(slp_to_pkl, [src_dir, dst_dir, name]))
    for r in results:
      r.wait()

def main(_):
  # print(len(get_fox_ditto_names()))
  batch_slp_to_pkl(
      paths.DATASET_PATH,
      paths.FOX_DITTO_PATH,
      get_fox_ditto_names(),
      FLAGS.cores)

if __name__ == '__main__':
  app.run(main)
