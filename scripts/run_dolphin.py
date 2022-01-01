import time

from absl import app
from absl import flags

from slippi_ai import dolphin as dolphin_lib

FLAGS = flags.FLAGS
flags.DEFINE_string('dolphin_path', None, 'Path to dolphin directory.', required=True)
flags.DEFINE_string('iso_path', None, 'Path to SSBM iso.', required=True)
flags.DEFINE_integer('N', 1, 'number of dolphin instances')
flags.DEFINE_integer('frames', 1 * 60 * 60, 'number of frames to run for')
flags.DEFINE_boolean('render', False, 'render graphics')

def main(_):
  players = {1: dolphin_lib.CPU(), 2: dolphin_lib.CPU()}

  dolphin = dolphin_lib.Dolphin(
      FLAGS.dolphin_path, FLAGS.iso_path, players,
      render=FLAGS.render,
  )

  dolphin.step()

  start_time = time.perf_counter()

  for _ in range(FLAGS.frames):
    dolphin.step()

  run_time = time.perf_counter() - start_time
  dolphin.stop()

  fps = FLAGS.frames / run_time

  print(f'fps: {fps:.1f}')

if __name__ == '__main__':
  app.run(main)
