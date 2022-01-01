import time

from absl import app
from absl import flags
import ray

from slippi_ai import dolphin

FLAGS = flags.FLAGS
flags.DEFINE_string('dolphin_path', None, 'Path to dolphin directory.', required=True)
flags.DEFINE_string('iso_path', None, 'Path to SSBM iso.', required=True)
flags.DEFINE_integer('N', 1, 'number of dolphin instances')
flags.DEFINE_integer('frames', 1 * 60 * 60, 'number of frames to run for')
flags.DEFINE_boolean('render', False, 'render graphics')
flags.DEFINE_float('overclock', None, 'cpu overclock')

Dolphin = ray.remote(dolphin.Dolphin)

def main(_):
  players = {1: dolphin.CPU(), 2: dolphin.CPU()}
  dolphins = []

  for i in range(FLAGS.N):
    dolphins.append(Dolphin.remote(
        FLAGS.dolphin_path, FLAGS.iso_path, players,
        slippi_port=51441 + i,
        render=FLAGS.render,
        overclock=FLAGS.overclock,
        exe_name='dolphin-emu-nogui',
    ))

  def sync_step():
    states = [d.step.remote() for d in dolphins]
    ray.wait(states, num_returns=len(states), fetch_local=False)

  sync_step()

  start_time = time.perf_counter()

  for d in dolphins:
    d.multi_step.remote(FLAGS.frames)

  sync_step()
  run_time = time.perf_counter() - start_time

  for d in dolphins:
    d.stop.remote()

  fps = FLAGS.frames / run_time
  total_fps = fps * FLAGS.N

  print(f'fps: {fps:.1f}, total_fps: {total_fps: .1f}')

if __name__ == '__main__':
  app.run(main)
