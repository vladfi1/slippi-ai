"""Profile multiple dolphins on a single cpu core.

The point of this is to better match model inference, which even on cpu
(and very much on gpu) can slowly execute a large number of steps in parallel.
"""

import time

from absl import app
from absl import flags
import fancyflags as ff
import ray

from slippi_ai import eval_lib, profiling_utils

DOLPHIN = ff.DEFINE_dict('dolphin', **eval_lib.DOLPHIN_FLAGS)

flags.DEFINE_multi_integer('n', [1], 'dolphins per core')
flags.DEFINE_integer('cpus', 1, 'number of cpu cores')
flags.DEFINE_bool('set_affinity', True, 'set cpu affinity')
flags.DEFINE_integer('runtime', 5, 'Running time, in seconds.')
flags.DEFINE_boolean('ray', False, 'Use ray for multiprocessing.')

FLAGS = flags.FLAGS


def run(runtime: int, n: int, cpus: int):
  env_class = (
    profiling_utils.RayMultiSerialEnv if FLAGS.ray else
    profiling_utils.MultiSerialEnv)

  env = env_class(n, cpus, FLAGS.set_affinity, DOLPHIN.value)

  # warmup gets through menus
  print('Warmup step.')
  env.step()
  print('Warmup done.')

  start_time = time.perf_counter()
  run_time = 0.
  count = 0

  while run_time < runtime:
    env.step()
    count += 1
    run_time = time.perf_counter() - start_time

  env.stop()

  sps = count / run_time
  tot = n * cpus * sps
  print(f'{n:03d}: sps: {sps:.1f}, tot: {tot:.1f}')
  return sps, tot

def main(_):
  if FLAGS.ray:
    ray.init()  # necessary for RayMultiSerialEnv

  ns = FLAGS.n
  stats = [run(FLAGS.runtime, n, FLAGS.cpus) for n in ns]

  for n, (sps, tot) in zip(ns, stats):
    print(f'{n:03d}: sps: {sps:.1f}, tot: {tot:.1f}')

  for n, (sps, tot) in zip(ns, stats):
    print(f'{n} {sps:.1f} {tot:.1f}')

if __name__ == '__main__':
  app.run(main)