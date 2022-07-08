"""Profile multiple dolphins on a single cpu core.

The point of this is to better match model inference, which even on cpu
(and very much on gpu) can slowly execute a large number of steps in parallel.
"""

import time

from absl import app
from absl import flags
import fancyflags as ff
import psutil

from slippi_ai import eval_lib
from slippi_ai import dolphin as dolphin_lib

DOLPHIN = ff.DEFINE_dict('dolphin', **eval_lib.DOLPHIN_FLAGS)

flags.DEFINE_multi_integer('n', [1], 'number of dolphin instances')
flags.DEFINE_integer('runtime', 5, 'Running time, in seconds.')

FLAGS = flags.FLAGS

def run(n: int, runtime: float, cpu: int = 0):
  players = {1: dolphin_lib.AI(), 2: dolphin_lib.CPU()}

  main_proc = psutil.Process()
  main_proc.cpu_affinity([cpu])

  dolphins = [
      dolphin_lib.Dolphin(players=players, **DOLPHIN.value)
      for _ in range(n)]

  for d in dolphins:
    proc = psutil.Process(d.console._process.pid)
    proc.cpu_affinity([cpu])

  def step():
    # intentionally serial
    for d in dolphins:
      d.step()

  print('Warmup step.')
  step()
  print('Warmup done, starting profiling.')

  start_time = time.perf_counter()

  count = 0
  while True:
    step()
    count += 1

    run_time = time.perf_counter() - start_time
    if run_time > runtime:
      break

  for d in dolphins:
    d.stop()

  fps = count / run_time
  sps = n * fps
  print(f'{n:03d}: fps: {fps:.1f}, sps: {sps:.1f}')
  return fps, sps

def main(_):
  ns = FLAGS.n
  stats = [run(n, FLAGS.runtime) for n in ns]

  for n, (fps, sps) in zip(ns, stats):
    print(f'{n:03d}: fps: {fps:.1f}, sps: {sps:.1f}')

  for n, (fps, sps) in zip(ns, stats):
    print(f'{n} {fps:.1f} {sps:.1f}')

if __name__ == '__main__':
  app.run(main)