#!/usr/bin/env python
"""Run decompression on our cluster, stopping afterwards.

This is a workaround for https://github.com/ray-project/ray/issues/17322#issuecomment-966115409.
"""

import subprocess

from absl import app
from absl import flags

# flags copied from decompress ray
ENV = flags.DEFINE_string('env', 'test', 'production environment')
IN_MEMORY = flags.DEFINE_bool('in_memory', True, 'Use ram instead of disk.')
PROCESSED = flags.DEFINE_bool(
    'processed', False, 'Decompress already-processed uploads.')
DRY_RUN = flags.DEFINE_bool('dry_run', False, 'Don\'t upload anything.')
STOP = flags.DEFINE_bool('stop', True, 'Stop the cluster after decompression.')

CLUSTER_FILE = 'slippi_db/decompression_cluster.yaml'

def _bool_flag(name: str, value: bool) -> str:
  return f'--{name}' if value else f'--no{name}'


def main(_):
  try:
    # Would be nicer to use the ray python API, but I'm not sure if it is supported.
    subprocess.check_call([
      'ray', 'submit', '--start', CLUSTER_FILE,
      'slippi_db/decompress_ray.py',
      '--env', ENV.value,
      _bool_flag('in_memory', IN_MEMORY.value),
      _bool_flag('processed', PROCESSED.value),
      _bool_flag('dry_run', DRY_RUN.value),
    ])
  finally:
    if STOP.value:
      subprocess.check_call(['ray', 'down', '-y', CLUSTER_FILE])

if __name__ == '__main__':
  app.run(main)
