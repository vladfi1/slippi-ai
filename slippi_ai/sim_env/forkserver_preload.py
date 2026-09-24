"""Preload for the sim env worker forkserver.

MultiprocessSimEnvironment starts its workers from a multiprocessing
forkserver that imports this module first. Everything loaded here is
inherited copy-on-write by every worker, so this is where the expensive
per-process state goes: the python imports the worker needs (~120MB) and
melee_sim's immutable game data (~300MB, loaded once by preload_game_data and
shared by every EnvBatch in the process, see melee-sim-light's api.h).
Measured on a 16-env worker, this takes its private memory from ~424MB down
to ~15MB.

The forkserver is a fresh interpreter, so unlike the training process it has
no JAX backend or CUDA context to worry about when forking.
"""

import gc
import logging
import os

# The env workers only step the simulator and copy numpy arrays; they have no
# use for a BLAS thread pool, and OpenBLAS would otherwise spin up one thread
# per core in every worker. Must be set before numpy is first imported.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import melee_sim

from slippi_ai.sim_env import multiprocess_env  # noqa: F401  (worker imports)


def preload_game_data():
  """Loads the game data workers will share. Best effort: on failure the
  workers each load their own copy, as they would without a forkserver."""
  try:
    data_dir = melee_sim.preload_game_data()
  except Exception:  # pylint: disable=broad-except
    logging.exception('melee_sim game data preload failed')
    return
  logging.info('preloaded melee_sim game data from %s', data_dir)


preload_game_data()

# Move everything loaded so far out of the cyclic GC's reach. Otherwise the
# first collection in each forked worker would write to the header of every
# tracked object, copying most of the inherited pages.
gc.collect()
gc.freeze()
