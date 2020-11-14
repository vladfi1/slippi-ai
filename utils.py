import time

import numpy as np
import tensorflow as tf

def np_array(*vals):
  return np.array(vals)

def batch_nest(nests):
  return tf.nest.map_structure(np_array, *nests)

class Profiler:
  def __init__(self):
    self.cumtime = 0
    self.num_calls = 0

  def __enter__(self):
    self._enter_time = time.perf_counter()

  def __exit__(self, type, value, traceback):
    self.cumtime += time.perf_counter() - self._enter_time
    self.num_calls += 1

  def mean_time(self):
    return self.cumtime / self.num_calls
