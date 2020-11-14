import numpy as np
import tensorflow as tf

def np_array(*vals):
  return np.array(vals)

def batch_nest(nests):
  return tf.nest.map_structure(np_array, *nests)
