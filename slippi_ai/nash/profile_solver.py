from absl import app, flags
import numpy as np
import tensorflow as tf
import tqdm

from slippi_ai.nash import optimization_test
from slippi_ai.nash import nash, optimization

ITERS = flags.DEFINE_integer('iters', 10, 'Number of tests to run')
SIZE = flags.DEFINE_integer('size', 10, 'Game size')
BATCH_SIZE = flags.DEFINE_integer('batch_size', 10, 'Batch size')
ERROR = flags.DEFINE_float('error', 1e-4, 'Algorithm error tolerance')
ATOL = flags.DEFINE_float('atol', 1e-1, 'Nash KL divergence tolerance')
VERIFY = flags.DEFINE_boolean('verify', False, 'Verify the solution')

def main(_):
  optimization_test.random_nash_tests(
      num_tests=ITERS.value,
      batch_size=BATCH_SIZE.value,
      size=(SIZE.value, SIZE.value),
      dtype=np.float64,
      error=ERROR.value,
      atol=ATOL.value,
      verify=VERIFY.value,
  )

if __name__ == '__main__':
  app.run(main)
