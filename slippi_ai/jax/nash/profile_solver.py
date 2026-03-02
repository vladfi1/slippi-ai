from absl import app, flags
import numpy as np

import jax

from slippi_ai.jax.nash import optimization_test

ITERS = flags.DEFINE_integer('iters', 10, 'Number of tests to run')
SIZE = flags.DEFINE_integer('size', 10, 'Game size')
BATCH_SIZE = flags.DEFINE_integer('batch_size', 10, 'Batch size')
ERROR = flags.DEFINE_float('error', 1e-4, 'Algorithm error tolerance')
ATOL = flags.DEFINE_float('atol', 1e-1, 'Nash KL divergence tolerance')
VERIFY = flags.DEFINE_boolean('verify', False, 'Verify the solution')
MAX_STEPS = flags.DEFINE_integer('max_steps', 200, 'Maximum number of optimization steps')

dtypes = {
    'f32': np.float32,
    'f64': np.float64,
}
DTYPE = flags.DEFINE_enum('dtype', 'f64', dtypes.keys(), 'float type')

LINEAR = flags.DEFINE_boolean('linear', True, 'Linearity optimization')
CHOLESKY = flags.DEFINE_boolean('cholesky', False, 'Cholesky optimization')

def main(_):
  jax.config.update('jax_enable_x64', True)

  optimization_test.random_nash_tests(
      num_tests=ITERS.value,
      batch_size=BATCH_SIZE.value,
      size=(SIZE.value, SIZE.value),
      dtype=dtypes[DTYPE.value],
      error=ERROR.value,
      atol=ATOL.value,
      verify=VERIFY.value,
      is_linear=LINEAR.value,
      cholesky=CHOLESKY.value,
      max_steps=MAX_STEPS.value,
  )

if __name__ == '__main__':
  app.run(main)
