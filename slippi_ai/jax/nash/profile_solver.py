from absl import app, flags
import numpy as np

import jax

from slippi_ai.jax.nash import nash
from slippi_ai.jax.nash import optimization_test

ITERS = flags.DEFINE_integer('iters', 10, 'Number of tests to run')
SIZE = flags.DEFINE_integer('size', 10, 'Game size')
BATCH_SIZE = flags.DEFINE_integer('batch_size', 10, 'Batch size')
ERROR = flags.DEFINE_float('error', 1e-4, 'Algorithm error tolerance')
ATOL = flags.DEFINE_float('atol', 1e-1, 'Nash KL divergence tolerance')
VERIFY = flags.DEFINE_boolean('verify', False, 'Verify the solution')
MAX_STEPS = flags.DEFINE_integer('max_steps', 200, 'Maximum number of optimization steps')
SOLVER = flags.DEFINE_enum('solver', 'ippd', ['ippd', 'qpax'], 'Solver to use')

dtypes = {
    'f32': np.float32,
    'f64': np.float64,
}
DTYPE = flags.DEFINE_enum('dtype', 'f64', dtypes.keys(), 'float type')

LINEAR = flags.DEFINE_boolean('linear', True, 'Linearity optimization')
CHOLESKY = flags.DEFINE_boolean('cholesky', False, 'Cholesky optimization')

PROFILE_SERVER_PORT = flags.DEFINE_integer('profile_server_port', None, 'Port for the profile server')
PROFILE_TRACE_DIR = flags.DEFINE_string('profile_trace_dir', None, 'Directory to save profile traces')

_SOLVERS = {
    'ippd': nash.solve_zero_sum_nash_ippd,
    'qpax': nash.solve_zero_sum_nash_qpax,
}

def main(_):
  jax.config.update('jax_enable_x64', True)

  if PROFILE_SERVER_PORT.value is not None:
    jax.profiler.start_server(PROFILE_SERVER_PORT.value)

  if PROFILE_TRACE_DIR.value is not None:
    jax.profiler.start_trace(PROFILE_TRACE_DIR.value)

  optimization_test.random_nash_tests(
      num_tests=ITERS.value,
      batch_size=BATCH_SIZE.value,
      size=(SIZE.value, SIZE.value),
      solver=_SOLVERS[SOLVER.value],
      dtype=dtypes[DTYPE.value],
      error=ERROR.value,
      atol=ATOL.value,
      verify=VERIFY.value,
      is_linear=LINEAR.value,
      cholesky=CHOLESKY.value,
      max_steps=MAX_STEPS.value,
  )

  if PROFILE_TRACE_DIR.value is not None:
    jax.profiler.stop_trace()

if __name__ == '__main__':
  app.run(main)
