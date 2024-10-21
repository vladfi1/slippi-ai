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
JIT = flags.DEFINE_boolean('jit', False, 'Enable JIT compilation')

dtypes = {
    'f32': np.float32,
    'f64': np.float64,
}
DTYPE = flags.DEFINE_enum('dtype', 'f64', dtypes.keys(), 'float type')

solvers = {
    'barrier': optimization.solve_optimization_interior_point_barrier,
    'primal_dual': optimization.solve_optimization_interior_point_primal_dual,
}
SOLVER = flags.DEFINE_enum('solver', 'primal_dual', solvers.keys(), 'Optimization solver')
LINEAR = flags.DEFINE_boolean('linear', True, 'Linearity optimization')
CHOLESKY = flags.DEFINE_boolean('cholesky', False, 'Cholesky optimization')

def main(_):
  optimization_test.random_nash_tests(
      optimization_solver=solvers[SOLVER.value],
      num_tests=ITERS.value,
      batch_size=BATCH_SIZE.value,
      size=(SIZE.value, SIZE.value),
      dtype=dtypes[DTYPE.value],
      error=ERROR.value,
      atol=ATOL.value,
      verify=VERIFY.value,
      jit_compile=JIT.value,
      is_linear=LINEAR.value,
      cholesky=CHOLESKY.value,
  )

if __name__ == '__main__':
  app.run(main)
