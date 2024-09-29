import numpy as np
import tensorflow as tf
import sonnet as snt

def main():
  # Bug only occurs with check_numerics.
  tf.debugging.enable_check_numerics()

  batch_size = 3
  core_size = 5
  core = snt.GRU(core_size)
  linear = snt.Linear(1)

  @tf.function
  def compute(xs):
    def fn(acc, x):
      _, state = acc
      return core(x, state)

    with tf.GradientTape() as tape:
      initializer = (
          tf.zeros([batch_size, core_size]),
          core.initial_state(batch_size))
      outputs = tf.scan(fn, xs, initializer)

      # The actual error is attributed to the gradient computation.
      tape.gradient(outputs, tape.watched_variables())

    # Control dependency on inputs is required.
    with tf.control_dependencies([xs]):
      # A variable must be created in this scope.
      linear(xs)

  xs = np.zeros([2, batch_size, 1], dtype=np.float32)

  compute(xs)

main()
