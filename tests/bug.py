import numpy as np
import tensorflow as tf
import sonnet as snt

class Thunk:

  def __init__(self, constructor):
    self.constructor = constructor
    self.value = None

  def get(self):
    if self.value is None:
      self.value = self.constructor()
    return self.value

def main():
  # Bug only occurs with check_numerics.
  tf.debugging.enable_check_numerics()

  batch_size = 3
  core_size = 5
  core = snt.GRU(core_size)
  v = Thunk(lambda: tf.Variable(1, dtype=tf.float32))

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

    # Error goes away if we initialize the variable outside the control-dep.
    # v.get()

    # Control dependency on inputs is required.
    with tf.control_dependencies([xs]):
      # A variable must be created in this scope.
      xs + v.get()

  xs = np.zeros([2, batch_size, 1], dtype=np.float32)

  compute(xs)

main()

# The resulting error is something like this:
# Inputs to operation cond/else/_1/cond/StatefulPartitionedCall/gradient_tape/scan/while/scan/while_grad/body/_372/gradient_tape/scan/while/gradients/AddN_6 of type AddN must have the same size and shape.  Input 0: [0] != input 1: [5,15]
# 	 [[{{node gradient_tape/scan/while/gradients/AddN_6}}]] [Op:__inference_fn_with_cond_1472]
