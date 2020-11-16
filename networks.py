import sonnet as snt
import tensorflow as tf

class Network(snt.Module):

  def initial_state(self, batch_size):
    raise NotImplementedError()

  def step(self, inputs, prev_state):
    """Returns outputs and next recurrent state."""
    raise NotImplementedError()

  def unroll(self, inputs, initial_state):
    # TODO: make sure this compiles to a dynamic unroll
    outputs = tf.TensorArray(dtype=tf.float32, size=inputs.shape[0])
    state = initial_state
    i = tf.constant(0)
    for input_ in inputs:
      output, state = self.step(input_, state)
      outputs = outputs.write(i, output)
      i = i + 1
    return outputs.stack(), state

class MLP(Network):

  CONFIG = dict(
      output_sizes=[256, 128],
      dropout_rate=0.,
  )

  def __init__(self, output_sizes, dropout_rate):
    super().__init__(name='MLP')
    self._mlp = snt.nets.MLP(
        output_sizes,
        activate_final=True,
        dropout_rate=dropout_rate)

  def initial_state(self, batch_size):
    return ()

  def step(self, inputs, prev_state):
    del prev_state
    return self._mlp(inputs), ()

  def unroll(self, inputs, initial_state):
    del initial_state
    return self._mlp(inputs), ()

class LSTM(Network):
  CONFIG=dict(hidden_size=128)

  def __init__(self, hidden_size):
    super().__init__(name='LSMT')
    self._hidden_size = hidden_size
    self._lstm = snt.LSTM(hidden_size)

  def initial_state(self, batch_size):
    return self._lstm.initial_state(batch_size)

  def step(self, inputs, prev_state):
    return self._lstm(inputs, prev_state)

CONSTRUCTORS = dict(
    mlp=MLP,
    lstm=LSTM,
)

DEFAULT_CONFIG = dict(
    name='mlp',
    mlp=MLP.CONFIG,
    lstm=LSTM.CONFIG,
)

def construct_network(name, **config):
  return CONSTRUCTORS[name](**config[name])
