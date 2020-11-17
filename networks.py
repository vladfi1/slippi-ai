import sonnet as snt
import tensorflow as tf

import embed

def dynamic_rnn(core, inputs, initial_state):
  outputs = tf.TensorArray(dtype=tf.float32, size=inputs.shape[0])
  state = initial_state
  for i in tf.range(tf.shape(inputs)[0]):
    input_ = inputs[i]  # TODO: handle nested inputs
    output, state = core(input_, state)
    outputs = outputs.write(i, output)
  return outputs.stack(), state

class Network(snt.Module):

  def initial_state(self, batch_size):
    raise NotImplementedError()

  def step(self, inputs, prev_state):
    """Returns outputs and next recurrent state."""
    raise NotImplementedError()

  def unroll(self, inputs, initial_state):
    return dynamic_rnn(self.step, inputs, initial_state)

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
    self._embed_game = embed.make_game_embedding()

  def initial_state(self, batch_size):
    return ()

  def step(self, inputs, prev_state):
    del prev_state
    flat_inputs = self._embed_game(inputs)
    return self._mlp(flat_inputs), ()

  def unroll(self, inputs, initial_state):
    del initial_state
    flat_inputs = self._embed_game(inputs)
    return self._mlp(flat_inputs), ()

class LSTM(Network):
  CONFIG=dict(hidden_size=128)

  def __init__(self, hidden_size):
    super().__init__(name='LSMT')
    self._hidden_size = hidden_size
    self._lstm = snt.LSTM(hidden_size)
    self._embed_game = embed.make_game_embedding()

  def initial_state(self, batch_size):
    return self._lstm.initial_state(batch_size)

  def step(self, inputs, prev_state):
    flat_inputs = self._embed_game(inputs)
    return self._lstm(flat_inputs, prev_state)

  def unroll(self, inputs, prev_state):
    flat_inputs = self._embed_game(inputs)
    return dynamic_rnn(self._lstm, flat_inputs, prev_state)

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
