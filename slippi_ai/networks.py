from typing import Any, Optional, Tuple

import sonnet as snt
import tensorflow as tf
from tensorflow.python.framework.tensor import Tensor

from slippi_ai import embed, utils

RecurrentState = Any
Inputs = tf.Tensor

# TODO: make configurable
INPUT_SIZE = embed.default_embed_game.size

class Network(snt.Module):

  def initial_state(self, batch_size: int) -> RecurrentState:
    raise NotImplementedError()

  def step(
      self,
      inputs: Inputs,
      prev_state: RecurrentState,
  ) -> Tuple[tf.Tensor, RecurrentState]:
    '''
      Returns outputs and next recurrent state.
      inputs: (batch_size, x_dim)
      prev_state: (batch, state_dim)

    Returns a tuple (outputs, final_state)
      outputs: (batch, out_dim)
      final_state: (batch, state_dim)
    '''
    raise NotImplementedError()

  def unroll(
      self,
      inputs: Inputs,
      initial_state: RecurrentState,
  ) -> Tuple[tf.Tensor, RecurrentState]:
    '''
    Arguments:
      inputs: (time, batch, x_dim)
      initial_state: (batch, state_dim)

    Returns a tuple (outputs, final_state)
      outputs: (time, batch, out_dim)
      final_state: (batch, state_dim)
    '''
    return utils.dynamic_rnn(self.step, inputs, initial_state)

class MLP(Network):
  CONFIG = dict(
      depth=2,
      width=128,
      dropout_rate=0.0,
  )

  def __init__(self, depth, width, dropout_rate):
    super().__init__(name='MLP')
    self._mlp = snt.nets.MLP(
        [width] * depth,
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

class FrameStackingMLP(Network):

  CONFIG = dict(
      output_sizes=[256, 128],
      dropout_rate=0.,
      num_frames=5,
  )

  def __init__(self, output_sizes, dropout_rate, num_frames):
    super().__init__(name='MLP')
    self._mlp = snt.nets.MLP(
        output_sizes,
        activate_final=True,
        dropout_rate=dropout_rate)
    self._num_frames = num_frames

  def initial_state(self, batch_size):
    return [
        tf.zeros([batch_size, INPUT_SIZE])
        for _ in range(self._num_frames-1)
    ]

  def step(self, inputs, prev_state):
    frames = prev_state + [inputs]
    next_state = frames[1:]
    stacked_frames = tf.concat(frames, -1)
    return self._mlp(stacked_frames), next_state

  def unroll(self, inputs, initial_state):
    past_frames = tf.stack(initial_state)  # [N-1, B, ...]
    all_frames = tf.concat([past_frames, inputs], 0)  # [N-1 + T, B, ...]

    n = self._num_frames - 1

    # slices has N tensors of dimension [T, B, ...]
    slices = [all_frames[i:i-n] for i in range(n)]
    slices.append(all_frames[n:])
    stacked_frames = tf.concat(slices, -1)

    final_state = [all_frames[i] for i in range(-n, 0)]

    return self._mlp(stacked_frames), final_state

class LayerNorm(snt.Module):
  """Normalize the mean (to 0) and standard deviation (to 1) of the last dimension.

  We use our own instead of sonnet's because sonnet doesn't allow varying rank.
  """

  def __init__(self):
    super().__init__(name='LayerNorm')

  @snt.once
  def _initialize(self, inputs):
    feature_shape = inputs.shape[-1:]
    self.scale = tf.Variable(
        tf.ones(feature_shape, dtype=inputs.dtype),
        name='scale')
    self.bias = tf.Variable(
        tf.zeros(feature_shape, dtype=inputs.dtype),
        name='bias')

  def __call__(self, inputs):
    self._initialize(inputs)

    mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
    inputs -= mean

    stddev = tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True))
    inputs /= stddev

    inputs *= self.scale
    inputs += self.bias

    return inputs

class ResBlock(snt.Module):

  def __init__(
      self,
      residual_size: int,
      hidden_size: Optional[int] = None,
      activation=tf.nn.relu,
      name='ResBlock'):
    super().__init__(name=name)
    self.block = snt.Sequential([
        # https://openreview.net/forum?id=B1x8anVFPr recommends putting the layernorm here
        LayerNorm(),
        snt.Linear(hidden_size or residual_size),
        activation,
        # initialize the resnet as the identity function
        snt.Linear(residual_size, w_init=tf.zeros_initializer()),
    ])

  def __call__(self, residual):
    return residual + self.block(residual)

def resnet(num_blocks, residual_size, hidden_size=None, with_encoder=True):
  layers = []
  if with_encoder:
    layers.append(snt.Linear(residual_size))
  for _ in range(num_blocks):
    layers.append(ResBlock(residual_size, hidden_size))
  return snt.Sequential(layers)

class LSTM(Network):
  CONFIG=dict(
      hidden_size=128,
      num_res_blocks=0,
  )

  def __init__(self, hidden_size, num_res_blocks):
    super().__init__(name='LSTM')
    self._hidden_size = hidden_size
    self._lstm = snt.LSTM(hidden_size)

    # use a resnet before the LSTM
    self._resnet = resnet(num_res_blocks, hidden_size)

  def initial_state(self, batch_size):
    return self._lstm.initial_state(batch_size)

  def step(self, inputs, prev_state):
    inputs = self._resnet(inputs)
    return self._lstm(inputs, prev_state)

  def unroll(self, inputs, prev_state):
    inputs = self._resnet(inputs)
    return utils.dynamic_rnn(self._lstm, inputs, prev_state)


class ResLSTMBlock(snt.RNNCore):

  def __init__(self, residual_size, hidden_size=None, name='ResLSTMBlock'):
    super().__init__(name=name)
    self.layernorm = LayerNorm()
    self.lstm = snt.LSTM(hidden_size or residual_size)
    # initialize the resnet as the identity function
    self.decoder = snt.Linear(residual_size, w_init=tf.zeros_initializer())

  def initial_state(self, batch_size):
    return self.lstm.initial_state(batch_size)

  def __call__(self, residual, prev_state):
    x = residual
    x = self.layernorm(x)
    x, next_state = self.lstm(x, prev_state)
    x = self.decoder(x)
    return residual + x, next_state

class DeepResLSTM(Network):
  CONFIG=dict(
      hidden_size=128,
      num_layers=1,
  )

  def __init__(self, hidden_size, num_layers):
    super().__init__(name='DeepResLSTM')
    self.encoder = snt.Linear(hidden_size)
    self.deep_rnn = snt.DeepRNN(
        [ResLSTMBlock(hidden_size) for _ in range(num_layers)])

  def initial_state(self, batch_size):
    return self.deep_rnn.initial_state(batch_size)

  def step(self, inputs, prev_state):
    inputs = self.encoder(inputs)
    return self.deep_rnn(inputs, prev_state)

  def unroll(self, inputs, prev_state):
    inputs = self.encoder(inputs)
    return utils.dynamic_rnn(self.deep_rnn, inputs, prev_state)

class GRU(Network):
  CONFIG=dict(hidden_size=128)

  def __init__(self, hidden_size):
    super().__init__(name='GRU')
    self._hidden_size = hidden_size
    self._gru = snt.GRU(hidden_size)

  def initial_state(self, batch_size):
    return self._gru.initial_state(batch_size)

  def step(self, inputs, prev_state):
    return self._gru(inputs, prev_state)

  def unroll(self, inputs, prev_state):
    return utils.dynamic_rnn(self._gru, inputs, prev_state)

class RecurrentWrapper(Network):
  """Wraps an RNNCore as a Network."""

  def __init__(self, core: snt.RNNCore, name='RecurrentWrapper'):
    super().__init__(name=name)
    self._core = core

  def initial_state(self, batch_size):
    return self._core.initial_state(batch_size)

  def step(self, inputs, prev_state):
    return self._core(inputs, prev_state)

  def unroll(self, inputs, prev_state):
    return utils.dynamic_rnn(self._core, inputs, prev_state)

class FFWWrapper(Network):
  """Wraps an MLP as a Network."""

  def __init__(self, mlp: snt.Module, name='FFWWrapper'):
    super().__init__(name=name)
    self._mlp = mlp

  def initial_state(self, batch_size):
    return ()

  def step(self, inputs, prev_state):
    del prev_state
    return self._mlp(inputs), ()

  def unroll(self, inputs, prev_state):
    del prev_state
    return self._mlp(inputs), ()

class Sequential(Network):

  def __init__(self, layers: list[Network], name='Sequential'):
    super().__init__(name=name)
    self._layers = layers

  def initial_state(self, batch_size):
    return [layer.initial_state(batch_size) for layer in self._layers]

  def step(self, inputs, prev_state):
    next_states = []
    for layer, state in zip(self._layers, prev_state):
      inputs, next_state = layer.step(inputs, state)
      next_states.append(next_state)
    return inputs, next_states

  def unroll(self, inputs, prev_state):
    final_states = []
    for layer, state in zip(self._layers, prev_state):
      inputs, final_state = layer.unroll(inputs, state)
      final_states.append(final_state)
    return inputs, final_states

class ResidualWrapper(Network):

  def __init__(self, net: Network):
    super().__init__(name='ResidualWrapper')
    self._net = net

  def initial_state(self, batch_size):
    return self._net.initial_state(batch_size)

  def step(self, inputs, prev_state):
    outputs, next_state = self._net.step(inputs, prev_state)
    return inputs + outputs, next_state

  def unroll(self, inputs, prev_state):
    outputs, final_state = self._net.unroll(inputs, prev_state)
    return inputs + outputs, final_state

class TransformerLike(Network):
  """Alternates recurrent and FFW layers."""

  CONFIG=dict(
      hidden_size=128,
      num_layers=1,
      ffw_multiplier=4,
      recurrent_layer='lstm',
      activation='gelu',
  )

  def __init__(
      self,
      hidden_size: int,
      num_layers: int,
      ffw_multiplier: int,
      recurrent_layer: str,
      activation: str,
  ):
    super().__init__(name=f'TransformerLike')
    self._hidden_size = hidden_size
    self._num_layers = num_layers
    self._ffw_multiplier = ffw_multiplier

    self._recurrent_constructor = dict(
        lstm=snt.LSTM,
        gru=snt.GRU,
    )[recurrent_layer]

    layers = []

    # We need to encode for the first residual
    encoder = snt.Linear(hidden_size, name='encoder')
    layers.append(FFWWrapper(encoder))

    for _ in range(num_layers):
      # consider adding layernorm here?
      recurrent_layer = ResidualWrapper(RecurrentWrapper(
          self._recurrent_constructor(hidden_size)))
      layers.append(recurrent_layer)

      ffw_layer = ResBlock(
          hidden_size, hidden_size * ffw_multiplier,
          activation=getattr(tf.nn, activation))
      layers.append(FFWWrapper(ffw_layer))

    self._net = Sequential(layers)

  def initial_state(self, batch_size: int) -> RecurrentState:
    return self._net.initial_state(batch_size)

  def step(self, inputs: Inputs, prev_state: RecurrentState) -> Tuple[Tensor, RecurrentState]:
    return self._net.step(inputs, prev_state)

  def unroll(self, inputs: Inputs, initial_state: RecurrentState) -> Tuple[Tensor, RecurrentState]:
    return self._net.unroll(inputs, initial_state)

CONSTRUCTORS = dict(
    mlp=MLP,
    frame_stack_mlp=FrameStackingMLP,
    lstm=LSTM,
    gru=GRU,
    res_lstm=DeepResLSTM,
    tx_like=TransformerLike,
)

DEFAULT_CONFIG = dict(
    name='mlp',
    mlp=MLP.CONFIG,
    frame_stack_mlp=FrameStackingMLP.CONFIG,
    lstm=LSTM.CONFIG,
    gru=GRU.CONFIG,
    res_lstm=DeepResLSTM.CONFIG,
    tx_like=TransformerLike.CONFIG,
)

def construct_network(name, **config):
  return CONSTRUCTORS[name](**config[name])
