from typing import Any, Tuple

import sonnet as snt
import tensorflow as tf

from slippi_ai import embed, utils, types

RecurrentState = Any
Inputs = Tuple[types.Nest[tf.Tensor], tf.Tensor]

# don't use opponent's controller
# our own will be exposed in the input
embed_game = embed.make_game_embedding(
    player_config=dict(with_controller=False))

def process_inputs(inputs: Inputs) -> tf.Tensor:
  gamestate, p1_controller_embed = inputs
  gamestate_embed = embed_game(gamestate)
  return tf.concat([gamestate_embed, p1_controller_embed], -1)

class Network(snt.Module):

  def initial_state(self, batch_size: int):
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
      output_sizes=[256, 128],
      dropout_rate=0.0,
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
    flat_inputs = process_inputs(inputs)
    return self._mlp(flat_inputs), ()

  def unroll(self, inputs, initial_state):
    del initial_state
    flat_inputs = process_inputs(inputs)
    return self._mlp(flat_inputs), ()

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
        tf.zeros([batch_size, embed_game.size])
        for _ in range(self._num_frames-1)
    ]

  def step(self, inputs, prev_state):
    gamestate, p1_controller_embed = inputs
    frames = prev_state + [embed_game(gamestate)]
    next_state = frames[1:]
    stacked_frames = tf.concat(frames + [p1_controller_embed], -1)
    return self._mlp(stacked_frames), next_state

  def unroll(self, inputs, initial_state):
    gamestate, p1_controller_embed = inputs
    gamestate_embed = embed_game(gamestate)  # [T, B, ...]

    past_frames = tf.stack(initial_state)  # [N-1, B, ...]
    all_frames = tf.concat([past_frames, gamestate_embed], 0)  # [N-1 + T, B, ...]

    n = self._num_frames - 1

    # slices has N tensors of dimension [T, B, ...]
    slices = [all_frames[i:i-n] for i in range(n)]
    slices.append(all_frames[n:])
    stacked_frames = tf.concat(slices + [p1_controller_embed], -1)

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
        tf.ones(feature_shape, inputs.dtype),
        name='scale')
    self.bias = tf.Variable(
        tf.zeros(feature_shape, inputs.dtype),
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

  def __init__(self, residual_size, hidden_size=None, name='ResBlock'):
    super().__init__(name=name)
    self.block = snt.Sequential([
        # https://openreview.net/forum?id=B1x8anVFPr recommends putting the layernorm here
        LayerNorm(),
        snt.Linear(hidden_size or residual_size),
        tf.nn.relu,
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
    flat_inputs = process_inputs(inputs)
    flat_inputs = self._resnet(flat_inputs)
    return self._lstm(flat_inputs, prev_state)

  def unroll(self, inputs, prev_state):
    flat_inputs = process_inputs(inputs)
    flat_inputs = self._resnet(flat_inputs)
    return utils.dynamic_rnn(self._lstm, flat_inputs, prev_state)


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
    flat_inputs = process_inputs(inputs)
    flat_inputs = self.encoder(flat_inputs)
    return self.deep_rnn(flat_inputs, prev_state)

  def unroll(self, inputs, prev_state):
    flat_inputs = process_inputs(inputs)
    flat_inputs = self.encoder(flat_inputs)
    return utils.dynamic_rnn(self.deep_rnn, flat_inputs, prev_state)

class GRU(Network):
  CONFIG=dict(hidden_size=128)

  def __init__(self, hidden_size):
    super().__init__(name='GRU')
    self._hidden_size = hidden_size
    self._gru = snt.GRU(hidden_size)

  def initial_state(self, batch_size):
    return self._gru.initial_state(batch_size)

  def step(self, inputs, prev_state):
    flat_inputs = process_inputs(inputs)
    return self._gru(flat_inputs, prev_state)

  def unroll(self, inputs, prev_state):
    flat_inputs = process_inputs(inputs)
    return utils.dynamic_rnn(self._gru, flat_inputs, prev_state)

class Copier(Network):
  '''
    No parameters - simply returns the previous controller state.
  '''
  CONFIG=dict()

  def __init__(self):
    super().__init__(name='Copier')

  def initial_state(self, batch_size):
    return ()

  def step(self, inputs, prev_state):
    return inputs[1], ()

  def unroll(self, inputs, prev_state):
    return inputs[1], ()

CONSTRUCTORS = dict(
    mlp=MLP,
    frame_stack_mlp=FrameStackingMLP,
    lstm=LSTM,
    gru=GRU,
    copier=Copier,
    res_lstm=DeepResLSTM,
)

DEFAULT_CONFIG = dict(
    name='mlp',
    mlp=MLP.CONFIG,
    frame_stack_mlp=FrameStackingMLP.CONFIG,
    lstm=LSTM.CONFIG,
    gru=GRU.CONFIG,
    copier=Copier.CONFIG,
    res_lstm=DeepResLSTM.CONFIG,
)

def construct_network(name, **config):
  return CONSTRUCTORS[name](**config[name])
