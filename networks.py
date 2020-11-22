import sonnet as snt
import tensorflow as tf

import embed
import utils

# don't use opponent's controller
# our own will be exposed in the input
embed_game = embed.make_game_embedding(
    player_config=dict(with_controller=False))

def process_inputs(inputs):
  gamestate, p1_controller_embed = inputs
  gamestate_embed = embed_game(gamestate)
  return tf.concat([gamestate_embed, p1_controller_embed], -1)

class Network(snt.Module):

  def initial_state(self, batch_size):
    raise NotImplementedError()

  def step(self, inputs, prev_state):
    '''
      Returns outputs and next recurrent state.
      inputs: (batch_size, x_dim)
    '''
    raise NotImplementedError()

  def unroll(self, inputs, initial_state):
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

class LSTM(Network):
  CONFIG=dict(
      hidden_size=128,
      num_mlp_layers=0,
  )

  def __init__(self, hidden_size, num_mlp_layers):
    super().__init__(name='LSTM')
    self._hidden_size = hidden_size
    self._lstm = snt.LSTM(hidden_size)
    if num_mlp_layers:
      self._mlp = snt.nets.MLP(
          [hidden_size] * num_mlp_layers,
          activate_final=True)
    else:
      self._mlp = lambda x: x

  def initial_state(self, batch_size):
    return self._lstm.initial_state(batch_size)

  def step(self, inputs, prev_state):
    flat_inputs = process_inputs(inputs)
    flat_inputs = self._mlp(flat_inputs)
    return self._lstm(flat_inputs, prev_state)

  def unroll(self, inputs, prev_state):
    flat_inputs = process_inputs(inputs)
    flat_inputs = self._mlp(flat_inputs)
    return utils.dynamic_rnn(self._lstm, flat_inputs, prev_state)

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
)

DEFAULT_CONFIG = dict(
    name='mlp',
    mlp=MLP.CONFIG,
    frame_stack_mlp=FrameStackingMLP.CONFIG,
    lstm=LSTM.CONFIG,
    gru=GRU.CONFIG,
    copier=Copier.CONFIG,
)

def construct_network(name, **config):
  return CONSTRUCTORS[name](**config[name])
