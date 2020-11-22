import sonnet as snt
import tensorflow as tf

import embed
import utils

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

class FrameStackingMLP(Network):
  '''
    MLP on frames concatenated through time; e.g.,
      [t-5, t-4, t-3, t-2, t-1] -> predict t

    Optionally, can drop last frame_delay frames to force human reaction time:
      [t-23, t-22, t-20, t-19, t-18] -> predict t
    Humans do not have to react to their own previous inputs, so we suggest coupling
    frame_delay with a residual prediction policy.
  '''

  CONFIG = dict(
      output_sizes=[256, 128],
      dropout_rate=0.,
      num_frames=5,
      frame_delay=0,
  )

  def __init__(self, output_sizes, dropout_rate, num_frames, frame_delay):
    super().__init__(name='FrameStackingMLP')
    self._mlp = snt.nets.MLP(
        output_sizes,
        activate_final=True,
        dropout_rate=dropout_rate)
    self._frame_delay = frame_delay
    self._num_frames = num_frames
    self._frame_buffer_len = num_frames + frame_delay
    self._embed_game = embed.make_game_embedding()

  def initial_state(self, batch_size):
    return [
        tf.zeros([batch_size, self._embed_game.size])
        for _ in range(self._frame_buffer_len-1)
    ]

  def step(self, inputs, prev_state):
    flat_inputs = self._embed_game(inputs)
    frames = prev_state + [flat_inputs]
    trimmed_frames = frames[:len(frames)-self._frame_delay]
    stacked_trimmed_frames = tf.concat(trimmed_frames, -1)
    next_state = frames[1:]
    return self._mlp(stacked_trimmed_frames), next_state

  def unroll(self, inputs, initial_state):
    flat_inputs = self._embed_game(inputs)  # [T, B, ...]

    past_frames = tf.stack(initial_state)  # [N-1, B, ...]
    all_frames = tf.concat([past_frames, flat_inputs], 0)  # [N-1 + T, B, ...]

    n = self._frame_buffer_len - 1

    slices = [all_frames[i:i-n] for i in range(n)]
    slices.append(all_frames[n:])
    # slices has num_frames tensors of dimension [T, B, ...]
    trimmed_slices = slices[:len(slices)-self._frame_delay]
    stacked_trimmed_frames = tf.concat(trimmed_slices, -1)

    final_state = [all_frames[i] for i in range(-n, 0)]

    return self._mlp(stacked_trimmed_frames), final_state

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
    self._embed_game = embed.make_game_embedding()

  def initial_state(self, batch_size):
    return self._lstm.initial_state(batch_size)

  def step(self, inputs, prev_state):
    flat_inputs = self._embed_game(inputs)
    flat_inputs = self._mlp(flat_inputs)
    return self._lstm(flat_inputs, prev_state)

  def unroll(self, inputs, prev_state):
    flat_inputs = self._embed_game(inputs)
    flat_inputs = self._mlp(flat_inputs)
    return utils.dynamic_rnn(self._lstm, flat_inputs, prev_state)

class GRU(Network):
  CONFIG=dict(hidden_size=128)

  def __init__(self, hidden_size):
    super().__init__(name='GRU')
    self._hidden_size = hidden_size
    self._gru = snt.GRU(hidden_size)
    self._embed_game = embed.make_game_embedding()

  def initial_state(self, batch_size):
    return self._gru.initial_state(batch_size)

  def step(self, inputs, prev_state):
    flat_inputs = self._embed_game(inputs)
    return self._gru(flat_inputs, prev_state)

  def unroll(self, inputs, prev_state):
    flat_inputs = self._embed_game(inputs)
    return utils.dynamic_rnn(self._gru, flat_inputs, prev_state)

class Copier(Network):
  '''
    No parameters - simply returns the previous controller state.
  '''
  CONFIG=dict()

  def __init__(self):
    super().__init__(name='Copier')
    self._embed_controller = embed.embed_controller

  def initial_state(self, batch_size):
    return ()

  def step(self, inputs, prev_state):
    outputs = self._embed_controller(inputs['player'][1]['controller_state'])
    return outputs, ()

  def unroll(self, inputs, prev_state):
    outputs = self._embed_controller(inputs['player'][1]['controller_state'])
    return outputs, ()

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
