import sonnet as snt
import tensorflow as tf

class Network(snt.Module):

  def initial_state(self, batch_size):
    raise NotImplementedError()

  def step(self, inputs, prev_state):
    """Returns outputs and next recurrent state."""
    raise NotImplementedError()

  def unroll(self, inputs, state, restarting):
    # TODO: make sure this compiles to a dynamic unroll
    outputs = tf.TensorArray(dtype=tf.float32, size=inputs.shape[0])
    i = tf.constant(0)
    for input_ in inputs:
      output, state = self.step(input_, state)
      outputs = outputs.write(i, output)
      i = i + 1
    return outputs.stack(), state

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
    return self._mlp(inputs), ()

  def unroll(self, inputs, state, restarting):
    del state
    del restarting
    return self._mlp(inputs), ()

class FrameStackingMLPNoRestart(Network):
  CONFIG = dict(
    output_sizes=[256, 128],
    dropout_rate=0.0,
    n_frames=5,
  )

  def __init__(self, output_sizes, dropout_rate, n_frames):
    super().__init__(name='FrameStackingMLP')
    self._mlp = snt.nets.MLP(
      output_sizes,
      activate_final=True,
      dropout_rate=dropout_rate)
    self._n_frames = n_frames

  def initial_state(self, batch_size):
    return tf.zeros([batch_size, self._n_frames-1])

  def unroll(self, inputs, state, restarting):
    '''
      Note -- does not implement restarting.
    '''
    n_t, n_batch, n_x = inputs.shape
    n_frames = self._n_frames

    '''
      At init, state has size (n_batch, n_frames-1)
      Once we have used it once, state has size (n_batch, n_frames-1, x_dim)
    '''
    if tf.size(state.shape) == 2:
      state = tf.expand_dims(state, -1)
      state = tf.broadcast_to(state, state.shape[:-1]+[n_x])

    '''
      Expand and reshape input data:
        (num_timepoints, batch_size, x_dim) ->
        (num_timepoints, batch_size, x_dim * num_frames)
      To retain data order, we convert to batch-major, then back to time-major.
    '''
    inp_b_major = tf.transpose(inputs, [1, 0, 2])
    final_state = inp_b_major[:, -(n_frames-1):, :]
    prepend_inputs = tf.concat([state, inp_b_major], axis = 1)
    newdata = []
    for t in range(n_t):
      d = tf.reshape(prepend_inputs[:, t:t+n_frames, :], [n_batch, n_frames*n_x])
      newdata.append(d)
    nd2 = tf.stack(newdata, axis=1)
    stacked_inputs = tf.transpose(nd2, [1, 0, 2])

    return self._mlp(stacked_inputs), final_state

class FrameStackingMLP(Network):
  CONFIG = dict(
    output_sizes=[256, 128],
    dropout_rate=0.0,
    n_frames=5,
  )

  def __init__(self, output_sizes, dropout_rate, n_frames):
    super().__init__(name='FrameStackingMLP')
    self._mlp = snt.nets.MLP(
      output_sizes,
      activate_final=True,
      dropout_rate=dropout_rate)
    self._n_frames = n_frames

  def initial_state(self, batch_size):
    return tf.zeros([batch_size, self._n_frames-1])

  def expand_state(self, state, x_dim):
    '''
      At init, state has size (n_batch, n_frames-1)
      Once we know x_dim, state has size (n_batch, n_frames-1, x_dim)
    '''
    if tf.size(state.shape) == 2:
      state = tf.expand_dims(state, -1)
      state = tf.broadcast_to(state, state.shape[:-1]+[x_dim])
    return state

  def partial_restart_state(self, state, init_state, restart_mask):
    expand = lambda x: tf.expand_dims(x, axis = -1)
    restart_mask = expand(expand(restart_mask))
    return tf.multiply(state, restart_mask)-tf.multiply(init_state, 1-restart_mask)

  def step(self, inputs, state):
    '''
      inputs: (n_batch, x_dim)
      state: (n_batch, x_dim * (n_frames) - 1); flattened
    '''
    return self._mlp(tf.concat([state, inputs], axis=-1)), ()

  def unroll(self, inputs, state, restarting):
    '''
      state: (n_batch, n_frames-1, x_dim)
      flattened state: (n_batch, (n_frames-1)*x_dim)
      inputs: (n_timepoints, n_batch, x_dim)
    '''
    n_t, n_batch, n_x = inputs.shape
    n_frames = self._n_frames

    rs_mat = tf.cast(restarting, tf.float32) 
    
    state = self.expand_state(state, n_x)
    init_state = self.expand_state(self.initial_state(n_batch), n_x)

    outputs = tf.TensorArray(dtype=tf.float32, size=inputs.shape[0])
    i = tf.constant(0)
    for input_ in inputs:
      if tf.reduce_any(restarting[i]):
        state = self.partial_restart_state(state, init_state, rs_mat[i])
      flat_state = tf.reshape(state, [n_batch, n_x*(n_frames-1)])
      output, _ = self.step(input_, flat_state)
      state = tf.concat([state[:, 1:, :], tf.expand_dims(input_, 1)], axis=1)
      outputs = outputs.write(i, output)
      i = i + 1
    return outputs.stack(), state


class FrameStackingMLPNoRestart(Network):
  CONFIG = dict(
    output_sizes=[256, 128],
    dropout_rate=0.0,
    n_frames=5,
  )

  def __init__(self, output_sizes, dropout_rate, n_frames):
    super().__init__(name='FrameStackingMLP')
    self._mlp = snt.nets.MLP(
      output_sizes,
      activate_final=True,
      dropout_rate=dropout_rate)
    self._n_frames = n_frames

  def initial_state(self, batch_size):
    return tf.zeros([batch_size, self._n_frames-1])

  def unroll(self, inputs, state, restarting):
    '''
      Note -- does not implement restarting.
    '''
    n_t, n_batch, n_x = inputs.shape
    n_frames = self._n_frames

    '''
      At init, state has size (n_batch, n_frames-1)
      Once we have used it once, state has size (n_batch, n_frames-1, x_dim)
    '''
    if tf.size(state.shape) == 2:
      state = tf.expand_dims(state, -1)
      state = tf.broadcast_to(state, state.shape[:-1]+[n_x])

    '''
      Expand and reshape input data:
        (num_timepoints, batch_size, x_dim) ->
        (num_timepoints, batch_size, x_dim * num_frames)
      To retain data order, we convert to batch-major, then back to time-major.
    '''
    inp_b_major = tf.transpose(inputs, [1, 0, 2])
    final_state = inp_b_major[:, -(n_frames-1):, :]
    prepend_inputs = tf.concat([state, inp_b_major], axis = 1)
    newdata = []
    for t in range(n_t):
      d = tf.reshape(prepend_inputs[:, t:t+n_frames, :], [n_batch, n_frames*n_x])
      newdata.append(d)
    nd2 = tf.stack(newdata, axis=1)
    stacked_inputs = tf.transpose(nd2, [1, 0, 2])

    return self._mlp(stacked_inputs), final_state

class FrameStackingMLP(Network):
  CONFIG = dict(
    output_sizes=[256, 128],
    dropout_rate=0.0,
    n_frames=5,
  )

  def __init__(self, output_sizes, dropout_rate, n_frames):
    super().__init__(name='FrameStackingMLP')
    self._mlp = snt.nets.MLP(
      output_sizes,
      activate_final=True,
      dropout_rate=dropout_rate)
    self._n_frames = n_frames

  def initial_state(self, batch_size):
    return tf.zeros([batch_size, self._n_frames-1])

  def expand_state(self, state, x_dim):
    '''
      At init, state has size (n_batch, n_frames-1)
      Once we know x_dim, state has size (n_batch, n_frames-1, x_dim)
    '''
    if tf.size(state.shape) == 2:
      state = tf.expand_dims(state, -1)
      state = tf.broadcast_to(state, state.shape[:-1]+[x_dim])
    return state

  def partial_restart_state(self, state, init_state, restart_mask):
    expand = lambda x: tf.expand_dims(x, axis = -1)
    restart_mask = expand(expand(restart_mask))
    return tf.multiply(state, restart_mask)-tf.multiply(init_state, 1-restart_mask)

  def step(self, inputs, state):
    '''
      inputs: (n_batch, x_dim)
      state: (n_batch, x_dim * (n_frames) - 1); flattened
    '''
    return self._mlp(tf.concat([state, inputs], axis=-1)), ()

  def unroll(self, inputs, state, restarting):
    '''
      state: (n_batch, n_frames-1, x_dim)
      flattened state: (n_batch, (n_frames-1)*x_dim)
      inputs: (n_timepoints, n_batch, x_dim)
    '''
    n_t, n_batch, n_x = inputs.shape
    n_frames = self._n_frames

    rs_mat = tf.cast(restarting, tf.float32) 
    
    state = self.expand_state(state, n_x)
    init_state = self.expand_state(self.initial_state(n_batch), n_x)

    outputs = tf.TensorArray(dtype=tf.float32, size=inputs.shape[0])
    i = tf.constant(0)
    for input_ in inputs:
      if tf.reduce_any(restarting[i]):
        state = self.partial_restart_state(state, init_state, rs_mat[i])
      flat_state = tf.reshape(state, [n_batch, n_x*(n_frames-1)])
      output, _ = self.step(input_, flat_state)
      state = tf.concat([state[:, 1:, :], tf.expand_dims(input_, 1)], axis=1)
      outputs = outputs.write(i, output)
      i = i + 1
    return outputs.stack(), state


class LSTM(Network):
  CONFIG=dict(hidden_size=128)

  def __init__(self, hidden_size):
    super().__init__(name='LSTM')
    self._hidden_size = hidden_size
    self._lstm = snt.LSTM(hidden_size)

  def initial_state(self, batch_size):
    return self._lstm.initial_state(batch_size)

  def step(self, inputs, prev_state):
    # Warning: Does not use restarting. Predictions can differ from unroll.
    return self._lstm(inputs, prev_state)

  def partial_restart_state(self, state, init_state, restart_mask):
    from sonnet import LSTMState
    expand = lambda x: tf.expand_dims(x, axis = -1)
    restart_mask = expand(restart_mask)
    return LSTMState(
      hidden = tf.multiply(state.hidden, restart_mask) - \
        tf.multiply(init_state.hidden, 1-restart_mask),
      cell = tf.multiply(state.cell, restart_mask) - \
        tf.multiply(init_state.cell, 1-restart_mask)
    )

  def unroll(self, inputs, state, restarting):
    # TODO: make sure this compiles to a dynamic unroll
    rs_mat = tf.cast(restarting, tf.float32) 
    init_state = self.initial_state(inputs.shape[1])

    outputs = tf.TensorArray(dtype=tf.float32, size=inputs.shape[0])
    i = tf.constant(0)
    for input_ in inputs:
      if tf.reduce_any(restarting[i]):
        state = self.partial_restart_state(state, init_state, rs_mat[i])
      output, state = self.step(input_, state)
      outputs = outputs.write(i, output)
      i = i + 1
    return outputs.stack(), state

class GRU(Network):
  CONFIG=dict(hidden_size=128)

  def __init__(self, hidden_size):
    super().__init__(name='GRU')
    self._hidden_size = hidden_size
    self._gru = snt.GRU(hidden_size)

  def initial_state(self, batch_size):
    return self._gru.initial_state(batch_size)

  def step(self, inputs, prev_state):
    # Warning: Does not use restarting. Predictions can differ from unroll.
    return self._gru(inputs, prev_state)

  def partial_restart_state(self, state, init_state, restart_mask):
    expand = lambda x: tf.expand_dims(x, axis = -1)
    restart_mask = expand(restart_mask)
    keep_mask = expand(keep_mask)
    return tf.multiply(state, restart_mask) - tf.multiply(init_state, 1-restart_mask)

  def unroll(self, inputs, state, restarting):
    # TODO: make sure this compiles to a dynamic unroll
    rs_mat = tf.cast(restarting, tf.float32) 
    init_state = self.initial_state(inputs.shape[1])

    outputs = tf.TensorArray(dtype=tf.float32, size=inputs.shape[0])
    i = tf.constant(0)
    for input_ in inputs:
      if tf.reduce_any(restarting[i]):
        state = self.partial_restart_state(state, init_state, rs_mat[i])
      output, state = self.step(input_, state)
      outputs = outputs.write(i, output)
      i = i + 1
    return outputs.stack(), state

CONSTRUCTORS = dict(
  mlp=MLP,
  lstm=LSTM,
  gru=GRU,
  framestackingmlp=FrameStackingMLP,
  framestackingmlpnorestart=FrameStackingMLPNoRestart,
)

DEFAULT_CONFIG = dict(
  name='mlp',
  mlp=MLP.CONFIG,
  lstm=LSTM.CONFIG,
  gru=GRU.CONFIG,
  framestackingmlp=FrameStackingMLP.CONFIG,
  framestackingmlpnorestart=FrameStackingMLPNoRestart.CONFIG,
)

def construct_network(name, **config):
  return CONSTRUCTORS[name](**config[name])
