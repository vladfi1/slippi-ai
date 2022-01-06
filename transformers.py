import sonnet as snt
import tensorflow as tf
import math
import numpy as np

def positional_encoding(seq_len, d_model, batch_size=1):
    """
    Returns a tensor following the postional encoding function
     (sinusodal from Vaswani et. all 2017).

    Return shape: [batch, seq_len, d_model]
    """
    def encoding_angle(pos, i):
        pos = tf.cast(pos, tf.dtypes.float32)
        i = tf.cast(i, tf.dtypes.float32)
        d = tf.cast(d_model, tf.dtypes.float32)
        denom = tf.math.pow(10000., 2. * i/d)
        return pos / denom

    i_tensor = tf.expand_dims(tf.range(0, d_model//2), 0) # [1, d_model/2]
    i_tensor = tf.broadcast_to(i_tensor, [seq_len, d_model//2]) # [seq_len, d_model/2]
    j_tensor = tf.expand_dims(tf.range(0, seq_len), 1) # [seq_len, 1]
    j_tensor = tf.broadcast_to(j_tensor, [seq_len, d_model//2]) # [seq_len, d_model/2]
    angles = encoding_angle(j_tensor, i_tensor) # [seq_len, d_model/2]

    # Apply sin to even indices, cos to odd indices
    sins_angle = tf.math.sin(angles) # [seq_len, d_model/2]
    coss_angles = tf.math.cos(angles) # [seq_len, d_model/2]
    joined = tf.concat([sins_angle, coss_angles], -1) # [s, d]

    #Add in batch
    encoding = tf.expand_dims(joined, 0)
    encoding = tf.repeat(encoding, [batch_size], axis=0) # [b, s, d]
    return encoding

def attention(queries, keys, values, masked=True, mem_size=None):
    """
    Returns the 'attention' between three sequences: keys, queries, and values
    Specifically this implementation uses 'scaled dot-product' attention.

    This can be seen as a measure of the compatibility or relative importance between the keys and queries.
    This compatilbility is then applied to the 'input' sequence represented by values.

    Returns a tensor with the same shape as Values where [b, i ,j] represents the relative importance
    "attention" of element j in the sequence.

    When masked is True, only the elements prior to current position in the sequence are considered for compatibility.

    When mem_size is not None, only consider the mem_size prior elements for compatibility. Requires masked to be True

    keys: (batch, seq_len, D_k)
    queries: (batch, seq_len, D_k)
    values: (batch, seq_len, D_v)
    mem_size: default is None, when set to positive integer will only consider mem_size prior elements for computation
    returns: (batch, seq_len, D_v)
    """
    if mem_size is not None:
      # TODO change this assumption (low priority and probably not useful)
      assert masked is True, "In order to use attention memory masked must be set to true"
    tf.debugging.assert_shapes([
        (keys, ('B', 'S', 'D_k')),
        (queries, ('B', 'S', 'D_k')),
        (values, ('B', 'S', 'D_v')),
    ])
    B, S, D_k = tf.unstack(tf.shape(keys))

    # compat [b, i, j] is the dot product of key i and query j (for batch # b)
    compat = tf.matmul(queries, keys, transpose_b=True) # [B, S, S]
    norm_compat = compat / tf.sqrt(tf.cast(D_k, compat.dtype)) # [B, S, S]
    if masked:
      i = tf.expand_dims(tf.range(S), 1)  # [S, 1]
      j = tf.expand_dims(tf.range(S), 0)  # [1, S]
      mask = i >= j # [S, S], mask[i, j] == i >= j
      if mem_size is not None:
        b_mask = i <= (j + mem_size)
        mask = tf.math.logical_and(b_mask, mask)
      norm_compat = tf.where(mask, norm_compat, np.NINF)
    probs = tf.nn.softmax(norm_compat) # [B, S, S]
    att = tf.matmul(probs, values) # [B, S, D_V]
    return att

class MultiHeadAttentionBlock(snt.Module):
  def __init__(self, num_heads, output_size,  mem_size=None, name='MultiHeadAttentionBlock'):
    super(MultiHeadAttentionBlock, self).__init__()
    self.num_heads = num_heads
    self.output_size = output_size
    self.mem_size = mem_size
    self.W_K = []
    self.W_V = []
    self.W_Q = []
    assert output_size%num_heads == 0, "output_size must be a multiple of num_heads"
    projection_size = output_size//num_heads
    for _ in range(num_heads):
        # TODO there's a more efficient way to do this https://github.com/wmcnicho/slippi-ai/pull/2/commits/3354fcb94aeb2c2354352eb893fb454ed2dab9e3
      self.W_K.append(snt.Linear(int(projection_size))) #output is d_model/num_heads
      self.W_V.append(snt.Linear(int(projection_size))) #output is d_model/num_heads
      self.W_Q.append(snt.Linear(int(projection_size))) #output is d_model/num_heads
    self.W_O = snt.Linear(output_size)
    self.l1_embed = snt.Linear(output_size)

  def initial_state(self, batch_size):
    if self.mem_size is None:
      raise NotImplementedError()
    else:
      return tf.zeros([batch_size, self.mem_size, self.output_size])

  def __call__(self, inputs, prev_state):
    """
    For each head, this block will project input into 3 spaces (keys, queries, values)
    and subsequently run an attention block on each projection. The results of each heads are
    combined (via concat) into the final output.

    prev_state: [B, M, D_m]
    inputs: [B, S, D_m]
    returns: (output: [B, S, D_m], next_state: [B, M, D_m])
    """
    B, S, D_m = tf.unstack(tf.shape(inputs))
    if inputs.shape[-1] is not self.output_size:
      # TODO update the unit tests and clear this out
      # In the first layer we need to embed/project the input
      # This is only hit during testing, could be removed
      inputs = self.l1_embed(inputs)
    heads = []
    combined_input = tf.concat([prev_state, inputs], 1) # [B, M+S, D_m]
    for i in range(self.num_heads):
      # head_i <- Attention(QW_i^Q, KW_i^K, VW_i^V)
      # TODO technically we don't need to use queries:  
      # See https://github.com/vladfi1/slippi-ai/commit/bfaefd000279fb187f8e155c74a02f3a547e6154
      queries = self.W_Q[i](combined_input) # [B, M+S, D_m/h]
      keys = self.W_K[i](combined_input) # [B, M+S, D_m/h]
      values = self.W_V[i](combined_input) # [B, M+S, D_m/h]
      head_i = attention(queries, keys, values, mem_size=self.mem_size) # [B, M+S, D_m/h]
      heads.append(head_i)
    # MHA(Q, K, V) <- Concat(head_1...head_h)W^O
    proj_heads = self.W_O(tf.concat(heads, -1))  # [B, M+S, D_m]
    multi_head = proj_heads[:, self.mem_size:] # [B, S, D_m]
    next_state = proj_heads[:,S:] # [B, M, D_m]
    return multi_head, next_state

class TransformerEncoderBlock(snt.Module):
  def __init__(self, output_size, ffw_size, num_heads, mem_size, name="EncoderTransformerBlock"):
      super(TransformerEncoderBlock, self).__init__()
      self.output_size = output_size
      self.ffw_size = ffw_size
      self.mem_size = mem_size
      self.attention = MultiHeadAttentionBlock(num_heads, output_size, mem_size)
      self.feed_forward_in = snt.Linear(ffw_size)
      self.feed_forward_out = snt.Linear(output_size)
      self.norm_1 = snt.LayerNorm(-1, False, False)
      self.norm_2 = snt.LayerNorm(-1, False, False)

  def initial_state(self, batch_size):
      return self.attention.initial_state(batch_size)

  def __call__(self, inputs, prev_state):
    # MHAB
    att, next_state = self.attention(inputs, prev_state)
    # Add (res) + LayerNorm
    res_norm_att = self.norm_1(att + inputs)
    # Feed forward
    feed_in = self.feed_forward_in(res_norm_att)
    act = tf.nn.relu(feed_in)
    feed_out = self.feed_forward_out(act)
    # Add (res) + LayerNorm
    output = self.norm_2(res_norm_att + feed_out)
    return output, next_state

class EncoderOnlyTransformer(snt.Module):
  def __init__(self, output_size, num_layers, ffw_size, num_heads, mem_size, name="EncoderTransformer"):
    super(EncoderOnlyTransformer, self).__init__()
    self.num_layers = num_layers
    self.transformer_blocks = []
    self.mem_size = mem_size
    for _ in range(num_layers):
      t = TransformerEncoderBlock(output_size, ffw_size, num_heads, mem_size)
      self.transformer_blocks.append(t)
    # maybe add assertion about attention size and output_size
    self.shape_convert = snt.Linear(output_size)

  def initial_state(self, batch_size):
    return [t.initial_state(batch_size) for t in self.transformer_blocks]

  def __call__(self, inputs, prev_state):
    """
      inputs: [T, B, O]
      prev_sate: [L, B, M, O]
    """
    inputs = self.shape_convert(inputs) # [T, B, O]
    inputs = tf.transpose(inputs, [1, 0, 2]) # [B, T, O]
    i_shape = tf.shape(inputs)
    # HACKED FOR TESTING UNTIL RELATIVE ENCODING
    #encoding = positional_encoding(i_shape[1], i_shape[2], batch_size=i_shape[0])
    #x = inputs + encoding # [B, T, O]
    x = inputs
    next_state = []
    for t, p in zip(self.transformer_blocks, prev_state):
      x,n = t(x, p) # [B, T, O]
      next_state.append(n)
    x = tf.transpose(x, [1, 0, 2]) # [T, B, O]
    return x, next_state