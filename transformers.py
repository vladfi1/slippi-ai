import sonnet as snt
import tensorflow as tf
import math

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

    # BUG: Fails to serialize these calls:
    # "Failed to convert object of type <class 'list'> to Tensor. Contents: [None, 480]. Consider casting elements to a supported type."
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

def attention(queries, keys, values, masked=True):
    """
    Returns the 'attention' between three sequences: keys, queries, and values
    Specifically this implementation uses 'scaled dot-product' attention.

    This can be seen as a measure of the compatibility or relative importance between the keys and queries.
    This compatilbility is then applied to the 'input' sequence represented by values.

    Returns a tensor with the same shape as Values where [b, i ,j] represents the relative importance
    "attention" of element j in the sequence.

    keys: (batch, seq_len, D_k)
    queries: (batch, seq_len, D_k)
    values: (batch, seq_len, D_v)
    returns: (batch, seq_len, D_v)
    """
    assert keys.shape == queries.shape, "keys and values must have equivalent shapes"
    # compat [b, i, j] is the dot product of key i and query j (for batch # b)
    compat = tf.matmul(queries, keys, transpose_b=True) # [B, S, S]
    norm_compat = compat / math.sqrt(keys.shape[-1]) # [B, S, S]
    probs = tf.nn.softmax(norm_compat) # [B, S, S]
    if masked:
      mask = tf.linalg.band_part(tf.ones((compat.shape)), -1, 0) # [B, S, S]
      masked_probs = mask * probs
      probs = masked_probs / tf.reduce_sum(masked_probs, -1, keepdims=True)
    att = tf.matmul(probs, values) # [B, S, D_V]
    return att

class MultiHeadAttentionBlock(snt.Module):
  def __init__(self, num_heads, output_size, name='MultiHeadAttentionBlock'):
    super(MultiHeadAttentionBlock, self).__init__()
    self.num_heads = num_heads
    self.W_K = []
    self.W_V = []
    self.W_Q = []
    assert output_size%num_heads == 0, "output_size must be a multiple of num_heads"
    projection_size = output_size//num_heads
    for _ in range(num_heads):
        # TODO there's a more efficient way to do this
      self.W_K.append(snt.Linear(int(projection_size))) #output is d_model/num_heads
      self.W_V.append(snt.Linear(int(projection_size))) #output is d_model/num_heads
      self.W_Q.append(snt.Linear(int(projection_size))) #output is d_model/num_heads
    self.W_O = snt.Linear(output_size)

  def initial_state(self, batch_size):
    raise NotImplementedError()

  def __call__(self, inputs):
    """
    For each head, this block will project input into 3 spaces (keys, queries, values)
    and subsequently run an attention block on each projection. The results of each heads are
    combined (via concat) into the final output.

    inputs: [B, S, D_m]
    returns: [B, S, D_m]
    """
    # MHA(Q, K, V) = Concat(head_1...head_h)W^O
    heads = []
    for i in range(self.num_heads):
      # head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
      head_i = attention(self.W_Q[i](inputs), self.W_K[i](inputs), self.W_V[i](inputs)) # [B, S, D_m/h]
      heads.append(head_i)
    multi_head = self.W_O(tf.concat(heads, -1))  # [B, S, D_m]
    return multi_head

class TransformerEncoderBlock(snt.Module):
  def __init__(self, output_size, ffw_size, num_heads, name="EncoderTransformerBlock"):
      super(TransformerEncoderBlock, self).__init__()
      self.output_size = output_size
      self.ffw_size = ffw_size
      self.attention = MultiHeadAttentionBlock(num_heads, output_size)
      self.feed_forward_in = snt.Linear(ffw_size)
      self.feed_forward_out = snt.Linear(output_size)
      self.norm_1 = snt.LayerNorm(-1, False, False)
      self.norm_2 = snt.LayerNorm(-1, False, False)

  def initial_state(self, batch_size):
      raise NotImplementedError()

  def __call__(self, inputs):
    # MHAB
    att = self.attention(inputs)
    # Add (res) + LayerNorm
    res_norm_att = self.norm_1(att + inputs)
    # Feed forward
    feed_in = self.feed_forward_in(res_norm_att)
    act = tf.nn.relu(feed_in)
    feed_out = self.feed_forward_out(act)
    # Add (res) + LayerNorm
    output = self.norm_2(res_norm_att + feed_out)
    return output

class EncoderOnlyTransformer(snt.Module):
  def __init__(self, output_size, num_blocks, ffw_size, num_heads, name="EncoderTransformer"):
    super(EncoderOnlyTransformer, self).__init__()
    self.num_blocks = num_blocks
    self.transformer_blocks = []
    for _ in range(num_blocks):
      t = TransformerEncoderBlock(output_size, ffw_size, num_heads)
      self.transformer_blocks.append(t)
    # maybe add assertion about attention size and output_size
    self.shape_convert = snt.Linear(output_size)

  def initial_state(self, batch_size):
    raise NotImplementedError()

  def __call__(self, inputs):
    inputs = self.shape_convert(inputs)
    inputs = tf.transpose(inputs, [1, 0, 2])
    i_shape = inputs.shape
    encoding = positional_encoding(i_shape[1], i_shape[2], batch_size=i_shape[0])
    x = inputs + encoding
    for t in self.transformer_blocks:
      x = t(x)
    x = tf.transpose(x, [1, 0, 2])
    return x