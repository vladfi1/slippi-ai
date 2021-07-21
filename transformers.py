import sonnet as snt
import tensorflow as tf


class MultiHeadAttentionBlock(snt.Module):
  def __init__(self, num_heads, output_size, name='AttentionBlock'):
    super(MultiHeadAttentionBlock, self).__init__()
    self.num_heads = num_heads
    self.W_K = []
    self.W_V = []
    self.W_Q = []
    assert output_size%num_heads == 0, "outputsize must be a multiple of num_heads"
    projection_size = output_size/num_heads
    for _ in range(num_heads):
        # TODO there's a more efficient way to do this
      self.W_K.append(snt.Linear(int(projection_size))) #output is d_model/num_heads
      self.W_V.append(snt.Linear(int(projection_size))) #output is d_model/num_heads
      self.W_Q.append(snt.Linear(int(projection_size))) #output is d_model/num_heads
    self.W_O = snt.Linear(output_size)

  def initial_state(self, batch_size):
    raise NotImplementedError()
  
  def attention(self, queries, keys, values):
    compat = queries * tf.transpose(keys) # check rank
    norm_compat = compat / (keys.shape[-1] ** .05) # check shape
    probs = tf.nn.softmax(norm_compat)
    att = probs * values
    return att

  def __call__(self, queries, keys, values):
    # MHA(Q, K, V) = Concat(head_1...head_h)W^O
    heads = []
    for i in range(self.num_heads):
      # head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
      head_i = self.attention(self.W_Q[i](queries), self.W_K[i](keys), self.W_V[i](values))
      heads.append(head_i)
    multi_head = self.W_O(tf.concat(heads, -1))
    return multi_head