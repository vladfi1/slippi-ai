"""
Converts SSBM types to Tensorflow types.
"""

from typing import List, Tuple

import tensorflow as tf
import math
from melee import enums

float_type = tf.float32

class Embedding:
  def from_state(self, state):
    return state

class BoolEmbedding(Embedding):
  size = 1

  def __init__(self, on=1., off=0.):
    self.on = on
    self.off = off

  def __call__(self, t):
    return tf.expand_dims(tf.where(t, self.on, self.off), -1)

  def distance(self, predicted, target):
    return tf.nn.sigmoid_cross_entropy_with_logits(
        logits=tf.squeeze(predicted, [-1]),
        labels=tf.cast(target, float_type))

embed_bool = BoolEmbedding()

class FloatEmbedding(Embedding):
  def __init__(self, name, scale=None, bias=None, lower=-10., upper=10.):
    self.name = name
    self.scale = scale
    self.bias = bias
    self.lower = lower
    self.upper = upper
    self.size = 1

  def __call__(self, t, **_):
    if t.dtype is not float_type:
      t = tf.cast(t, float_type)

    if self.bias is not None:
      t += self.bias

    if self.scale is not None:
      t *= self.scale

    if self.lower:
      t = tf.maximum(t, self.lower)

    if self.upper:
      t = tf.minimum(t, self.upper)

    return tf.expand_dims(t, -1)

  def extract(self, t):
    if self.scale:
      t /= self.scale

    if self.bias:
      t -= self.bias

    return tf.squeeze(t, [-1])

  def to_input(self, t):
    return t

  def distance(self, predicted, target):
    if target.dtype is not float_type:
      target = tf.cast(target, float_type)

    if self.scale:
      target *= self.scale

    if self.bias:
      target += self.bias

    predicted = tf.squeeze(predicted, [-1])
    return tf.square(predicted - target)

embed_float = FloatEmbedding("float")

class OneHotEmbedding(Embedding):
  def __init__(self, name, size):
    self.name = name
    self.size = size
    self.input_size = size

  def __call__(self, t, residual=False, **_):
    one_hot = tf.one_hot(t, self.size, 1., 0.)

    if residual:
      logits = math.log(self.size * 10) * one_hot
      return logits
    else:
      return one_hot

  def to_input(self, logits):
    return tf.nn.softmax(logits)

  def extract(self, embedded):
    # TODO: pick a random sample?
    return tf.argmax(t, -1)

  def distance(self, embedded, target):
    logprobs = tf.nn.log_softmax(embedded)
    target = self(target)
    return -tfl.batch_dot(logprobs, target)

class EnumEmbedding(Embedding):
  def __init__(self, enum_class):
    self._class = enum_class
    self.size = len(enum_class)
    self._map = {obj: i for i, obj in enumerate(enum_class)}

  def from_state(self, obj):
    return self._map[obj]

  def __call__(self, t):
    return tf.one_hot(t, self.size, 1., 0.)

class LookupEmbedding(object):

  def __init__(self, name, input_size, output_size):
    self.name = name
    with tf.variable_scope(name):
      self.table = tfl.weight_variable([input_size, output_size], "table")

    self.size = output_size
    self.input_size = input_size

  def __call__(self, indices, **kwargs):
    return tf.nn.embedding_lookup(self.table, indices)

  def to_input(self, input_):
    # FIXME: "to_input" is the wrong name. model.py uses it to go from logits ("residual" embedding, used for prediction) to probabilites ("input" embedding, used for passing through the network). Here we interpret it as going from embedding space (output_size) to probabilities (input_size), which changes the dimensionality and would break model.py.
    logits = tfl.matmul(input_, tf.transpose(self.table))
    return tf.nn.softmax(logits)

class StructEmbedding(Embedding):
  def __init__(self, name: str, embedding: List[Tuple[object, Embedding]],
               is_dict=False, key_map=None):
    self.name = name
    self.embedding = embedding
    if is_dict:
      self.getter = lambda d, k: d[k]
    else:
      self.getter = getattr

    self.key_map = key_map or {}

    self.size = 0
    for _, op in embedding:
      self.size += op.size

  def from_state(self, state) -> dict:
    struct = {}
    for field, op in self.embedding:
      key = self.key_map.get(field, field)
      struct[field] = op.from_state(self.getter(state, key))
    return struct

  def __call__(self, struct: dict, **kwargs):
    embed = []

    rank = None
    for field, op in self.embedding:
      t = op(struct[field], **kwargs)

      if rank is None:
        rank = len(t.get_shape())
      else:
        assert(rank == len(t.get_shape()))

      embed.append(t)
    return tf.concat(axis=rank-1, values=embed)

  def to_input(self, embedded):
    rank = len(embedded.get_shape())
    begin = (rank-1) * [0]
    size = (rank-1) * [-1]

    inputs = []
    offset = 0

    for _, op in self.embedding:
      t = tf.slice(embedded, begin + [offset], size + [op.size])
      inputs.append(op.to_input(t))
      offset += op.size

    return tf.concat(axis=rank-1, values=inputs)

  def extract(self, embedded):
    rank = len(embedded.get_shape())
    begin (rank-1) * [0]
    size = (rank-1) * [-1]

    struct = {}
    offset = 0

    for field, op in self.embedding:
      t = tf.slice(embedded, begin + [offset], size + [op.size])
      struct[field] = op.extract(t)
      offset += op.size

    return struct

  def distance(self, embedded, target):
    rank = len(embedded.get_shape())
    begin = (rank-1) * [0]
    size = (rank-1) * [-1]

    distances = {}
    offset = 0

    for field, op in self.embedding:
      t = tf.slice(embedded, begin + [offset], size + [op.size])
      distances[field] = op.distance(t, target[field])
      offset += op.size

    return distances

class ArrayEmbedding(Embedding):
  def __init__(self, name: str, op: Embedding, permutation: List[int]):
    self.name = name
    self.op = op
    self.permutation = permutation
    self.size = len(permutation) * op.size

  def from_state(self, state):
    return [self.op.from_state(state[i]) for i in self.permutation]

  def __call__(self, array, **kwargs):
    embed = []
    rank = None
    for i in self.permutation:
      with tf.name_scope(str(i)):
        t = self.op(array[i], **kwargs)
        if rank is None:
          rank = len(t.get_shape())
        else:
          assert(rank == len(t.get_shape()))

        embed.append(t)
    return tf.concat(axis=rank-1, values=embed)

  def to_input(self, embedded):
    rank = len(embedded.get_shape())
    ts = tf.split(axis=rank-1, num_or_size_splits=len(self.permutation), value=embedded)
    inputs = list(map(self.op.to_input, ts))
    return tf.concat(axis=rank-1, values=inputs)

  def extract(self, embedded):
    # a bit suspect here, we can't recreate the original array,
    # only the bits that were embedded. oh well
    array = max(self.permutation) * [None]

    ts = tf.split(axis=tf.rank(embedded)-1, num_or_size_splits=len(self.permutation), value=embedded)

    for i, t in zip(self.permutation, ts):
      array[i] = self.op.extract(t)

    return array

  def distance(self, embedded, target):
    distances = []

    ts = tf.split(axis=tf.rank(embedded)-1, num_or_size_splits=len(self.permutation), value=embedded)

    for i, t in zip(self.permutation, ts):
      distances.append(self.op.distance(t, target[i]))

    return distances

embed_action = EnumEmbedding(enums.Action)
embed_char = EnumEmbedding(enums.Character)

# 8 should be enough for puff and kirby
embed_jumps_left = OneHotEmbedding("jumps_left", 8)

def make_player_embedding(
    xy_scale: float = 0.05,
    shield_scale: float = 0.01,
    speed_scale: float = 0.5,
    ):
    embed_xy = FloatEmbedding("xy", scale=xy_scale)
    embed_speed = FloatEmbedding("speed", scale=speed_scale)

    embedding = [
      ("percent", FloatEmbedding("percent", scale=0.01)),
      ("facing", embed_bool),
      ("x", embed_xy),
      ("y", embed_xy),
      ("action", embed_action),
      # ("action_frame", FloatEmbedding("action_frame", scale=0.02)),
      ("character", embed_char),
      ("invulnerable", embed_bool),
      # ("hitlag_frames_left", embedFrame),
      # ("hitstun_frames_left", embedFrame),
      ("jumps_left", embed_jumps_left),
      # ("charging_smash", embedFloat),
      ("shield_strength", FloatEmbedding("shield_size", scale=shield_scale)),
      ("on_ground", embed_bool),
      # ('speed_air_x_self', embed_speed),
      # ('speed_ground_x_self', embed_speed),
      # ('speed_y_self', embed_speed),
      # ('speed_x_attack', embed_speed),
      # ('speed_y_attack', embed_speed),
      ('controller_state', embed_controller),
    ]

    return StructEmbedding("player", embedding)

embed_stage = EnumEmbedding(enums.Stage)

def make_game_embedding(player_config={}, players=(1, 2)):
  embed_player = make_player_embedding(**player_config)
  embed_players = StructEmbedding(
      "players",
      [(i, embed_player) for i in players],
      is_dict=True)

  embedding = [
    ('stage', embed_stage),
    ('player', embed_players),
  ]

  return StructEmbedding("game", embedding)

banned_buttons = (
    enums.Button.BUTTON_MAIN,
    enums.Button.BUTTON_C,
)

embed_buttons = StructEmbedding(
    "buttons",
    [(b.value, embed_bool) for b in enums.Button if b not in banned_buttons],
    is_dict=True,
    key_map={b.value: b for b in enums.Button},
    )

# each controller axis is in [0, 1]
embed_stick = ArrayEmbedding("stick", embed_float, [0, 1])

embed_controller = StructEmbedding("controller", [
    ("button", embed_buttons),
    ("main_stick", embed_stick),
    ("c_stick", embed_stick),
    ("l_shoulder", embed_float),
    ("r_shoulder", embed_float),
])
