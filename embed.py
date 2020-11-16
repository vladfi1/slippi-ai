"""
Converts SSBM types to Tensorflow types.
"""

import math
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from melee import enums

float_type = tf.float32

class Embedding:
  def from_state(self, state):
    return self.dtype(state)

  def input_signature(self):
    return tf.TensorSpec((), self.dtype)

  def map(self, f):
    return f(self)

class BoolEmbedding(Embedding):
  size = 1
  dtype = bool

  def __init__(self, name='bool', on=1., off=0.):
    self.name = name
    self.on = on
    self.off = off

  def __call__(self, t):
    return tf.expand_dims(tf.where(t, self.on, self.off), -1)

  def distance(self, predicted, target):
    return tf.nn.sigmoid_cross_entropy_with_logits(
        logits=tf.squeeze(predicted, [-1]),
        labels=tf.cast(target, float_type))

  def sample(self, t):
    t = tf.squeeze(t, -1)
    dist = tfp.distributions.Bernoulli(logits=t, dtype=tf.bool)
    return dist.sample()

embed_bool = BoolEmbedding()

class FloatEmbedding(Embedding):
  dtype = np.float32

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

  def sample(self, t):
    return self.extract(t)

embed_float = FloatEmbedding("float")

class OneHotEmbedding(Embedding):

  def __init__(self, name, size, dtype=np.int64):
    self.name = name
    self.size = size
    self.input_size = size
    self.dtype = dtype

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
    return -tf.reduce_sum(logprobs * target, -1)

class EnumEmbedding(OneHotEmbedding):
  def __init__(self, enum_class, **kwargs):
    super().__init__(str(enum_class), len(enum_class), **kwargs)
    self._class = enum_class
    self._map = {obj: i for i, obj in enumerate(enum_class)}

  def from_state(self, obj):
    return self.dtype(self._map[obj])

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

  def map(self, f):
    return {k: e.map(f) for k, e in self.embedding}

  def from_state(self, state) -> dict:
    struct = {}
    for field, op in self.embedding:
      key = self.key_map.get(field, field)
      struct[field] = op.from_state(self.getter(state, key))
    return struct

  def input_signature(self):
    return {k: e.input_signature() for k, e in self.embedding}

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

  def split(self, embedded):
    fields, ops = zip(*self.embedding)
    sizes = [op.size for op in ops]
    splits = tf.split(embedded, sizes, -1)
    return dict(zip(fields, splits))

  def distance(self, embedded, target):
    distances = {}
    split = self.split(embedded)
    for field, op in self.embedding:
      distances[field] = op.distance(split[field], target[field])
    return distances

  def sample(self, embedded):
    samples = {}
    split = self.split(embedded)
    for field, op in self.embedding:
      samples[field] = op.sample(split[field])
    return samples


class ArrayEmbedding(Embedding):
  def __init__(self, name: str, op: Embedding, permutation: List[int]):
    self.name = name
    self.op = op
    self.permutation = permutation
    self.size = len(permutation) * op.size

  def map(self, f):
    return [self.op.map(f)] * len(self.permutation)

  def from_state(self, state):
    return [self.op.from_state(state[i]) for i in self.permutation]

  def input_signature(self):
    return [self.op.input_signature()] * len(self.permutation)

  def __call__(self, array, **kwargs):
    return tf.concat(
        [self.op(array[i], **kwargs) for i in self.permutation], axis=-1)

  def split(self, embedded):
    return tf.split(embedded, len(self.permutation), -1)

  def extract(self, embedded):
    # a bit suspect here, we can't recreate the original array,
    # only the bits that were embedded. oh well
    array = max(self.permutation) * [None]

    ts = tf.split(axis=tf.rank(embedded)-1, num_or_size_splits=len(self.permutation), value=embedded)

    for i, t in zip(self.permutation, ts):
      array[i] = self.op.extract(t)

    return array

  def distance(self, embedded, target):
    splits = self.split(embedded)
    return [self.op.distance(s, t) for s, t in zip(splits, target)]

  def sample(self, embedded):
    return [self.op.sample(s) for s in self.split(embedded)]

embed_action = EnumEmbedding(enums.Action)
embed_char = EnumEmbedding(enums.Character)

# puff and kirby have 6 jumps
embed_jumps_left = OneHotEmbedding("jumps_left", 6)

def make_player_embedding(
    xy_scale: float = 0.05,
    shield_scale: float = 0.01,
    speed_scale: float = 0.5,
    with_speeds: bool = False,
    ):
    embed_xy = FloatEmbedding("xy", scale=xy_scale)

    embedding = [
      ("percent", FloatEmbedding("percent", scale=0.01)),
      ("facing", BoolEmbedding("facing", off=-1.)),
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
      ('controller_state', embed_controller),
    ]

    if with_speeds:
      embed_speed = FloatEmbedding("speed", scale=speed_scale)
      embedding.extend([
          ('speed_air_x_self', embed_speed),
          ('speed_ground_x_self', embed_speed),
          ('speed_y_self', embed_speed),
          ('speed_x_attack', embed_speed),
          ('speed_y_attack', embed_speed),
      ])

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
