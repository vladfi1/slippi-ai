"""
Converts SSBM types to Tensorflow types.
"""

import collections
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

  def map(self, f, *args):
    return f(self, *args)

  def flatten(self, struct):
    yield struct

  def unflatten(self, seq):
    return next(seq)

  def preprocess(self, x):
    """Used by discretization."""
    return x
  
  def dummy(self):
    """A dummy value."""
    return self.dtype(0)

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

  def sample(self, t, temperature=None):
    t = tf.squeeze(t, -1)
    if temperature is not None:
      t = t / temperature
    dist = tfp.distributions.Bernoulli(logits=t, dtype=tf.bool)
    return dist.sample()
  
  def dummy(self):
    return False

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

  def encode(self, t):
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
    return t

  def __call__(self, t, **_):
    return tf.expand_dims(self.encode(t), -1)

  def extract(self, t):
    if self.scale:
      t /= self.scale
    if self.bias:
      t -= self.bias
    return tf.squeeze(t, [-1])

  def to_input(self, t):
    return t

  def distance(self, predicted, target):
    target = self.encode(target)
    predicted = tf.squeeze(predicted, [-1])
    return tf.square(predicted - target)

  def sample(self, t, **_):
    # TODO: make this an actual sample from a Gaussian
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

  def sample(self, embedded, temperature=None):
    logits = embedded
    if temperature is not None:
      logits = logits / temperature
    return tfp.distributions.Categorical(logits=logits).sample()

class EnumEmbedding(OneHotEmbedding):
  def __init__(self, enum_class, **kwargs):
    super().__init__(str(enum_class), len(enum_class), **kwargs)
    self._class = enum_class
    self._map = {obj: i for i, obj in enumerate(enum_class)}

  def from_state(self, obj):
    return self.dtype(self._map[obj])

def get_dict(d, k):
  return d[k]

class StructEmbedding(Embedding):
  def __init__(self, name: str, embedding: List[Tuple[object, Embedding]],
               is_dict=False, key_map=None):
    self.name = name
    self.embedding = embedding
    if is_dict:
      self.getter = get_dict
    else:
      self.getter = getattr

    self.key_map = key_map or {}

    self.size = 0
    for _, op in embedding:
      self.size += op.size

  def map(self, f, *args):
    return collections.OrderedDict(
        (k, e.map(f, *[x[k] for x in args]))
        for k, e in self.embedding)

  def flatten(self, struct):
    for k, e in self.embedding:
      yield from e.flatten(struct[k])

  def unflatten(self, seq):
    return {k: e.unflatten(seq) for k, e in self.embedding}

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

  def sample(self, embedded, **kwargs):
    samples = {}
    split = self.split(embedded)
    for field, op in self.embedding:
      samples[field] = op.sample(split[field], **kwargs)
    return samples

  def dummy(self):
    return self.map(lambda e: e.dummy())

class ArrayEmbedding(Embedding):
  def __init__(self, name: str, op: Embedding, permutation: List[int]):
    self.name = name
    self.op = op
    self.permutation = permutation
    self.size = len(permutation) * op.size

  def map(self, f, *args):
    return [
        self.op.map(f, *[x[i] for x in args])
        for i in range(len(self.permutation))
    ]

  def flatten(self, array):
    assert len(array) == len(self.permutation)
    for a in array:
      yield from self.op.flatten(a)

  def unflatten(self, seq):
    return [self.op.unflatten(seq) for _ in self.permutation]

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

    ts = tf.split(embedded, num_or_size_splits=len(self.permutation), axis=-1)

    for i, t in zip(self.permutation, ts):
      array[i] = self.op.extract(t)

    return array

  def distance(self, embedded, target):
    splits = self.split(embedded)
    return [self.op.distance(s, t) for s, t in zip(splits, target)]

  def sample(self, embedded, **kwargs):
    return [self.op.sample(s, **kwargs) for s in self.split(embedded)]

  def dummy(self):
    return self.map(lambda e: e.dummy())

embed_action = EnumEmbedding(enums.Action)
embed_char = EnumEmbedding(enums.Character)

# puff and kirby have 6 jumps
embed_jumps_left = OneHotEmbedding("jumps_left", 6, dtype=np.uint8)

def make_player_embedding(
    xy_scale: float = 0.05,
    shield_scale: float = 0.01,
    speed_scale: float = 0.5,
    with_speeds: bool = False,
    with_controller: bool = True,
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
    ]

    if with_controller:
      # TODO: make this configurable
      embedding.append(('controller_state', embed_controller_default))

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

def make_game_embedding(player_config={}, players=(1, 2), ports=None):
  embed_player = make_player_embedding(**player_config)
  key_map = dict(zip(players, ports)) if ports else None
  embed_players = StructEmbedding(
      "players",
      [(i, embed_player) for i in players],
      is_dict=True,
      key_map=key_map,
  )

  embedding = [
    ('stage', embed_stage),
    ('player', embed_players),
  ]

  return StructEmbedding("game", embedding)

# Embeddings for controllers

# this is the autoregressive order
LEGAL_BUTTONS = [
    enums.Button.BUTTON_A,
    enums.Button.BUTTON_B,
    enums.Button.BUTTON_X,
    enums.Button.BUTTON_Y,
    enums.Button.BUTTON_Z,
    enums.Button.BUTTON_L,
    enums.Button.BUTTON_R,
    enums.Button.BUTTON_D_UP,
]
embed_buttons = StructEmbedding(
    "buttons",
    [(b.value, BoolEmbedding(name=b.value)) for b in LEGAL_BUTTONS],
    is_dict=True,
    key_map={b.value: b for b in LEGAL_BUTTONS},
)

class DiscreteEmbedding(OneHotEmbedding):
  """Buckets float inputs in [0, 1]."""

  def __init__(self, n=16):
    super().__init__('DiscreteEmbedding', n+1)
    self.n = n

  def maybe_bucket(self, t):
    if t.dtype == tf.float32:
      t = tf.cast(t * self.n + 0.5, tf.int32)
    return t

  def __call__(self, t):
    return super().__call__(self.maybe_bucket(t))

  def distance(self, embedded, target):
    return super().distance(embedded, self.maybe_bucket(target))

  def sample(self, embedded, **kwargs):
    discrete = super().sample(embedded, **kwargs)
    return tf.cast(discrete, tf.float32) / self.n

  def preprocess(self, a):
    return (a * self.n + 0.5).astype(self.dtype)

def get_controller_embedding(discrete_axis_spacing=0):
  if discrete_axis_spacing:
    embed_axis = DiscreteEmbedding(discrete_axis_spacing)
  else:
    embed_axis = embed_float
  embed_stick = ArrayEmbedding("stick", embed_axis, [0, 1])

  return StructEmbedding("controller", [
      ("button", embed_buttons),
      ("main_stick", embed_stick),
      ("c_stick", embed_stick),
      ("l_shoulder", embed_float),
      ("r_shoulder", embed_float),
  ])

embed_controller_default = get_controller_embedding()  # continuous sticks
embed_controller_discrete = get_controller_embedding(16)

def get_controller_embedding_with_action_repeat(embed_controller, max_repeat):
  return StructEmbedding("controller_with_action_repeat", [
      ("controller", embed_controller),
      ("action_repeat", OneHotEmbedding('action_repeat', max_repeat+1)),
  ])

def _stick_to_str(stick):
  return f'({stick[0].item():.2f}, {stick[1].item():.2f})'

def controller_to_str(controller):
  """Pretty-prints a sampled controller."""
  buttons = [b.value for b in LEGAL_BUTTONS if controller['button'][b.value].item()]

  components = [
      f'Main={_stick_to_str(controller["main_stick"])}',
      f'C={_stick_to_str(controller["c_stick"])}',
      ' '.join(buttons),
      f'LS={controller["l_shoulder"].item():.2f}',
      f'RS={controller["r_shoulder"].item():.2f}',
  ]

  return ' '.join(components)
