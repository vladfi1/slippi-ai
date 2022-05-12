"""
Converts SSBM types to Tensorflow types.
"""

import abc
import collections
import math
from typing import Generic, List, Tuple, TypeVar

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from melee import enums

from slippi_ai.types import Controller, Nest, Player

float_type = tf.float32
In = TypeVar('In')
Out = TypeVar('Out')

class Embedding(Generic[In, Out], abc.ABC):
  """Embeds game type (In) into tf-ready type Out."""

  def from_state(self, state: In) -> Out:
    """Takes a parsed state and returns"""
    return self.dtype(state)

  @abc.abstractmethod
  def __call__(self, x: Out) -> tf.Tensor:
    """Embed the input state as a flat tensor."""

  def map(self, f, *args: Out) -> Out:
    return f(self, *args)

  def flatten(self, struct):
    yield struct

  def unflatten(self, seq):
    return next(seq)

  # def preprocess(self, x: In):
  #   """Used by discretization."""
  #   return x

  # def dummy(self):
  #   """A dummy value."""
  #   return self.dtype(0)

class BoolEmbedding(Embedding[bool, np.bool_]):
  size = 1
  dtype = np.bool_

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

class FloatEmbedding(Embedding[float, np.float32]):
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
    raise NotImplementedError("Can't sample floats yet.")

embed_float = FloatEmbedding("float")

class OneHotEmbedding(Embedding[int, int]):

  def __init__(self, name, size, dtype=np.int32):
    self.name = name
    self.size = size
    self.input_size = size
    self.dtype = dtype

  def __call__(self, t, residual=False, **_):
    one_hot = tf.one_hot(t, self.size)

    if residual:
      logits = math.log(self.size * 10) * one_hot
      return logits
    else:
      return one_hot

  def to_input(self, logits):
    return tf.nn.softmax(logits)

  def extract(self, embedded):
    # TODO: pick a random sample?
    return tf.argmax(embedded, -1, output_type=self.dtype)

  def distance(self, embedded, target):
    logprobs = tf.nn.log_softmax(embedded)
    target = self(target)
    return -tf.reduce_sum(logprobs * target, -1)

  def sample(self, embedded, temperature=None):
    logits = embedded
    if temperature is not None:
      logits = logits / temperature
    return tfp.distributions.Categorical(logits=logits).sample()

def get_dict(d, k):
  return d[k]

class StructEmbedding(Embedding[In, Nest]):
  def __init__(self, name: str, embedding: List[Tuple[str, Embedding]],
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

  def flatten(self, struct: dict):
    for k, e in self.embedding:
      yield from e.flatten(struct[k])

  def unflatten(self, seq):
    return {k: e.unflatten(seq) for k, e in self.embedding}

  def from_state(self, state: In) -> dict:
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

  def split(self, embedded: tf.Tensor):
    fields, ops = zip(*self.embedding)
    sizes = [op.size for op in ops]
    splits = tf.split(embedded, sizes, -1)
    return dict(zip(fields, splits))

  def distance(self, embedded: tf.Tensor, target: Nest[tf.Tensor]) -> Nest[tf.Tensor]:
    distances = {}
    split = self.split(embedded)
    for field, op in self.embedding:
      distances[field] = op.distance(split[field], target[field])
    return distances

  def sample(self, embedded: tf.Tensor, **kwargs):
    samples = {}
    split = self.split(embedded)
    for field, op in self.embedding:
      samples[field] = op.sample(split[field], **kwargs)
    return samples

  def dummy(self):
    return self.map(lambda e: e.dummy())

# one larger than KIRBY_STONE_UNFORMING
# embed_action = EnumEmbedding(enums.Action, size=0x18F, dtype=np.int16)
embed_action = OneHotEmbedding('Action', size=0x18F, dtype=np.int32)

# one larger than SANDBAG
# embed_char = EnumEmbedding(enums.Character, size=0x21, dtype=np.uint8)
embed_char = OneHotEmbedding('Character', size=0x21, dtype=np.uint8)

# puff and kirby have 6 jumps
embed_jumps_left = OneHotEmbedding("jumps_left", 6, dtype=np.uint8)

def make_player_embedding(
    xy_scale: float = 0.05,
    shield_scale: float = 0.01,
    speed_scale: float = 0.5,
    with_speeds: bool = False,
    with_controller: bool = True,
) -> StructEmbedding[Player, Nest]:
    embed_xy = FloatEmbedding("xy", scale=xy_scale)

    embedding = [
      ("percent", FloatEmbedding("percent", scale=0.01)),
      ("facing", BoolEmbedding("facing", off=-1.)),
      ('x', embed_xy),
      ('y', embed_xy),
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

# future proof in case we want to play on wacky stages
# embed_stage = EnumEmbedding(enums.Stage, size=64, dtype=np.uint8)
embed_stage = OneHotEmbedding('Stage', size=64, dtype=np.uint8)

_PORTS = (0, 1)
_PLAYERS = tuple(f'p{p}' for p in _PORTS)
# _SWAP_MAP = dict(zip(_PLAYERS, reversed(_PLAYERS)))

def make_game_embedding(player_config={}):
  embed_player = make_player_embedding(**player_config)

  embedding = [
    ('stage', embed_stage),
  ] + [(p, embed_player) for p in _PLAYERS]

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
)

class DiscreteEmbedding(OneHotEmbedding):
  """Buckets float inputs in [0, 1]."""

  def __init__(self, n=16):
    super().__init__('DiscreteEmbedding', n+1, dtype=np.uint8)
    self.n = n

  # def sample(self, embedded, **kwargs):
  #   discrete = super().sample(embedded, **kwargs)
  #   return tf.cast(discrete, tf.float32) / self.n

  def from_state(self, a):
    return (a * self.n + 0.5).astype(self.dtype)

  def decode(self, a):
    return np.array(a, np.float32) / self.n

embed_shoulder = DiscreteEmbedding(4)

def get_controller_embedding(
    discrete_axis_spacing: int = 0,
) -> StructEmbedding[Controller, Nest]:
  if discrete_axis_spacing:
    embed_axis = DiscreteEmbedding(discrete_axis_spacing)
  else:
    embed_axis = embed_float

  embed_stick = StructEmbedding(
      "stick", [('x', embed_axis), ('y', embed_axis)])

  return StructEmbedding("controller", [
      ("buttons", embed_buttons),
      ("main_stick", embed_stick),
      ("c_stick", embed_stick),
      ("shoulder", embed_shoulder),
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
