"""
Converts SSBM types to Tensorflow types.
"""

import abc
import dataclasses
import math
from typing import (
    Any, Callable, Dict, Generic, Iterator, Mapping, NamedTuple, Optional, Sequence,
    Tuple, Type, TypeVar, Union
)

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from slippi_ai import utils
from slippi_ai.types import Buttons, Controller, Game, Nest, Player, Stick, Randall
from slippi_ai.controller_lib import LEGAL_BUTTONS
from slippi_ai.data import Action, StateAction

float_type = tf.float32
In = TypeVar('In')
Out = TypeVar('Out')

class Embedding(Generic[In, Out], abc.ABC):
  """Embeds game type (In) into tf-ready type Out."""

  def from_state(self, state: In) -> Out:
    """Encodes a parsed state."""
    return self.dtype(state)

  @abc.abstractmethod
  def __call__(self, x: Out) -> tf.Tensor:
    """Embed the input state as a flat tensor."""

  def map(self, f, *args: Out) -> Out:
    return f(self, *args)

  def flatten(self, struct: Out) -> Iterator[Any]:
    yield struct

  def unflatten(self, seq: Iterator[Any]) -> Out:
    return next(seq)

  def decode(self, out: Out) -> In:
    """Inverse of `from_state`."""
    return out

  # def preprocess(self, x: In):
  #   """Used by discretization."""
  #   return x

  def dummy(self, shape: Sequence[int] = ()) -> Out:
    """A dummy value."""
    return np.zeros(shape, self.dtype)

  def dummy_embedding(self, shape: Sequence[int] = ()):
    return np.zeros(shape + [self.size], np.float32)

  def sample(self, embedded: tf.Tensor, **kwargs) -> Out:
    raise NotImplementedError

  def distance(self, embedded: tf.Tensor, target: Out) -> Out:
    """Negative log-prob of the target sample."""
    raise NotImplementedError

  def distribution(self, embedded: tf.Tensor):
    raise NotImplementedError


class BoolEmbedding(Embedding[bool, np.bool_]):
  size = 1
  dtype = np.bool_

  def __init__(self, name='bool', on=1., off=0.):
    self.name = name
    self.on = on
    self.off = off

  def __call__(self, t):
    return tf.expand_dims(tf.where(t, self.on, self.off), -1)

  def distance(self, embedded: tf.Tensor, target: bool):
    logits = tf.squeeze(embedded, [-1])
    labels = tf.cast(target, float_type)

    common_shape = tf.broadcast_static_shape(logits.shape, labels.shape)
    logits = tf.broadcast_to(logits, common_shape)
    labels = tf.broadcast_to(labels, common_shape)

    return tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=labels)

  def sample(self, embedded: tf.Tensor, temperature=None):
    logits = tf.squeeze(embedded, -1)
    if temperature is not None:
      logits = logits / temperature
    dist = tfp.distributions.Bernoulli(logits=logits, dtype=tf.bool)
    return dist.sample()

  def distribution(self, embedded: tf.Tensor):
    logits = tf.squeeze(embedded, -1)
    return tfp.distributions.Bernoulli(logits=logits, dtype=tf.bool)

embed_bool = BoolEmbedding()

class FloatEmbedding(Embedding[float, np.float32]):
  dtype = np.float32
  size = 1

  def __init__(self, name, scale=None, bias=None, lower=-10., upper=10.):
    self.name = name
    self.scale = scale
    self.bias = bias
    self.lower = lower
    self.upper = upper

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

class OneHotEmbedding(Embedding[int, np.int32]):

  def __init__(self, name, size, dtype=np.int32):
    self.name = name
    self.size = size
    self.input_size = size
    self.dtype = dtype
    self.tf_dtype = tf.dtypes.as_dtype(dtype)

  def __call__(self, t: tf.Tensor, residual=False, **_):
    if t.dtype != self.tf_dtype:
      raise ValueError(f"Expected {self.tf_dtype}, got {t.dtype}.")

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
    dist = tfp.distributions.Categorical(logits=logits, dtype=self.tf_dtype)
    return dist.sample()

  def distribution(self, embedded: tf.Tensor):
    return tfp.distributions.Categorical(logits=embedded, dtype=self.tf_dtype)

NT = TypeVar("NT")

class StructEmbedding(Embedding[NT, NT]):
  """Embeds structures: dictionaries or NamedTuples or dataclasses.

  Sub-embeddings are a subset of the keys/fields in the input type.
  The order of sub-embeddings determines the order of traversal, which
  is important for autoregressive sampling.
  """
  def __init__(
      self,
      name: str,
      embedding: Sequence[Tuple[str, Embedding]],
      builder: Callable[[Mapping[str, Any]], NT],
      getter: Callable[[NT, str], Any],
  ):
    self.name = name
    self.embedding = embedding
    self.builder = builder
    self.getter = getter

    self.size = 0
    for _, op in embedding:
      self.size += op.size

  def map(self, f, *args: NT) -> NT:
    result = {
        k: e.map(f, *(self.getter(x, k) for x in args))
        for k, e in self.embedding}
    return self.builder(result)

  def flatten(self, struct: NT):
    for k, e in self.embedding:
      yield from e.flatten(self.getter(struct, k))

  def unflatten(self, seq: Iterator[Any]) -> NT:
    return self.builder({k: e.unflatten(seq) for k, e in self.embedding})

  def from_state(self, state: NT) -> NT:
    struct = {k: e.from_state(self.getter(state, k)) for k, e in self.embedding}
    return self.builder(struct)

  def __call__(self, struct: NT, **kwargs) -> tf.Tensor:
    embed = []

    for field, op in self.embedding:
      if op.size == 0:
        continue

      t = op(self.getter(struct, field), **kwargs)
      embed.append(t)

    assert embed, "Embedding must not be empty"
    return tf.concat(axis=-1, values=embed)

  # def split(self, embedded: tf.Tensor) -> Mapping[str, tf.Tensor]:
  #   fields, ops = zip(*self.embedding)
  #   sizes = [op.size for op in ops]
  #   splits = tf.split(embedded, sizes, -1)
  #   return dict(zip(fields, splits))

  # def distance(self, embedded: tf.Tensor, target: NT) -> NT:
  #   distances = {}
  #   split = self.split(embedded)
  #   for field, op in self.embedding:
  #     distances[field] = op.distance(split[field], self.getter(target, field))
  #   return self.builder(distances)

  # def sample(self, embedded: tf.Tensor, **kwargs):
  #   """Samples sub-components independently."""
  #   samples = {}
  #   split = self.split(embedded)
  #   for field, op in self.embedding:
  #     samples[field] = op.sample(split[field], **kwargs)
  #   return self.builder(samples)

  def dummy(self, shape):
    return self.map(lambda e: e.dummy(shape))

  def dummy_embedding(self, shape):
    return self.map(lambda e: e.dummy_embedding(shape))

  def decode(self, struct: NT) -> NT:
    return self.map(lambda e, x: e.decode(x), struct)

T = TypeVar("T")

# use this because lambdas can't be properly pickled :(
class SplatKwargs(Generic[T]):
  """Wraps a function that takes kwargs."""

  def __init__(self, f: Callable[..., T], fixed_kwargs: Mapping[str, Any] = {}):
    self._func = f
    self._fixed_kwargs = fixed_kwargs

  def __call__(self, kwargs: Mapping[str, Any]) -> T:
    return self._func(**kwargs, **self._fixed_kwargs)

def struct_embedding_from_nt(name: str, nt: NT) -> StructEmbedding[NT]:
  return StructEmbedding(
      name=name,
      embedding=list(zip(nt._fields, nt)),
      builder=SplatKwargs(type(nt)),
      getter=getattr,
  )

# annoyingly, type inference doesn't work here
def ordered_struct_embedding(
    name: str,
    embedding: Sequence[Tuple[str, Embedding]],
    nt_type: Type[NT],
) -> StructEmbedding[NT]:
  """Supports missing fields, which will appear as ()."""
  existing_fields = set(k for k, _ in embedding)
  missing_fields = set(nt_type._fields) - existing_fields
  missing_kwargs = {k: () for k in missing_fields}

  return StructEmbedding(
      name=name,
      embedding=embedding,
      builder=SplatKwargs(nt_type, missing_kwargs),
      getter=getattr,
  )

K = TypeVar("K")
V = TypeVar("V")

def get_dict(d: Mapping[K, V], k: K) -> V:
  return d[k]

id_fn = lambda x: x

def dict_embedding(
    name: str,
    embedding: Sequence[Tuple[str, Embedding]],
) -> StructEmbedding[Dict[str, Any]]:
  return StructEmbedding(
      name=name,
      embedding=embedding,
      builder=id_fn,
      getter=get_dict,
  )

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
) -> StructEmbedding[Player]:
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
    embed_controller_default = get_controller_embedding()  # continuous sticks
    embedding.append(('controller', embed_controller_default))

  if with_speeds:
    embed_speed = FloatEmbedding("speed", scale=speed_scale)
    embedding.extend([
        ('speed_air_x_self', embed_speed),
        ('speed_ground_x_self', embed_speed),
        ('speed_y_self', embed_speed),
        ('speed_x_attack', embed_speed),
        ('speed_y_attack', embed_speed),
    ])

  return ordered_struct_embedding("player", embedding, Player)

@dataclasses.dataclass
class PlayerConfig:
  xy_scale: float = 0.05
  shield_scale: float = 0.01
  speed_scale: float = 0.5
  with_speeds: bool = False
  # don't use opponent's controller
  # our own will be embedded separately
  with_controller: bool = False

default_player_config = PlayerConfig()

# future proof in case we want to play on wacky stages
# embed_stage = EnumEmbedding(enums.Stage, size=64, dtype=np.uint8)
embed_stage = OneHotEmbedding('Stage', size=64, dtype=np.uint8)

_PORTS = (0, 1)
# _PLAYERS = tuple(f'p{p}' for p in _PORTS)
# _SWAP_MAP = dict(zip(_PLAYERS, reversed(_PLAYERS)))

def make_game_embedding(
    with_randall: bool = False,
    player_config: dict = dataclasses.asdict(default_player_config),
):
  embed_player = make_player_embedding(**player_config)

  if with_randall:
    embed_xy = FloatEmbedding("randall_xy", scale=player_config['xy_scale'])
    embed_randall = struct_embedding_from_nt(
        "randall", Randall(x=embed_xy, y=embed_xy))
  else:
    # Older agents don't use Randall
    embed_randall = ordered_struct_embedding(
        "randall", [], Randall)
    assert embed_randall.size == 0

  embedding = Game(
      p0=embed_player,
      p1=embed_player,
      stage=embed_stage,
      randall=embed_randall,
  )

  return struct_embedding_from_nt("game", embedding)

# Embeddings for controllers
embed_buttons = ordered_struct_embedding(
    'buttons',
    [(b.value, BoolEmbedding(name=b.value)) for b in LEGAL_BUTTONS],
    Buttons,
)

class DiscreteEmbedding(OneHotEmbedding):
  """Buckets float inputs in [0, 1]."""

  def __init__(self, n=16):
    super().__init__('DiscreteEmbedding', n+1, dtype=np.uint8)
    self.n = n

  # def sample(self, embedded, **kwargs):
  #   discrete = super().sample(embedded, **kwargs)
  #   return tf.cast(discrete, tf.float32) / self.n

  def from_state(self, a: Union[np.float32, np.ndarray]):
    assert a.dtype == np.float32
    return (a * self.n + 0.5).astype(self.dtype)

  def decode(self, a: Union[np.uint8, np.ndarray]) -> Union[np.float32, np.ndarray]:
    assert a.dtype == self.dtype
    return (a / self.n).astype(np.float32)

NATIVE_AXIS_SPACING = 160
NATIVE_SHOULDER_SPACING = 140

def get_controller_embedding(
    axis_spacing: int = 0,
    shoulder_spacing: int = 4,
) -> StructEmbedding[Controller]:
  """Controller embedding. Used for autoregressive sampling, so order matters."""
  if axis_spacing:
    if NATIVE_AXIS_SPACING % axis_spacing != 0:
      raise ValueError(
        f'Axis spacing must divide {NATIVE_AXIS_SPACING}, got {axis_spacing}.')
    embed_axis = DiscreteEmbedding(axis_spacing)
  else:
    embed_axis = embed_float

  embed_stick = struct_embedding_from_nt(
      "stick", Stick(x=embed_axis, y=embed_axis))

  if NATIVE_SHOULDER_SPACING % shoulder_spacing != 0:
    raise ValueError(
      f'Shoulder spacing must divide {NATIVE_SHOULDER_SPACING}, got {shoulder_spacing}.')
  embed_shoulder = DiscreteEmbedding(shoulder_spacing)

  return ordered_struct_embedding(
      "controller", [
          ("buttons", embed_buttons),
          ("main_stick", embed_stick),
          ("c_stick", embed_stick),
          ("shoulder", embed_shoulder),
      ], Controller)

@dataclasses.dataclass
class ControllerConfig:
  axis_spacing: int = 16
  shoulder_spacing: int = 4

@dataclasses.dataclass
class EmbedConfig:
  player: PlayerConfig = utils.field(PlayerConfig)
  controller: ControllerConfig = utils.field(ControllerConfig)
  with_randall: bool = False

NAME_DTYPE = np.int32

def get_state_action_embedding(
  embed_game: Embedding[Game, Any],
  embed_action: Embedding[Action, Any],
  num_names: int,
) -> StructEmbedding[StateAction]:
  embedding = StateAction(
      state=embed_game,
      action=embed_action,
      name=OneHotEmbedding('name', num_names, dtype=NAME_DTYPE),
  )
  return struct_embedding_from_nt("state_action", embedding)

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
