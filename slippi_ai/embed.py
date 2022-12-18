"""
Converts SSBM types to Tensorflow types.
"""

import abc
import math
from typing import (
    Any, Callable, Dict, Generic, Iterator, Mapping, NamedTuple, Optional, Sequence,
    Tuple, Type, TypeVar, Union
)

import numpy as np
from gym import spaces
import tensorflow as tf
import tensorflow_probability as tfp

from melee import enums

from slippi_ai.types import Buttons, Controller, Game, Nest, Player, Stick

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

  def flatten(self, struct: Out) -> Iterator[Any]:
    yield struct

  def unflatten(self, seq: Iterator[Any]) -> Out:
    return next(seq)

  def decode(self, out: Out) -> Out:
    return out

  def dummy(self, shape: Sequence[int] = ()) -> Out:
    """A dummy value."""
    return np.zeros(shape, self.dtype)

  def space(self) -> spaces.Space:
    raise NotImplementedError()

  def from_nest(self, nest: Nest[Any]) -> Out:
    return self.dtype(nest)

  def to_nest(self, x: Out) -> Nest:
    return x

# we don't use np.bool_ because it isn't accepted by spaces.Discrete
class BoolEmbedding(Embedding[bool, np.uint8]):
  size = 1
  dtype = np.uint8

  def __init__(self, name='bool', on=1., off=0.):
    self.name = name
    self.on = on
    self.off = off

  def __call__(self, t):
    t = tf.cast(t, tf.bool)
    return tf.expand_dims(tf.where(t, self.on, self.off), -1)

  def distance(self, predicted, target):
    return tf.nn.sigmoid_cross_entropy_with_logits(
        logits=tf.squeeze(predicted, [-1]),
        labels=tf.cast(target, predicted.dtype))

  def sample(self, t, temperature=None):
    t = tf.squeeze(t, -1)
    if temperature is not None:
      t = t / temperature
    dist = tfp.distributions.Bernoulli(logits=t, dtype=tf.bool)
    return dist.sample()

  def space(self) -> spaces.Space:
    return spaces.Discrete(2)

embed_bool = BoolEmbedding()


class FloatEmbedding(Embedding[float, np.float32]):
  dtype = np.float32
  size = 1

  def __init__(self, name, scale=None, bias=None, lower=None, upper=None):
    self.name = name
    self.scale = scale
    self.bias = bias
    self.lower = -np.inf if lower is None else lower
    self.upper = np.inf if upper is None else upper

  def encode(self, t):
    if t.dtype is not float_type:
      t = tf.cast(t, float_type)
    t = tf.clip_by_value(t, self.lower, self.upper)
    if self.bias is not None:
      t += self.bias
    if self.scale is not None:
      t *= self.scale
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

  def space(self):
    return spaces.Box(self.lower, self.upper, shape=(), dtype=self.dtype)

embed_float = FloatEmbedding("float")

class OneHotEmbedding(Embedding[int, np.int32]):

  def __init__(self, name, size, dtype=np.int32):
    self.name = name
    self.size = size
    self.input_size = size
    self.dtype = dtype
    self.tf_dtype = tf.dtypes.as_dtype(dtype)

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
    dist = tfp.distributions.Categorical(logits=logits, dtype=self.tf_dtype)
    return dist.sample()

  def space(self) -> spaces.Space:
    return spaces.Discrete(self.size)

class NullEmbedding(Embedding[In, None]):
  size = 0

  def from_state(self, state: In) -> None:
    return None

  def __call__(self, t):
    assert t is None
    return None

  def map(self, f, *args: None) -> None:
    for x in args:
      assert x is None
    return None

  def flatten(self, none: None):
    pass

  def unflatten(self, seq):
    return None

  def decode(self, out: None) -> None:
    assert out is None
    return None

  def dummy(self) -> None:
    """A dummy value."""
    return None

  def sample(self, embedded, **_):
    return None

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
    # return self.type(*map(lambda e, *xs: e.map(f, *xs), self.embedding, *args))
    result = {k: e.map(f, *(self.getter(x, k) for x in args)) for k, e in self.embedding}
    return self.builder(result)

  def flatten(self, struct: NT):
    for k, e in self.embedding:
      yield from e.flatten(self.getter(struct, k))
    # for e, x in zip(self.embedding, struct):
    #   yield from e.flatten(x)

  def unflatten(self, seq: Iterator[Any]) -> NT:
    return self.builder({k: e.unflatten(seq) for k, e in self.embedding})

  def from_state(self, state: NT) -> NT:
    struct = {k: e.from_state(self.getter(state, k)) for k, e in self.embedding}
    return self.builder(struct)

  def input_signature(self):
    return {k: e.input_signature() for k, e in self.embedding}

  def __call__(self, struct: NT, **kwargs) -> tf.Tensor:
    embed = []

    for field, op in self.embedding:
      t = op(self.getter(struct, field), **kwargs)
      embed.append(t)

    assert embed
    return tf.concat(axis=-1, values=embed)

  def split(self, embedded: tf.Tensor) -> Mapping[str, tf.Tensor]:
    fields, ops = zip(*self.embedding)
    sizes = [op.size for op in ops]
    splits = tf.split(embedded, sizes, -1)
    return dict(zip(fields, splits))

  def distance(self, embedded: tf.Tensor, target: NT) -> NT:
    distances = {}
    split = self.split(embedded)
    for field, op in self.embedding:
      distances[field] = op.distance(split[field], self.getter(target, field))
    return self.builder(distances)

  def sample(self, embedded: tf.Tensor, **kwargs):
    samples = {}
    split = self.split(embedded)
    for field, op in self.embedding:
      samples[field] = op.sample(split[field], **kwargs)
    return self.builder(samples)

  def dummy(self, shape):
    return self.map(lambda e: e.dummy(shape))

  def decode(self, struct: NT) -> NT:
    return self.map(lambda e, x: e.decode(x), struct)

  def space(self) -> spaces.Space:
    return spaces.Dict({k: e.space() for k, e in self.embedding})

  def from_nest(self, nest: Nest) -> NT:
    result = {k: e.from_nest(nest[k]) for k, e in self.embedding}
    return self.builder(result)

  def to_nest(self, nt: NT) -> Nest:
    return {k: e.to_nest(self.getter(nt, k)) for k, e in self.embedding}

T = TypeVar("T")

# use this because lambdas can't be properly pickled :(
class SplatKwargs:
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

def ordered_struct_embedding(
    name: str,
    embedding: Sequence[Tuple[str, Embedding]],
    nt_type: Type[NT],
) -> StructEmbedding[NT]:
  """Supports missing fields, which will appear as None."""
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
    with_controller: bool = True,
) -> StructEmbedding[Player]:
    embed_xy = FloatEmbedding("xy", scale=xy_scale, lower=-300, upper=300)

    embedding = [
      ("percent", FloatEmbedding("percent", scale=0.01, lower=0, upper=999)),
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
      ("shield_strength", FloatEmbedding("shield_size", scale=shield_scale, lower=0, upper=60)),
      ("on_ground", embed_bool),
    ]

    if with_controller:
      # TODO: make this configurable
      embedding.append(('controller', embed_controller_default))

    return ordered_struct_embedding("player", embedding, Player)

# future proof in case we want to play on wacky stages
# embed_stage = EnumEmbedding(enums.Stage, size=64, dtype=np.uint8)
embed_stage = OneHotEmbedding('Stage', size=64, dtype=np.uint8)

_PORTS = (0, 1)
# _PLAYERS = tuple(f'p{p}' for p in _PORTS)
# _SWAP_MAP = dict(zip(_PLAYERS, reversed(_PLAYERS)))

def make_game_embedding(player_config={}):
  embed_player = make_player_embedding(**player_config)

  embedding = Game(
      p0=embed_player,
      p1=embed_player,
      stage=embed_stage,
  )

  return struct_embedding_from_nt("game", embedding)

# don't use opponent's controller
# our own will be exposed in the input
default_embed_game = make_game_embedding(
    player_config=dict(with_controller=False))

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

  def decode(self, a: Union[np.uint8, np.ndarray]):
    assert a.dtype == self.dtype
    return a.astype(np.float32) / self.n

embed_shoulder = DiscreteEmbedding(4)

def get_controller_embedding(
    discrete_axis_spacing: int = 0,
) -> StructEmbedding[Controller]:
  """Controller embedding. Used for autoregressive sampling, so order matters."""
  if discrete_axis_spacing:
    embed_axis = DiscreteEmbedding(discrete_axis_spacing)
  else:
    embed_axis = embed_float

  embed_stick = struct_embedding_from_nt(
      "stick", Stick(x=embed_axis, y=embed_axis))

  return ordered_struct_embedding(
      "controller", [
          ("buttons", embed_buttons),
          ("main_stick", embed_stick),
          ("c_stick", embed_stick),
          ("shoulder", embed_shoulder),
      ], Controller)

embed_controller_default = get_controller_embedding()  # continuous sticks
embed_controller_discrete = get_controller_embedding(16)

# Sadly NamedTuples can't be generic. We could use dataclasses, but TF can't
# trace them.

# Action = TypeVar('Action')
Action = Controller

# @dataclass
class ActionWithRepeat(NamedTuple):
  action: Action
  repeat: np.int32

def get_controller_embedding_with_action_repeat(
    embed_controller: Embedding[Controller, Any],
    max_repeat: int,
) -> StructEmbedding[ActionWithRepeat]:
  embedding = ActionWithRepeat(
      action=embed_controller,
      repeat=OneHotEmbedding('action_repeat', max_repeat+1)
  )
  return struct_embedding_from_nt("controller_with_action_repeat", embedding)

# @dataclass
class StateActionReward(NamedTuple):
  state: Game
  action: ActionWithRepeat
  reward: np.float32

def get_state_action_embedding(
  embed_game: Embedding[Game, Any],
  embed_action: Embedding[Action, Any],
) -> StructEmbedding[StateActionReward]:
  embedding = StateActionReward(
      state=embed_game,
      action=embed_action,
      # ignore incoming reward
      reward=FloatEmbedding('reward', scale=0),
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
