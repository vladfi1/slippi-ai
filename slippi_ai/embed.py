"""
Converts SSBM types to Tensorflow types.
"""

import abc
import dataclasses
import enum
import math
import typing as tp
from typing import (
    Any, Callable, Dict, Generic, Iterator, Mapping, Sequence,
    Tuple, Type, TypeVar, Union
)

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt

from slippi_ai import utils, types
from slippi_ai.types import (
  Buttons, Controller, Game, Player, Stick,
  Randall, FoDPlatforms, Item, Items,
)
from slippi_ai.controller_lib import LEGAL_BUTTONS
from slippi_ai.data import Action, StateAction

float_type = tf.float32
In = TypeVar('In')
Out = TypeVar('Out')

class Embedding(Generic[In, Out], abc.ABC, snt.Module):
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
    super().__init__(name=name)
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

class BitwiseEmbedding(Embedding[int, tuple[bool, ...]]):

  def __init__(
      self,
      num_bits: int
  ):
    super().__init__(name=f'bitwise_{num_bits}')
    self.size = num_bits

  def from_state(self, state: np.ndarray):
    return np.stack([
        np.bitwise_and(np.bitwise_right_shift(state, i), 1).astype(np.bool_)
        for i in range(self.size)
    ], axis=-1)

  def __call__(self, x: tf.Tensor) -> tf.Tensor:
    return tf.cast(x, tf.float32)

embed_byte = BitwiseEmbedding(8)

class FloatEmbedding(Embedding[float, np.float32]):
  dtype = np.float32
  size = 1

  def __init__(self, name, scale=None, bias=None, lower=-10., upper=10.):
    super().__init__(name=name)
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

class OneHotPolicy(enum.Enum):
  CLAMP = 0  # Clamp to 0 or size-1
  ERROR = 1  # Raise an error for invalid inputs
  EXTRA = 2  # Add an extra dimension for invalid inputs
  EMPTY = 3  # Treat invalid inputs as empty (all off)

T = tp.TypeVar("T")

class OneHotEmbedding(Embedding[int, T]):
  """Embeds integers in the range [0, size) as one-hot vectors."""

  def __init__(
      self, name: str, size: int,
      dtype: type[T] = np.int32,
      one_hot_policy: OneHotPolicy = OneHotPolicy.ERROR,
  ):
    super().__init__(name=name)
    self.one_hot_policy = one_hot_policy
    self.size = size
    if one_hot_policy is OneHotPolicy.EXTRA:
      self.size += 1
    self.input_size = size
    self.dtype = dtype
    self.tf_dtype = tf.dtypes.as_dtype(dtype)

  def from_state(self, state: np.ndarray):
    if self.one_hot_policy is OneHotPolicy.CLAMP:
      state = np.clip(state, 0, self.input_size - 1)
    elif self.one_hot_policy is OneHotPolicy.ERROR:
      if np.any(state < 0):
        raise ValueError(f"Got negative input in {self.name}")
      if np.any(state >= self.input_size):
        x = np.max(state)
        raise ValueError(f"Got invalid input {x} >= {self.input_size} in {self.name}")
    elif self.one_hot_policy is OneHotPolicy.EXTRA:
      invalid = (state < 0) | (state >= self.input_size)
      if np.any(invalid):
        state = state.copy()
        state[invalid] = self.input_size

    # For EMPTY, tf.one_hot already does the right thing of setting everything
    # to 0 for invalid inputs, so we don't need to do anything here.

    return state.astype(self.dtype)


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
    super().__init__(name=name)
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

class MLPWrapper(Embedding[In, Out]):

  def __init__(
      self,
      output_sizes: Sequence[int],
      embed: Embedding[In, Out],
  ):
    super().__init__(name=f'MLP_{embed.name}')
    self._output_sizes = output_sizes
    self.size = output_sizes[-1]
    self._embed = embed

    self._mlp = snt.nets.MLP(
        output_sizes, activate_final=True,
        activation=tf.nn.relu,
    )

  def from_state(self, state: In) -> Out:
    return self._embed.from_state(state)

  def __call__(self, inputs: Out) -> tf.Tensor:
    embedded = self._embed(inputs)
    return self._mlp(embedded)

  def dummy(self, shape: Sequence[int] = ()) -> Out:
    return self._embed.dummy(shape)

# Note: some Kirby ability-copy actions states go beyond this.
embed_action = OneHotEmbedding(
    'Action', size=0x18F, dtype=np.int32,
    one_hot_policy=OneHotPolicy.CLAMP)

# one larger than SANDBAG
# embed_char = EnumEmbedding(enums.Character, size=0x21, dtype=np.uint8)
embed_char = OneHotEmbedding('Character', size=0x21, dtype=np.uint8)

# puff and kirby have 6 jumps
legacy_embed_jumps_left = OneHotEmbedding(
    "jumps_left", 6, dtype=np.uint8, one_hot_policy=OneHotPolicy.EMPTY)
# we initially forgot to add 1 to the max jumps left, fixed Sep 2025
embed_jumps_left = OneHotEmbedding("jumps_left", 7, dtype=np.uint8)

def _base_player_embedding(
    xy_scale: float = 0.05,
    shield_scale: float = 0.01,
    speed_scale: float = 0.5,
    with_speeds: bool = False,
    legacy_jumps_left: bool = False,
) -> list[tuple[str, Embedding]]:
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
      ("jumps_left", legacy_embed_jumps_left if legacy_jumps_left else embed_jumps_left),
      # ("charging_smash", embedFloat),
      ("shield_strength", FloatEmbedding("shield_size", scale=shield_scale)),
      ("on_ground", embed_bool),
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

  return embedding

def make_player_embedding(
    xy_scale: float = 0.05,
    shield_scale: float = 0.01,
    speed_scale: float = 0.5,
    with_speeds: bool = False,
    with_controller: bool = False,
    with_nana: bool = True,
    legacy_jumps_left: bool = False,
) -> StructEmbedding[Player]:
  embedding = _base_player_embedding(
      xy_scale=xy_scale,
      shield_scale=shield_scale,
      speed_scale=speed_scale,
      with_speeds=with_speeds,
      legacy_jumps_left=legacy_jumps_left,
  )

  if with_nana:
    nana_embedding = embedding.copy()
    nana_embedding.append(('exists', embed_bool))
    embed_nana = ordered_struct_embedding(
        "nana", nana_embedding, types.Nana)
    embedding.append(('nana', embed_nana))

  if with_controller:
    # TODO: make this configurable
    embed_controller_default = get_controller_embedding()  # continuous sticks
    embedding.append(('controller', embed_controller_default))

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
  with_nana: bool = True
  legacy_jumps_left: bool = False

default_player_config = PlayerConfig()

# future proof in case we want to play on wacky stages
# embed_stage = EnumEmbedding(enums.Stage, size=64, dtype=np.uint8)
embed_stage = OneHotEmbedding('Stage', size=64, dtype=np.uint8)

# https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit?gid=20#gid=20
MAX_ITEM_TYPE = 0xEC
embed_item_type = OneHotEmbedding(
    'ItemType', size=MAX_ITEM_TYPE + 1, dtype=np.int32,
    one_hot_policy=OneHotPolicy.EXTRA)

MAX_ITEM_STATE = 11  # empirically determined from a ~1K sample
embed_item_state = OneHotEmbedding(
    'ItemState', size=MAX_ITEM_STATE + 1, dtype=np.uint8,
    one_hot_policy=OneHotPolicy.EXTRA)

def make_item_embedding(xy_scale: float) -> StructEmbedding[Item]:
  embed_xy = FloatEmbedding("xy", scale=xy_scale)
  return struct_embedding_from_nt("Item", Item(
      exists=embed_bool,
      type=embed_item_type,
      state=embed_item_state,
      x=embed_xy, y=embed_xy,
  ))

class ItemsType(enum.Enum):
  SKIP = 'skip'
  FLAT = 'flat'
  MLP = 'mlp'

@dataclasses.dataclass
class ItemsConfig:
  type: ItemsType = ItemsType.MLP
  mlp_sizes: tuple[int, ...] = (128, 32)

def make_items_embedding(
    items_config: ItemsConfig,
    xy_scale: float,
) -> Embedding[Items, tp.Any]:
  if items_config.type is ItemsType.SKIP:
    return ordered_struct_embedding("items", [], Items)

  embed_item_flat = make_item_embedding(xy_scale)

  if items_config.type is ItemsType.FLAT:
    embed_item = embed_item_flat
  elif items_config.type is ItemsType.MLP:
    embed_item = MLPWrapper(
        output_sizes=items_config.mlp_sizes,
        embed=embed_item_flat,
    )
  else:
    raise ValueError(f"Unsupported items config type: {items_config.type}")

  # TODO: Can bulk-embed these more efficiently
  return ordered_struct_embedding("items", [
      (field, embed_item) for field in Items._fields
  ], Items)

def make_game_embedding(
    with_randall: bool = True,
    with_fod: bool = True,
    items_config: ItemsConfig = ItemsConfig(),
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

  if with_fod:
    embed_height = FloatEmbedding("fod_height", scale=player_config['xy_scale'])
    embed_fod = struct_embedding_from_nt(
        "fod", FoDPlatforms(left=embed_height, right=embed_height))
  else:
    # Older agents don't use FoD
    embed_fod = ordered_struct_embedding(
        "fod", [], FoDPlatforms)
    assert embed_fod.size == 0

  embed_items = make_items_embedding(
      items_config=items_config,
      xy_scale=player_config['xy_scale'])

  embedding = Game(
      p0=embed_player,
      p1=embed_player,
      stage=embed_stage,
      randall=embed_randall,
      fod_platforms=embed_fod,
      items=embed_items,
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
  with_randall: bool = True
  with_fod: bool = True
  items: ItemsConfig = utils.field(ItemsConfig)

NAME_DTYPE = np.int32

def get_state_action_embedding(
  embed_game: Embedding[Game, Any],
  embed_action: Embedding[Action, Any],
  num_names: int,
) -> StructEmbedding[StateAction]:
  embedding = StateAction(
      state=embed_game,
      action=embed_action,
      name=OneHotEmbedding(
          'name', num_names,
          dtype=NAME_DTYPE,
          one_hot_policy=OneHotPolicy.EMPTY),
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
