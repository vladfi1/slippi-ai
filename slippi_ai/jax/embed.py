"""
Converts SSBM types to JAX types.
"""

import abc
import dataclasses
import enum
import math
import typing as tp
from typing import (
    Any, Callable, Generic, Iterator, Mapping, Sequence,
    Tuple, Type, TypeVar, Union
)

import jax
import jax.numpy as jnp
import optax
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from slippi_ai import utils, types
from slippi_ai.types import (
  Buttons, Controller, Game, Player, Stick,
  Randall, FoDPlatforms, Item, Items,
)
from slippi_ai.controller_lib import LEGAL_BUTTONS
from slippi_ai.data import Action, StateAction
from slippi_ai.action_space import custom_v1 as cv1

Array = jax.Array
In = TypeVar('In')
Out = TypeVar('Out')
Inputs = tp.TypeVarTuple('Inputs')

# TODO: distinguish between leaf and composite embeddings
class Embedding(Generic[In, Out], abc.ABC):
  """Embeds game type (In) into jax-ready type Out."""

  @property
  @abc.abstractmethod
  def dtype(self) -> type:
    """The numpy dtype of the input/output."""

  @property
  @abc.abstractmethod
  def size(self) -> int:
    """The size of the embedded output."""

  def from_state(self, state: In) -> Out:
    """Encodes a parsed state."""
    return self.dtype(state)

  @abc.abstractmethod
  def __call__(self, x: Out) -> Array:
    """Embed the input state as a flat tensor."""

  def map(self, f: tp.Callable[[tp.Self, *Inputs], Out], *args: Out) -> Out:
    return f(self, *args)

  def flatten(self, struct: Out) -> Iterator[Any]:
    yield struct

  def unflatten(self, seq: Iterator[Any]) -> Out:
    return next(seq)

  def decode(self, out: Out) -> In:
    """Inverse of `from_state`."""
    return out

  def dummy(self, shape: Sequence[int] = ()) -> Out:
    """A dummy value."""
    return np.zeros(shape, self.dtype)

  def dummy_embedding(self, shape: Sequence[int] = ()) -> Out:
    return np.zeros(list(shape) + [self.size], np.float32)

  def distribution_size(self) -> int:
    """The size of the distribution parameters."""
    return self.size

  def sample(self, rng: jax.Array, distribution: Array, **kwargs) -> Out:
    """Sample from the distribution."""
    raise NotImplementedError

  def distance(self, distribution: Array, target: Out) -> Out:
    """Negative log-prob of the target sample."""
    raise NotImplementedError

  def distribution(self, distribution: Array) -> tfp.distributions.Distribution:
    raise NotImplementedError(
        f'{type(self).__name__} does not support distribution')

  def kl_divergence(self, logits_p: Array, logits_q: Array) -> Array:
    return self.distribution(logits_p).kl_divergence(self.distribution(logits_q))

  def entropy(self, logits: Array) -> Array:
    return self.distribution(logits).entropy()

class BoolEmbedding(Embedding[np.bool, np.bool]):

  def __init__(self, name='bool', on=1., off=0.):
    self.name = name
    self.on = on
    self.off = off

  @property
  def dtype(self):
    return np.bool

  @property
  def size(self):
    return 1

  def __call__(self, t):
    return jnp.expand_dims(jnp.where(t, self.on, self.off), -1)

  def distance(self, distribution: Array, target: Array):
    logits = jnp.squeeze(distribution, axis=-1)
    return optax.sigmoid_binary_cross_entropy(
        logits, target)

  def sample(self, rng: jax.Array, distribution: Array, temperature=None, **_):
    logits = jnp.squeeze(distribution, axis=-1)
    if temperature is not None:
      logits = logits / temperature
    probs = jax.nn.sigmoid(logits)
    return jax.random.bernoulli(rng, probs)

  def distribution(self, distribution: Array) -> tfp.distributions.Bernoulli:
    logits = jnp.squeeze(distribution, axis=-1)
    return tfp.distributions.Bernoulli(logits=logits)

embed_bool = BoolEmbedding()

class FloatEmbedding(Embedding[np.float32, np.float32]):

  def __init__(self, name, scale=None, bias=None, lower=-10., upper=10.):
    self.name = name
    self.scale = scale
    self.bias = bias
    self.lower = lower
    self.upper = upper

  @property
  def dtype(self):
    return np.float32

  @property
  def size(self):
    return 1

  def encode(self, t):
    t = t.astype(jnp.float32)
    if self.bias is not None:
      t = t + self.bias
    if self.scale is not None:
      t = t * self.scale
    if self.lower:
      t = jnp.maximum(t, self.lower)
    if self.upper:
      t = jnp.minimum(t, self.upper)
    return t

  def __call__(self, t, **_):
    return jnp.expand_dims(self.encode(t), -1)

  def extract(self, t):
    if self.scale:
      t = t / self.scale
    if self.bias:
      t = t - self.bias
    return jnp.squeeze(t, axis=-1)


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
    self.name = name
    self.one_hot_policy = one_hot_policy
    self._size = size
    if one_hot_policy is OneHotPolicy.EXTRA:
      self._size += 1
    self.input_size = size
    self._dtype = dtype

  @property
  def size(self) -> int:
    return self._size

  @property
  def dtype(self) -> type[T]:
    return self._dtype

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

    # For EMPTY, jax.nn.one_hot already does the right thing of setting everything
    # to 0 for invalid inputs, so we don't need to do anything here.

    return state.astype(self.dtype)

  def __call__(self, t: Array, residual=False, **_):
    one_hot = jax.nn.one_hot(t, self.size)

    if residual:
      logits = math.log(self.size * 10) * one_hot
      return logits
    else:
      return one_hot

  def extract(self, embedded):
    # TODO: pick a random sample?
    return jnp.argmax(embedded, axis=-1).astype(self.dtype)

  def distance(self, distribution: Array, target: Array):
    return optax.softmax_cross_entropy_with_integer_labels(distribution, target)

  def sample(self, rng: jax.Array, distribution: Array, temperature=None, **_):
    logits = distribution
    if temperature is not None:
      logits = logits / temperature
    return jax.random.categorical(rng, logits, axis=-1).astype(self.dtype)

  def distribution(self, distribution: Array) -> tfp.distributions.Categorical:
    return tfp.distributions.Categorical(logits=distribution, dtype=self.dtype)

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

    self._size = 0
    for _, op in embedding:
      self._size += op.size

  @property
  def size(self) -> int:
    return self._size

  @property
  def dtype(self):
    raise NotImplementedError('StructEmbedding has no single dtype')

  def map_shallow(self, f, *args: NT) -> NT:
    result = {
        k: f(e, *(self.getter(x, k) for x in args))
        for k, e in self.embedding}
    return self.builder(result)

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
    return self.map_shallow(lambda e, x: e.from_state(x), state)

  def __call__(self, struct: NT, **kwargs) -> Array:
    embed = []

    for field, op in self.embedding:
      if op.size == 0:
        continue

      t = op(self.getter(struct, field), **kwargs)
      embed.append(t)

    assert embed, "Embedding must not be empty"
    return jnp.concatenate(embed, axis=-1)

  def dummy(self, shape: Sequence[int] = ()):
    return self.map(lambda e: e.dummy(shape))

  def dummy_embedding(self, shape: Sequence[int] = ()):
    return self.map(lambda e: e.dummy_embedding(shape))

  def decode(self, out: NT) -> NT:
    return self.map(lambda e, x: e.decode(x), out)

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

@dataclasses.dataclass
class ItemsConfig:
  type: ItemsType = ItemsType.FLAT

def make_items_embedding(
    items_config: ItemsConfig,
    xy_scale: float,
) -> Embedding[Items, tp.Any]:
  if items_config.type is ItemsType.SKIP:
    return ordered_struct_embedding("items", [], Items)

  embed_item = make_item_embedding(xy_scale)

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
    assert embed_randall._size == 0

  if with_fod:
    embed_height = FloatEmbedding("fod_height", scale=player_config['xy_scale'])
    embed_fod = struct_embedding_from_nt(
        "fod", FoDPlatforms(left=embed_height, right=embed_height))
  else:
    # Older agents don't use FoD
    embed_fod = ordered_struct_embedding(
        "fod", [], FoDPlatforms)
    assert embed_fod._size == 0

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

  def from_state(self, state: Union[np.float32, np.ndarray]):
    assert state.dtype == np.float32
    return (state * self.n + 0.5).astype(self.dtype)

  def decode(self, out: Union[np.uint8, np.ndarray]) -> Union[np.float32, np.ndarray]:
    assert out.dtype == self.dtype
    return (out / self.n).astype(np.float32)

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
class DefaultControllerConfig:
  axis_spacing: int = 16
  shoulder_spacing: int = 4

  def make_embedding(self):
    return get_controller_embedding(
        axis_spacing=self.axis_spacing,
        shoulder_spacing=self.shoulder_spacing,
    )

Mid = tp.TypeVar("Mid")

class CompoundEmbedding(Embedding[In, Out]):

  def __init__(
      self,
      encode: Callable[[In], Mid],
      decode: Callable[[Mid], In],
      embed_mid: Embedding[Mid, Out],
  ):
    self._encode = encode
    self._decode = decode
    self._embed_mid = embed_mid

  @property
  def dtype(self) -> type:
    return self._embed_mid.dtype

  @property
  def size(self) -> int:
    return self._embed_mid.size

  def from_state(self, state: In) -> Out:
    mid = self._encode(state)
    return self._embed_mid.from_state(mid)

  def __call__(self, x: Out) -> Array:
    return self._embed_mid(x)

  def map(self, f, *args: Out) -> Out:
    return self._embed_mid.map(f, *args)

  def flatten(self, struct: Out) -> Iterator[Any]:
    yield from self._embed_mid.flatten(struct)

  def unflatten(self, seq: Iterator[Any]) -> Out:
    return self._embed_mid.unflatten(seq)

  def decode(self, out: Out) -> In:
    """Inverse of `from_state`."""
    mid = self._embed_mid.decode(out)
    return self._decode(mid)

  def dummy(self, shape: Sequence[int] = ()) -> Out:
    return self._embed_mid.dummy(shape)

  def dummy_embedding(self, shape: Sequence[int] = ()) -> Out:
    return self._embed_mid.dummy_embedding(shape)

  def distribution_size(self) -> int:
    return self._embed_mid.distribution_size()

  def sample(self, rng: jax.Array, distribution: Array, **kwargs) -> Out:
    return self._embed_mid.sample(rng, distribution, **kwargs)

  def distance(self, distribution: Array, target: Out) -> Out:
    return self._embed_mid.distance(distribution, target)

  def distribution(self, distribution: Array) -> tfp.distributions.Distribution:
    return self._embed_mid.distribution(distribution)

  def kl_divergence(self, logits_p: Array, logits_q: Array) -> Array:
    return self._embed_mid.kl_divergence(logits_p, logits_q)

  def entropy(self, logits: Array) -> Array:
    return self._embed_mid.entropy(logits)

def make_custom_v1_embedding(
    config: cv1.Config,
) -> Embedding[Controller, cv1.ControllerV1]:
  bucketer = config.create_bucketer()
  buttons_size, main_stick_size = bucketer.axis_sizes

  embed_custom_v1 = ordered_struct_embedding(
      name="custom_v1",
      embedding=[
          ('buttons', OneHotEmbedding('buttons', buttons_size, dtype=np.uint16)),
          ('main_stick', OneHotEmbedding('main_stick', main_stick_size, dtype=np.uint16)),
      ],
      nt_type=cv1.ControllerV1,
  )

  return CompoundEmbedding(
      encode=bucketer.bucket,
      decode=bucketer.decode,
      embed_mid=embed_custom_v1,
  )

class ControllerType(enum.Enum):
  DEFAULT = 'default'
  CUSTOM_V1 = 'custom_v1'

@dataclasses.dataclass
class ControllerConfig:
  type: str = ControllerType.DEFAULT.value  # make this an enum?
  default: DefaultControllerConfig = utils.field(DefaultControllerConfig)
  custom_v1: cv1.Config = utils.field(cv1.Config.default)

  def make_embedding(self):
    match ControllerType(self.type):
      case ControllerType.DEFAULT:
        return self.default.make_embedding()
      case ControllerType.CUSTOM_V1:
        return make_custom_v1_embedding(self.custom_v1)

@dataclasses.dataclass
class EmbedConfig:
  player: PlayerConfig = utils.field(PlayerConfig)
  controller: ControllerConfig = utils.field(ControllerConfig)
  with_randall: bool = True
  with_fod: bool = True
  items: ItemsConfig = utils.field(ItemsConfig)

  def make_game_embedding(self):
    return make_game_embedding(
        player_config=dataclasses.asdict(self.player),
        with_randall=self.with_randall,
        with_fod=self.with_fod,
        items_config=self.items,
    )

  def make_state_action_embedding(self, num_names: int):
    return get_state_action_embedding(
        embed_game=self.make_game_embedding(),
        embed_action=self.controller.make_embedding(),
        num_names=num_names,
    )

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
