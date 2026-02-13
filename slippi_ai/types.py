"""Parsed games are stored as Arrow `GAME_TYPE` StructArrays.

For each Struct[Array] type, we define a corresponding NamedTuple for easier
manipulation and type checking in Python. The type annotations in these
are scalars, but in practice they are often arrays of the given type with a
time and/or batch dimension.
TODO: use the "returns" library + mypy to get higher-kinded types working.
"""

import functools
from types import GenericAlias
from typing import Mapping, NamedTuple, TypeVar, Union, Generic
import typing as tp
import numpy as np

import pyarrow as pa
from melee.enums import Button

S = TypeVar('S', bound=tuple[int, ...])
T = TypeVar('T')
Nest = Union[Mapping[str, 'Nest'], T]

# we define NamedTuples for python typechecking and IDE integration

BoolDType = np.dtype[np.bool]
FloatDType = np.dtype[np.float32]
Int32DType = np.dtype[np.int32]

BoolArray: tp.TypeAlias = np.ndarray[S, BoolDType]
FloatArray: tp.TypeAlias =  np.ndarray[S, FloatDType]
Int32Array: tp.TypeAlias =  np.ndarray[S, Int32DType]
UInt8Array: tp.TypeAlias = np.ndarray[S, np.dtype[np.uint8]]
UInt16Array: tp.TypeAlias = np.ndarray[S, np.dtype[np.uint16]]

class Buttons(NamedTuple, Generic[S]):
  A: BoolArray[S]
  B: BoolArray[S]
  X: BoolArray[S]
  Y: BoolArray[S]
  Z: BoolArray[S]
  L: BoolArray[S]
  R: BoolArray[S]
  D_UP: BoolArray[S]

LIBMELEE_BUTTONS = {name: Button(name) for name in Buttons._fields}

class Stick(NamedTuple, Generic[S]):
  x: FloatArray[S]
  y: FloatArray[S]

class Controller(NamedTuple, Generic[S]):
  main_stick: Stick[S]
  c_stick: Stick[S]
  shoulder: FloatArray
  buttons: Buttons[S]

class Nana(NamedTuple, Generic[S]):
  exists: BoolArray[S]
  percent: UInt16Array[S]
  facing: BoolArray[S]
  x: FloatArray[S]
  y: FloatArray[S]
  action: UInt16Array[S]
  invulnerable: BoolArray[S]
  character: UInt8Array[S]
  jumps_left: UInt8Array[S]
  shield_strength: FloatArray[S]
  on_ground: BoolArray[S]

class Player(NamedTuple, Generic[S]):
  percent: UInt16Array[S]
  facing: BoolArray[S]
  x: FloatArray[S]
  y: FloatArray[S]
  action: UInt16Array[S]
  invulnerable: BoolArray[S]
  character: UInt8Array[S]
  jumps_left: UInt8Array[S]
  shield_strength: FloatArray[S]
  on_ground: BoolArray[S]
  controller: Controller[S]
  nana: Nana[S]

class Randall(NamedTuple, Generic[S]):
  x: FloatArray[S]
  y: FloatArray[S]

class FoDPlatforms(NamedTuple, Generic[S]):
  left: FloatArray[S]
  right: FloatArray[S]

MAX_ITEMS = 15  # Maximum number of items per frame

class Item(NamedTuple, Generic[S]):
  exists: BoolArray[S]  # Is the Item slot used
  type: UInt16Array[S]
  state: UInt8Array[S]
  # owner?
  # facing: np.ndarray[S, FloatDType]
  x: FloatArray[S]
  y: FloatArray[S]

# TODO: this is inelegant
class Items(NamedTuple, Generic[S]):
  item_0: Item[S]
  item_1: Item[S]
  item_2: Item[S]
  item_3: Item[S]
  item_4: Item[S]
  item_5: Item[S]
  item_6: Item[S]
  item_7: Item[S]
  item_8: Item[S]
  item_9: Item[S]
  item_10: Item[S]
  item_11: Item[S]
  item_12: Item[S]
  item_13: Item[S]
  item_14: Item[S]

# Items = NamedTuple('Items', [
#     (f'item_{i}', Item) for i in range(MAX_ITEMS)
# ])

class Game(NamedTuple, Generic[S]):
  p0: Player[S]
  p1: Player[S]

  stage: UInt8Array[S]
  randall: Randall[S]
  fod_platforms: FoDPlatforms[S]

  items: Items[S]

# maps pyarrow types back to NamedTuples
PA_TO_NT = {}

Leaf = type[np.generic]
Node = list[tuple[str, type]]

@functools.cache
def get_node_or_leaf(t: type | GenericAlias | tp._GenericAlias) -> Node | Leaf:
  if isinstance(t, GenericAlias):
    assert t.__origin__ is np.ndarray

    generic_dtype = t.__args__[1]

    if not isinstance(generic_dtype, GenericAlias) or generic_dtype.__origin__ is not np.dtype:
      raise ValueError(f"Expected a numpy dtype as the second argument of the GenericAlias, got {generic_dtype}")

    dtype = generic_dtype.__args__[0]
    assert issubclass(dtype, np.generic)
    return dtype

  if isinstance(t, tp._GenericAlias):
    t = t.__origin__

  assert issubclass(t, tuple)

  return [(name, t.__annotations__[name]) for name in t._fields]


@functools.lru_cache
def nt_to_pa(nt: type | GenericAlias) -> pa.StructType:
  """Convert and register a NamedTuple (or numpy) type."""

  node_or_leaf = get_node_or_leaf(nt)
  if isinstance(node_or_leaf, list):
    struct_type = pa.struct([
        (name, nt_to_pa(field_type))
        for name, field_type in node_or_leaf
    ])

    PA_TO_NT[struct_type] = nt
    return struct_type

  return pa.from_numpy_dtype(node_or_leaf)

BUTTONS_TYPE = nt_to_pa(Buttons)
STICK_TYPE = nt_to_pa(Stick)
CONTROLLER_TYPE = nt_to_pa(Controller)
NANA_TYPE = nt_to_pa(Nana)
PLAYER_TYPE = nt_to_pa(Player)
GAME_TYPE = nt_to_pa(Game)

def array_from_nest(val: Nest[np.ndarray]) -> pa.StructArray:
  if isinstance(val, Mapping):
    values = [array_from_nest(v) for v in val.values()]
    return pa.StructArray.from_arrays(values, names=val.keys())
  else:
    return val

def array_from_nt(val: Union[tuple, np.ndarray]) -> pa.StructArray:
  if isinstance(val, tuple):
    values = [array_from_nt(v) for v in val]
    return pa.StructArray.from_arrays(values, names=val._fields)
  else:
    return val

def nt_to_nest(val: Union[tuple, T]) -> Nest[T]:
  """ Converts a NamedTuple to a Nest."""
  if isinstance(val, tuple) and hasattr(val, '_fields'):
    return {k: nt_to_nest(v) for k, v in zip(val._fields, val)}
  return val

def array_to_nest(val: pa.Array) -> Nest[np.ndarray]:
  if isinstance(val.type, pa.StructType):
    result = {}
    for field in val.type:
      result[field.name] = array_to_nest(val.field(field.name))
    return result
  else:
    assert val.type.num_fields == 0
    return val.to_numpy(zero_copy_only=False)

def array_to_nt(nt: type[T], val: pa.Array) -> T:
  node_or_leaf = get_node_or_leaf(nt)

  if isinstance(node_or_leaf, list):
    result = {}
    for name, field_type in node_or_leaf:
      result[name] = array_to_nt(field_type, val.field(name))
    return nt(**result)

  # Player percent was cast to uint16 in the past.
  # assert val.type.to_pandas_dtype() == node_or_leaf

  assert val.type.num_fields == 0
  return val.to_numpy(zero_copy_only=False).astype(node_or_leaf)

def game_array_to_nt(game: pa.StructArray) -> Game:
  result = array_to_nt(Game, game)
  assert isinstance(result, Game)
  return result

class InvalidGameError(Exception):
  """Base class for invalid game exceptions."""

# Training-layer types: used by networks, policies, learners, etc.

Action = TypeVar('Action')

NAME_DTYPE = np.int32

class StateAction(NamedTuple, Generic[Action]):
  state: Game
  # The action could actually be an "encoded" action type,
  # which might discretize certain components of the controller
  # such as the sticks and shoulder. Unfortunately NamedTuples can't be
  # generic. We could use a dataclass instead, but TF can't trace them.
  # Note that this is the action taken on the _previous_ frame.
  action: Action

  # Encoded name
  name: NAME_DTYPE

class Frames(NamedTuple, Generic[Action]):
  state_action: StateAction[Action]
  is_resetting: bool
  # The reward will have length one less than the states and actions.
  reward: np.float32
