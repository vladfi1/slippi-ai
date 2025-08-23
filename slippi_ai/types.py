import functools
from typing import Mapping, NamedTuple, TypeVar, Union
import numpy as np

import pyarrow as pa
from melee.enums import Button

T = TypeVar('T')
Nest = Union[Mapping[str, 'Nest'], T]

# we define NamedTuples for python typechecking and IDE integration

class Buttons(NamedTuple):
  A: np.bool_
  B: np.bool_
  X: np.bool_
  Y: np.bool_
  Z: np.bool_
  L: np.bool_
  R: np.bool_
  D_UP: np.bool_

LIBMELEE_BUTTONS = {name: Button(name) for name in Buttons._fields}

class Stick(NamedTuple):
  x: np.float32
  y: np.float32

class Controller(NamedTuple):
  main_stick: Stick
  c_stick: Stick
  shoulder: np.float32
  buttons: Buttons

class Player(NamedTuple):
  percent: np.uint16
  facing: np.bool_
  x: np.float32
  y: np.float32
  action: np.uint16
  invulnerable: np.bool_
  character: np.uint8
  jumps_left: np.uint8
  shield_strength: np.float32
  on_ground: np.bool_
  controller: Controller

class Randall(NamedTuple):
  x: np.float32
  y: np.float32

class FoDPlatforms(NamedTuple):
  left: np.float32
  right: np.float32

class Game(NamedTuple):
  p0: Player
  p1: Player

  stage: np.uint8
  randall: Randall
  fod_platforms: FoDPlatforms


# maps pyarrow types back to NamedTuples
PA_TO_NT = {}

@functools.lru_cache
def nt_to_pa(nt: type) -> pa.StructType:
  """Convert and register a NamedTuple (or numpy) type."""

  if not issubclass(nt, tuple):
    return pa.from_numpy_dtype(nt)

  struct_type = pa.struct([
      (name, nt_to_pa(nt.__annotations__[name]))
      for name in nt._fields
  ])

  PA_TO_NT[struct_type] = nt
  return struct_type

BUTTONS_TYPE = nt_to_pa(Buttons)
STICK_TYPE = nt_to_pa(Stick)
CONTROLLER_TYPE = nt_to_pa(Controller)
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

def array_to_nt(nt: type, val: pa.Array) -> Union[tuple, np.ndarray]:
  if issubclass(nt, tuple):
    assert isinstance(val.type, pa.StructType)
    result = {}
    for name in nt._fields:
      result[name] = array_to_nt(
          nt.__annotations__[name],
          val.field(name))
    return nt(**result)

  assert val.type.num_fields == 0
  return val.to_numpy(zero_copy_only=False)

def game_array_to_nt(game: pa.StructArray) -> Game:
  result = array_to_nt(Game, game)
  assert isinstance(result, Game)
  return result

class InvalidGameError(Exception):
  """Base class for invalid game exceptions."""
