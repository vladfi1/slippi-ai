from typing import Mapping, NamedTuple, Sequence, Tuple, TypeVar, Union
import numpy as np

import pyarrow as pa
from melee.enums import Button

T = TypeVar('T')
Nest = Union[Mapping[str, 'Nest'], T]

# Controller = Nest
# Game = Nest

BUTTONS = [
    Button.BUTTON_A,
    Button.BUTTON_B,
    Button.BUTTON_X,
    Button.BUTTON_Y,
    Button.BUTTON_Z,
    Button.BUTTON_R,
    Button.BUTTON_L,
    Button.BUTTON_START,
    Button.BUTTON_D_LEFT,
    Button.BUTTON_D_RIGHT,
    Button.BUTTON_D_DOWN,
    Button.BUTTON_D_UP,
]

# we define NamedTuples for python typechecking

class Buttons(NamedTuple):
  a: np.bool_
  b: np.bool_
  x: np.bool_
  y: np.bool_
  z: np.bool_
  r: np.bool_
  d_up: np.bool_

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
  character: np.uint8
  jumps_left: np.uint8
  shield_strength: np.float32
  on_ground: np.bool_
  controller: Controller

class Game(NamedTuple):
  p0: Player
  p1: Player
  stage: np.uint8

# Ideally we would auto-generate the pyarrow types from the NamedTuples,
# but sadly NamedTuple._fields loses the typing information :(

PA_TO_NT = {}

BUTTONS_TYPE = pa.struct([(b.value, pa.bool_()) for b in BUTTONS])
PA_TO_NT[BUTTONS_TYPE] = Buttons

STICK_TYPE = pa.struct([
    ('x', pa.float32()),
    ('y', pa.float32()),
])
PA_TO_NT[STICK_TYPE] = Stick

CONTROLLER_TYPE = pa.struct([
    ('main_stick', STICK_TYPE),
    ('c_stick', STICK_TYPE),
    # libmelee reads the logical value and assigns it to both l/r
    ('shoulder', pa.float32()),
    ('buttons', BUTTONS_TYPE),
])
PA_TO_NT[CONTROLLER_TYPE] = Controller

PLAYER_TYPE = pa.struct([
    ('percent', pa.uint16()),
    ('facing', pa.bool_()),
    ('x', pa.float32()),
    ('y', pa.float32()),
    ('action', pa.uint16()),
    # invulnerable=get_post('hurtbox_state') != 0,  # libmelee does extra processing
    ('character', pa.uint8()),
    ('jumps_left', pa.uint8()),
    ('shield_strength', pa.float32()),
    ('on_ground', pa.bool_()),
    ('controller', CONTROLLER_TYPE),
])

GAME_TYPE = pa.struct([
  ('p0', PLAYER_TYPE),
  ('p1', PLAYER_TYPE),
  ('stage', pa.uint8()),
])

Array = Union[pa.Array, np.ndarray]
ArrayNest = Nest[Array]

def array_from_nest(val: ArrayNest) -> Array:
  if isinstance(val, Mapping):
    values = [array_from_nest(v) for v in val.values()]
    return pa.StructArray.from_arrays(values, names=val.keys())
  else:
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

def array_to_nt(val: pa.Array) -> Union[tuple, np.ndarray]:
  nt = PA_TO_NT.get(val.type)
  if nt is None:
    assert val.type.num_fields == 0
    return val.to_numpy(zero_copy_only=False)
  else:
    result = {}
    for field in val.type:
      result[field.name] = array_to_nt(val.field(field.name))
    return nt(**result)

def game_array_to_nt(game: pa.StructArray) -> Game:
  result = array_to_nt(game)
  assert isinstance(result, Game)
  return result

class InvalidGameError(Exception):
  """Base class for invalid game exceptions."""
