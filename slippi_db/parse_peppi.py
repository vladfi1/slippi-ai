import numpy as np
import pyarrow as pa

import melee
from melee import Button
import peppi_py
import peppi_py.frame

from slippi_ai import types, utils
from slippi_db import parsing_utils

BUTTON_MASKS = {
    Button.BUTTON_A: 0x0100,
    Button.BUTTON_B: 0x0200,
    Button.BUTTON_X: 0x0400,
    Button.BUTTON_Y: 0x0800,
    Button.BUTTON_START: 0x1000,
    Button.BUTTON_Z: 0x0010,
    Button.BUTTON_R: 0x0020,
    Button.BUTTON_L: 0x0040,
    Button.BUTTON_D_LEFT: 0x0001,
    Button.BUTTON_D_RIGHT: 0x0002,
    Button.BUTTON_D_DOWN: 0x0004,
    Button.BUTTON_D_UP: 0x0008,
}

def get_buttons(button_bits: np.ndarray) -> types.Buttons:
  return types.Buttons(**{
      name: np.asarray(
          np.bitwise_and(button_bits, BUTTON_MASKS[button]),
          dtype=bool)
      for name, button in types.LIBMELEE_BUTTONS.items()
  })

def to_libmelee_stick(raw_stick: pa.FloatArray) -> np.ndarray:
  return (raw_stick.to_numpy() / 2.) + 0.5

def get_stick(stick: peppi_py.frame.Position) -> types.Stick:
  return types.Stick(
      x=to_libmelee_stick(stick.x),
      y=to_libmelee_stick(stick.y),
  )

def get_player(player: peppi_py.frame.PortData) -> types.Player:
  leader = player.leader

  post = leader.post
  position = post.position
  pre = leader.pre

  return types.Player(
      percent=np.asarray(post.percent, dtype=np.uint16),
      facing=post.direction.to_numpy() > 0,
      x=position.x,
      y=position.y,
      action=post.state,
      invulnerable=post.hurtbox_state.to_numpy() != 0,
      character=post.character,  # uint8
      jumps_left=post.jumps,  # uint8
      shield_strength=post.shield,  # float
      controller=types.Controller(
          main_stick=get_stick(pre.joystick),
          c_stick=get_stick(pre.cstick),
          # libmelee reads the logical value and assigns it to both l/r
          shoulder=pre.triggers,
          buttons=get_buttons(pre.buttons_physical),
      ),
      on_ground=np.logical_not(post.airborne),
  )

RANDALL_INTERVAL = 1200
RANDALL_HLR = np.array([
    melee.stages.randall_position(frame)
    for frame in range(RANDALL_INTERVAL)
])

_ITEM_TYPE_STRUCT = utils.reify_tuple_type(types.Item)

def parse_items(
    game_length: int,
    peppi_items: list[peppi_py.frame.Item] | None,
) -> types.Items:
  transposed_items = utils.map_nt(
      lambda t: np.zeros([game_length, types.MAX_ITEMS], dtype=t),
      _ITEM_TYPE_STRUCT,
  )

  if peppi_items is not None:
    assert len(peppi_items) == game_length

    assigner = parsing_utils.ItemAssigner()

    for frame, peppi_item in enumerate(peppi_items):
      slots = assigner.assign(peppi_item.id.to_numpy())

      to_copy = [
          (peppi_item.type, transposed_items.type),
          (peppi_item.state, transposed_items.state),
          (peppi_item.position.x, transposed_items.x),
          (peppi_item.position.y, transposed_items.y),
      ]

      for pa, dst in to_copy:
        assert len(pa) == len(slots)
        dst[frame][slots] = pa.to_numpy()

      transposed_items.exists[frame][slots] = True

  items = utils.map_nt(np.transpose, transposed_items)

  return types.Items(**{
      f'item_{i}': utils.map_nt(lambda x: x[i], items)
      for i in range(types.MAX_ITEMS)
  })

def read_slippi(path: str) -> peppi_py.Game:
  return peppi_py.read_slippi(path, rollback_mode=peppi_py.RollbackMode.FIRST)

def from_peppi(peppi_game: peppi_py.Game) -> types.GAME_TYPE:
  frames = peppi_game.frames
  assert frames is not None, 'Game has no frames'

  game_length = len(frames.id)

  players = {}
  for i, player in enumerate(frames.ports):
    players[f'p{i}'] = get_player(player)

  stage = melee.enums.to_internal_stage(peppi_game.start.stage)

  if stage is melee.Stage.YOSHIS_STORY:
    randall_idx = (frames.id.to_numpy() + RANDALL_INTERVAL) % RANDALL_INTERVAL
    randall_hlr = RANDALL_HLR[randall_idx]
    randall_x = (randall_hlr[:, 1] + randall_hlr[:, 2]) / 2
    randall_y = randall_hlr[:, 0]
  else:
    randall_x = np.zeros(len(frames.id), dtype=np.float32)
    randall_y = np.zeros(len(frames.id), dtype=np.float32)

  # FoD Platforms
  fod_platform_heights = [
      np.zeros(game_length, dtype=np.float32),
      np.zeros(game_length, dtype=np.float32),
  ]

  RIGHT = peppi_py.frame.FodPlatform.RIGHT
  assert RIGHT.value == 0
  LEFT = peppi_py.frame.FodPlatform.LEFT
  assert LEFT.value == 1

  if stage is melee.Stage.FOUNTAIN_OF_DREAMS and frames.fod_platforms:
    # Initial heights are always 20 and 28
    current_heights = [28., 20.]

    for i, moves in enumerate(frames.fod_platforms):
      for move in moves:
        current_heights[move.platform.value] = move.height
      for platform in (LEFT, RIGHT):
        fod_platform_heights[platform.value][i] = current_heights[platform.value]

  game = types.Game(
      stage=np.full([game_length], stage.value, dtype=np.uint8),
      randall=types.Randall(
          x=randall_x.astype(np.float32),
          y=randall_y.astype(np.float32),
      ),
      fod_platforms=types.FoDPlatforms(
          left=fod_platform_heights[LEFT.value],
          right=fod_platform_heights[RIGHT.value],
      ),
      items=parse_items(game_length, frames.items),
      **players,
  )
  return types.array_from_nt(game)

def get_slp(path: str) -> types.GAME_TYPE:
  return from_peppi(read_slippi(path))
