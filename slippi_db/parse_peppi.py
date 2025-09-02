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

def get_buttons(button_bits: pa.UInt16Array) -> types.Buttons:
  button_bits = button_bits.to_numpy()

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

def get_base_player_data(data: peppi_py.frame.Data, handle_nulls: bool = False) -> dict:
  post = data.post
  position = post.position

  # Helper to convert arrays with potential nulls and replace NaN with 0
  def to_numpy_safe(arr):
    if handle_nulls:
      result = arr.to_numpy(zero_copy_only=False)
      return np.nan_to_num(result, nan=0.0)

    return arr.to_numpy()

  hurtbox_state = post.hurtbox_state
  if hurtbox_state is None:
    hurtbox_state_np = np.zeros(len(position.x), dtype=np.uint8)
  else:
    hurtbox_state_np = to_numpy_safe(hurtbox_state)

  return dict(
      percent=np.asarray(to_numpy_safe(post.percent), dtype=np.uint16),
      facing=to_numpy_safe(post.direction) > 0,
      x=to_numpy_safe(position.x),
      y=to_numpy_safe(position.y),
      action=to_numpy_safe(post.state),
      invulnerable=hurtbox_state_np != 0,
      character=to_numpy_safe(post.character),  # uint8
      jumps_left=to_numpy_safe(post.jumps),  # uint8
      shield_strength=to_numpy_safe(post.shield),  # float
      on_ground=np.logical_not(to_numpy_safe(post.airborne)),
  )

_NANA_TYPE = utils.reify_tuple_type(types.Nana)

def get_player(player: peppi_py.frame.PortData, game_length: int) -> types.Player:
  # Get the base player data for Popo/main player
  leader_data = get_base_player_data(player.leader)

  # Handle Nana (follower)
  if player.follower is not None:
    # Get nana data, allowing nulls to be copied
    follower_data = player.follower

    # Check which frames have valid nana data (not NaN)
    # Use position.x as indicator since it should always be present when nana exists
    x_array = follower_data.post.position.x.to_numpy(zero_copy_only=False)
    exists = ~np.isnan(x_array)

    # Get the base nana data (no controller), allowing nulls to be copied
    nana_data = get_base_player_data(follower_data, handle_nulls=True)

    # Set all fields to 0 where nana doesn't exist, matching libmelee behavior
    nana_data = utils.map_nt(lambda arr: np.where(exists, arr, 0), nana_data)

    nana = types.Nana(exists=exists, **nana_data)
  else:
    # Create empty nana with arrays of zeros for the entire game
    nana = utils.map_nt(
        lambda t: np.zeros(game_length, dtype=t),
        _NANA_TYPE
    )

  pre = player.leader.pre

  return types.Player(
      nana=nana,
      controller=types.Controller(
          main_stick=get_stick(pre.joystick),
          c_stick=get_stick(pre.cstick),
          # libmelee reads the logical value and assigns it to both l/r
          shoulder=pre.triggers.to_numpy(),
          # Use processed buttons because physical buttons aren't preserved across upgrading.
          buttons=get_buttons(pre.buttons),
      ),
      **leader_data)

RANDALL_INTERVAL = 1200
RANDALL_HLR = np.array([
    melee.stages.randall_position(frame)
    for frame in range(RANDALL_INTERVAL)
])

_ITEM_TYPE_STRUCT = utils.reify_tuple_type(types.Item)

def parse_items(
    game_length: int,
    peppi_items: peppi_py.frame.Item | None,
) -> types.Items:
  transposed_items = utils.map_nt(
      lambda t: np.zeros([game_length, types.MAX_ITEMS], dtype=t),
      _ITEM_TYPE_STRUCT,
  )

  if peppi_items is not None:
    assigner = parsing_utils.ItemAssigner()
    slots_by_frame: list[list[int]] = []

    for frame, ids in enumerate(peppi_items.id):
      # Converting slots to numpy saves a lot of time
      slots = assigner.assign(ids.values)
      slots_by_frame.append(slots)
      transposed_items.exists[frame][slots] = True

    to_copy = [
        (peppi_items.type, transposed_items.type),
        (peppi_items.state, transposed_items.state),
        (peppi_items.position.x, transposed_items.x),
        (peppi_items.position.y, transposed_items.y),
    ]

    for list_array, np_array in to_copy:
      assert len(list_array) == game_length
      for frame, (slots, array) in enumerate(zip(slots_by_frame, list_array)):
        # assert len(slots) == len(array)
        np_array[frame][slots] = array.values.to_numpy()

  items = utils.map_nt(np.transpose, transposed_items)

  return types.Items(**{
      f'item_{i}': utils.map_nt(lambda x: x[i], items)
      for i in range(types.MAX_ITEMS)
  })

def read_slippi(path: str) -> peppi_py.Game:
  # NOTE: we might want to switch to LAST as it is consistent across upgrades.
  return peppi_py.read_slippi(path, rollback_mode=peppi_py.RollbackMode.FIRST)

def from_peppi(peppi_game: peppi_py.Game) -> types.GAME_TYPE:
  frames = peppi_game.frames
  assert frames is not None, 'Game has no frames'

  if len(frames.ports) != 2:
    raise ValueError(f"Expected 2 players, got {len(frames.ports)}")

  game_length = len(frames.id)

  players = {}
  for i, player in enumerate(frames.ports):
    players[f'p{i}'] = get_player(player, game_length)

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

    for i, (platforms, heights) in enumerate(
        zip(frames.fod_platforms.platform, frames.fod_platforms.height)):

      for platform, height in zip(platforms, heights):
        current_heights[platform] = height.as_py()

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
