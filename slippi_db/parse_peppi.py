import numpy as np
import pyarrow as pa

import melee
from melee import Button
import peppi_py

from slippi_ai.types import ArrayNest, Buttons, Nest, array_from_nest

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

def get_buttons(button_bits: np.ndarray) -> Nest[np.ndarray]:
  return {
      b.value: np.asarray(
          np.bitwise_and(button_bits, BUTTON_MASKS[b]),
          dtype=bool)
      for b in BUTTON_MASKS
  }

def to_libmelee_stick(raw_stick: np.ndarray) -> np.ndarray:
  return (raw_stick / 2.) + 0.5

def get_stick(stick):
  return dict(
      x=to_libmelee_stick(stick.field('x').to_numpy()),
      y=to_libmelee_stick(stick.field('y').to_numpy()),
  )

def get_player(player: pa.StructArray) -> ArrayNest:
  leader = player.field('leader')

  post = leader.field('post')
  get_post = lambda key: post.field(key)
  position = post.field('position')
  pre = leader.field('pre')

  return dict(
      percent=np.asarray(get_post('damage'), dtype=np.uint16),
      facing=np.asarray(get_post('direction').to_numpy(), dtype=bool),
      x=position.field('x'),
      y=position.field('y'),
      action=get_post('state'),
      # invulnerable=get_post('hurtbox_state') != 0,  # libmelee does extra processing
      character=get_post('character'),  # uint8
      jumps_left=get_post('jumps'),  # uint8
      shield_strength=get_post('shield'),  # float
      controller=dict(
          main_stick=get_stick(pre.field('joystick')),
          c_stick=get_stick(pre.field('cstick')),
          # libmelee reads the logical value and assigns it to both l/r
          shoulder=pre.field('triggers').field('logical'),
          buttons=get_buttons(pre.field('buttons').field('physical')),
      ),
      on_ground=np.logical_not(
          post.field('airborne').to_numpy(zero_copy_only=False)),
  )

def get_players(ports: pa.StructArray) -> ArrayNest:
  return {f'p{i}': get_player(ports.field(str(i))) for i in range(2)}

def get_slp(path: str) -> pa.StructArray:
  game = peppi_py.game(path, rollbacks=True)
  frames = game['frames']

  players = get_players(frames.field('ports'))

  stage = melee.enums.to_internal_stage(game['start']['stage'])
  stage = np.full([len(frames)], stage.value, dtype=np.uint8)

  game = dict(players, stage=stage)
  game_array = array_from_nest(game)

  index = frames.field('index').to_numpy()
  first_indices = []
  next_idx = -123
  for i, idx in enumerate(index):
    if idx == next_idx:
      first_indices.append(i)
      next_idx += 1
  return game_array.take(first_indices)
