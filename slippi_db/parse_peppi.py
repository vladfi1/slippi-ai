import numpy as np
import pyarrow as pa

import melee
from melee import Button
import peppi_py

from slippi_ai import types

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

def to_libmelee_stick(raw_stick: np.ndarray) -> np.ndarray:
  return (raw_stick / 2.) + 0.5

def get_stick(stick) -> types.Stick:
  return types.Stick(
      x=to_libmelee_stick(np.array(stick.x)),
      y=to_libmelee_stick(np.array(stick.y)),
  )

def get_player(player) -> types.Player:
  leader = player.leader

  post = leader.post
  position = post.position
  pre = leader.pre

  try:
    if hasattr(post, 'state'):
      state = post.state
      if hasattr(state, 'to_numpy'):
        state_array = state.to_numpy()
        if len(state_array) > 0:
          action_value = int(state_array[0])
        else:
          action_value = melee.Action.STANDING.value
      elif isinstance(state, (list, tuple)) and len(state) > 0:
        action_value = int(state[0])
      else:
        action_value = int(state) if state is not None else melee.Action.STANDING.value
    else:
      action_value = melee.Action.STANDING.value
  except (TypeError, AttributeError, IndexError, ValueError):
    action_value = melee.Action.STANDING.value
    
  try:
    if hasattr(post.percent, 'to_numpy'):
      percent = np.asarray(post.percent.to_numpy(), dtype=np.uint16)
    else:
      percent = np.asarray(post.percent, dtype=np.uint16)
  except (AttributeError, TypeError):
    percent = np.zeros(1, dtype=np.uint16)
    
  try:
    if hasattr(post.direction, 'to_numpy'):
      facing = np.array(post.direction.to_numpy()) > 0
    else:
      facing = np.array(post.direction) > 0
  except (AttributeError, TypeError):
    facing = np.zeros(1, dtype=bool)
    
  try:
    if hasattr(position.x, 'to_numpy'):
      x = position.x.to_numpy()
      y = position.y.to_numpy()
    else:
      x = position.x
      y = position.y
  except (AttributeError, TypeError):
    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)
    
  try:
    if hasattr(post.hurtbox_state, 'to_numpy'):
      invulnerable = np.array(post.hurtbox_state.to_numpy()) != 0
    else:
      invulnerable = np.array(post.hurtbox_state) != 0
  except (AttributeError, TypeError):
    invulnerable = np.zeros(1, dtype=bool)
    
  try:
    if hasattr(post.airborne, 'to_numpy'):
      on_ground = np.logical_not(np.array(post.airborne.to_numpy()))
    else:
      on_ground = np.logical_not(np.array(post.airborne))
  except (AttributeError, TypeError):
    on_ground = np.ones(1, dtype=bool)
    
  num_frames = len(percent)
  
  try:
    if hasattr(post.character, 'to_numpy'):
      character = np.array(post.character.to_numpy(), dtype=np.uint8)
    elif isinstance(post.character, (list, tuple)):
      character = np.array(post.character, dtype=np.uint8)
    else:
      character = np.array([post.character] * num_frames, dtype=np.uint8)
  except (AttributeError, TypeError):
    character = np.zeros(num_frames, dtype=np.uint8)
    
  try:
    if hasattr(post.jumps, 'to_numpy'):
      jumps_left = np.array(post.jumps.to_numpy(), dtype=np.uint8)
    elif isinstance(post.jumps, (list, tuple)):
      jumps_left = np.array(post.jumps, dtype=np.uint8)
    else:
      jumps_left = np.array([post.jumps] * num_frames, dtype=np.uint8)
  except (AttributeError, TypeError):
    jumps_left = np.zeros(num_frames, dtype=np.uint8)
    
  try:
    if hasattr(post.shield, 'to_numpy'):
      shield_strength = np.array(post.shield.to_numpy(), dtype=np.float32)
    elif isinstance(post.shield, (list, tuple)):
      shield_strength = np.array(post.shield, dtype=np.float32)
    else:
      shield_strength = np.array([post.shield] * num_frames, dtype=np.float32)
  except (AttributeError, TypeError):
    shield_strength = np.zeros(num_frames, dtype=np.float32)
    
  try:
    if hasattr(pre.triggers, 'to_numpy'):
      shoulder = np.array(pre.triggers.to_numpy(), dtype=np.float32)
    elif isinstance(pre.triggers, (list, tuple)):
      shoulder = np.array(pre.triggers, dtype=np.float32)
    else:
      shoulder = np.array([pre.triggers] * num_frames, dtype=np.float32)
  except (AttributeError, TypeError):
    shoulder = np.zeros(num_frames, dtype=np.float32)
  
  return types.Player(
      percent=percent,
      facing=facing,
      x=x,
      y=y,
      action=np.array([action_value + 1] * num_frames, dtype=np.uint16),
      invulnerable=invulnerable,
      character=character,
      jumps_left=jumps_left,
      shield_strength=shield_strength,
      controller=types.Controller(
          main_stick=get_stick(pre.joystick),
          c_stick=get_stick(pre.cstick),
          shoulder=shoulder,
          buttons=get_buttons(pre.buttons_physical),
      ),
      on_ground=on_ground,
  )

def create_dummy_player(num_frames):
  """Create a dummy player with zeros/empty values."""
  return types.Player(
      percent=np.zeros(num_frames, dtype=np.uint16),
      facing=np.zeros(num_frames, dtype=bool),
      x=np.zeros(num_frames, dtype=np.float32),
      y=np.zeros(num_frames, dtype=np.float32),
      action=np.zeros(num_frames, dtype=np.uint16),
      invulnerable=np.zeros(num_frames, dtype=bool),
      character=np.zeros(num_frames, dtype=np.uint8),
      jumps_left=np.zeros(num_frames, dtype=np.uint8),
      shield_strength=np.zeros(num_frames, dtype=np.float32),
      on_ground=np.zeros(num_frames, dtype=bool),
      controller=types.Controller(
          main_stick=types.Stick(
              x=np.zeros(num_frames, dtype=np.float32),
              y=np.zeros(num_frames, dtype=np.float32),
          ),
          c_stick=types.Stick(
              x=np.zeros(num_frames, dtype=np.float32),
              y=np.zeros(num_frames, dtype=np.float32),
          ),
          shoulder=np.zeros(num_frames, dtype=np.float32),
          buttons=types.Buttons(
              A=np.zeros(num_frames, dtype=bool),
              B=np.zeros(num_frames, dtype=bool),
              X=np.zeros(num_frames, dtype=bool),
              Y=np.zeros(num_frames, dtype=bool),
              Z=np.zeros(num_frames, dtype=bool),
              L=np.zeros(num_frames, dtype=bool),
              R=np.zeros(num_frames, dtype=bool),
              D_UP=np.zeros(num_frames, dtype=bool),
          ),
      ),
  )

def from_peppi(game: peppi_py.Game) -> types.GAME_TYPE:
  frame = game.frames
  
  if hasattr(frame.id, 'to_numpy'):
    frame_ids = frame.id.to_numpy()
  else:
    frame_ids = np.array(frame.id)
  
  num_frames = len(frame_ids)
  
  players = {}
  
  port_names = []
  for p in game.start.players:
    if hasattr(p, 'port'):
      port_names.append(p.port)
  port_names = sorted(port_names)
  
  port_to_idx = {}
  for i, port_name in enumerate(port_names):
    port_to_idx[port_name] = i
  
  if isinstance(frame.ports, tuple) and len(frame.ports) > 0:
    for i, port_data in enumerate(frame.ports[:2]):  # Only process first two ports
      players[f'p{i}'] = get_player(port_data)
  
  if 'p0' not in players:
    players['p0'] = create_dummy_player(num_frames)
  
  if 'p1' not in players:
    players['p1'] = create_dummy_player(num_frames)
  
  stage = melee.enums.to_internal_stage(game.start.stage)
  stage = np.full([num_frames], stage.value, dtype=np.uint8)
  
  game = types.Game(stage=stage, **players)
  game_array = types.array_from_nt(game)
  
  index = frame_ids
  first_indices = []
  next_idx = -123
  for i, idx in enumerate(index):
    if idx == next_idx:
      first_indices.append(i)
      next_idx += 1
  
  if first_indices:
    return game_array.take(first_indices)
  else:
    return game_array

def get_slp(path: str) -> types.GAME_TYPE:
  game = peppi_py.read_slippi(path)
  return from_peppi(game)
