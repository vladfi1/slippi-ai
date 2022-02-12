from typing import Tuple

import numpy as np
import pyarrow as pa

import melee

from slippi_ai.types import (
  BUTTONS,
  CONTROLLER_TYPE,
  PLAYER_TYPE,
  GAME_TYPE,
  STICK_TYPE,
  InvalidGameError,
)

def get_stick(stick: Tuple[np.float32]) -> dict:
  val = dict(x=stick[0], y=stick[1])
  # return pa.scalar(val, type=STICK_TYPE)
  return val

def get_controller(cs: melee.ControllerState) -> dict:
  val = dict(
      main_stick=get_stick(cs.main_stick),
      c_stick=get_stick(cs.c_stick),
      shoulder=cs.l_shoulder,
      buttons={b.value: cs.button[b] for b in BUTTONS},
  )
  # return pa.scalar(val, type=CONTROLLER_TYPE)
  return val

def get_player(player: melee.PlayerState) -> dict:
  if player.action == melee.Action.UNKNOWN_ANIMATION:
    raise InvalidGameError('UNKNOWN_ANIMATION')

  val = dict(
      percent=player.percent,
      facing=player.facing,
      x=player.position.x,
      y=player.position.y,
      action=player.action.value,
      character=player.character.value,
      jumps_left=player.jumps_left,
      shield_strength=player.shield_strength,
      on_ground=player.on_ground,
      controller=get_controller(player.controller_state),
      # v2.1.0
      invulnerable=player.invulnerable,
      # v3.5.0
      # player.speed_air_x_self,
      # player.speed_ground_x_self,
      # player.speed_x_attack,
      # player.speed_y_attack,
      # player.speed_y_self,
  )
  # return pa.scalar(val, type=PLAYER_TYPE)
  return val

def get_game(game: melee.GameState) -> dict:
  ports = sorted(game.players)
  assert len(ports) == 2
  players = {
      f'p{i}': get_player(game.players[p])
      for i, p in enumerate(ports)}
  val = dict(
      players,
      stage=game.stage.value,
  )
  # return pa.scalar(val, type=GAME_TYPE)
  return val

def get_slp(path: str) -> pa.StructArray:
  """Processes a slippi replay file."""
  console = melee.Console(is_dolphin=False,
                          allow_old_version=True,
                          path=path)
  console.connect()

  gamestate = console.step()
  ports = sorted(gamestate.players)
  if len(ports) != 2:
    raise InvalidGameError(f'Not a 2-player game.')

  frames = []

  while gamestate:
    if sorted(gamestate.player) != ports:
      raise InvalidGameError(f'Ports changed on frame {len(frames)}')
    frames.append(get_game(gamestate))
    gamestate = console.step()

  return pa.array(frames, type=GAME_TYPE)
