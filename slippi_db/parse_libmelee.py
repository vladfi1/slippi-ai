from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa

import melee

from slippi_ai.types import (
  GAME_TYPE,
  LIBMELEE_BUTTONS,
  Buttons,
  Controller,
  Game,
  InvalidGameError,
  Player,
  Stick,
  Randall,
  nt_to_nest,
)

def get_stick(stick: Tuple[float]) -> Stick:
  return Stick(*map(np.float32, stick))

def get_buttons(button: Dict[melee.Button, bool]) -> Buttons:
  return Buttons(**{
      name: button[lm_button]
      for name, lm_button in LIBMELEE_BUTTONS.items()
  })

def get_controller(cs: melee.ControllerState) -> Controller:
  return Controller(
      main_stick=get_stick(cs.main_stick),
      c_stick=get_stick(cs.c_stick),
      shoulder=cs.l_shoulder,
      buttons=get_buttons(cs.button),
  )

def get_player(player: melee.PlayerState) -> Player:
  if player.action == melee.Action.UNKNOWN_ANIMATION:
    raise InvalidGameError('UNKNOWN_ANIMATION')

  return Player(
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

def get_game(
    game: melee.GameState,
    ports: Optional[Sequence[int]] = None,
) -> Game:
  ports = ports or sorted(game.players)
  assert len(ports) == 2
  players = {
      f'p{i}': get_player(game.players[p])
      for i, p in enumerate(ports)}

  if game.stage is melee.Stage.YOSHIS_STORY:
    randall_y, randall_x_left, randall_x_right = melee.randall_position(game.frame)
    randall_x = (randall_x_left + randall_x_right) / 2
  else:
    randall_y = randall_x = 0.0

  return Game(
      stage=np.uint8(game.stage.value),
      randall=Randall(
          x=randall_x,
          y=randall_y,
      ),
      **players,
  )

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
    game = get_game(gamestate)
    frames.append(nt_to_nest(game))
    gamestate = console.step()

  return pa.array(frames, type=GAME_TYPE)
