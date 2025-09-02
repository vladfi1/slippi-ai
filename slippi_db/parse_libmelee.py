from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa

import melee

from slippi_ai import utils
from slippi_ai.types import (
  GAME_TYPE,
  LIBMELEE_BUTTONS,
  Buttons,
  Controller,
  Game,
  InvalidGameError,
  Player, Nana,
  Stick,
  Randall,
  FoDPlatforms,
  Item, Items, MAX_ITEMS,
  nt_to_nest,
)
from slippi_db import parsing_utils

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
      buttons=get_buttons(cs.processed_button),
  )

def get_base_player(player: melee.PlayerState) -> dict:
  return dict(
      percent=player.percent,
      facing=player.facing,
      x=player.position.x,
      y=player.position.y,
      action=player.action.value,
      character=player.character.value,
      jumps_left=player.jumps_left,
      shield_strength=player.shield_strength,
      on_ground=player.on_ground,
      # v2.1.0
      invulnerable=player.invulnerable,
      # v3.5.0
      # player.speed_air_x_self,
      # player.speed_ground_x_self,
      # player.speed_x_attack,
      # player.speed_y_attack,
      # player.speed_y_self,
  )

_EMPTY_NANA = utils.map_nt(
    lambda t: t(0),
    utils.reify_tuple_type(Nana)
)
assert not _EMPTY_NANA.exists

def get_player(player: melee.PlayerState) -> Player:

  if player.nana is not None:
    nana_dict = get_base_player(player.nana)
    nana = Nana(exists=True, **nana_dict)
  else:
    nana = _EMPTY_NANA

  return Player(
      nana=nana,
      controller=get_controller(player.controller_state),
      **get_base_player(player))

_EMPTY_ITEM = utils.map_nt(
  lambda t: t(0),
  utils.reify_tuple_type(Item)
)
assert not _EMPTY_ITEM.exists

# TODO: have an actual flag for unused items?
_EMPTY_ITEMS = utils.map_nt(
    lambda t: t(0),
    utils.reify_tuple_type(Items)
)

def get_item(projectile: melee.Projectile) -> Item:
  return Item(
      exists=True,
      type=np.uint16(projectile.type.value),
      state=np.uint8(projectile.subtype),
      x=projectile.position.x,
      y=projectile.position.y,
  )

class Parser:

  def __init__(self, ports: Optional[Sequence[int]] = None):
    self.item_assigner = parsing_utils.ItemAssigner()
    self.ports = ports

  def get_game(
      self,
      game: melee.GameState,
  ) -> Game:
    ports_this_frame = sorted(game.players)
    assert len(ports_this_frame) == 2
    if self.ports is None:
      self.ports = ports_this_frame
    elif sorted(self.ports) != ports_this_frame:
      raise InvalidGameError(
          f'Ports changed from {self.ports} to {ports_this_frame} on frame {game.frame}')

    players = {
        f'p{i}': get_player(game.players[p])
        for i, p in enumerate(self.ports)}

    if game.stage is melee.Stage.YOSHIS_STORY:
      randall_y, randall_x_left, randall_x_right = melee.randall_position(game.frame)
      randall_x = (randall_x_left + randall_x_right) / 2
    else:
      randall_y = randall_x = 0.0

    if game.fod_platforms:
      fod_platforms = FoDPlatforms(
          left=game.fod_platforms.left,
          right=game.fod_platforms.right,
      )
    else:
      fod_platforms = FoDPlatforms(np.float32(0), np.float32(0))

    # Items
    slots = self.item_assigner.assign(
        [item.spawn_id for item in game.projectiles]
    )

    items_dict = {}
    for slot, item in zip(slots, game.projectiles):
      items_dict[f'item_{slot}'] = get_item(item)

    items = _EMPTY_ITEMS._replace(**items_dict)

    return Game(
        stage=np.uint8(game.stage.value),
        randall=Randall(
            x=np.float32(randall_x),
            y=np.float32(randall_y),
        ),
        fod_platforms=fod_platforms,
        items=items,
        **players,
    )


def get_slp(path: str) -> pa.StructArray:
    console = melee.Console(
      is_dolphin=False,
      allow_old_version=True,
      path=path)
    console.connect()

    gamestate = console.step()
    assert gamestate is not None
    ports = sorted(gamestate.players)
    if len(ports) != 2:
      raise InvalidGameError(f'Not a 2-player game.')

    parser = Parser(ports=ports)
    frames = []

    while gamestate:
      game = parser.get_game(gamestate)
      frames.append(nt_to_nest(game))
      gamestate = console.step()

    return pa.array(frames, type=GAME_TYPE)
