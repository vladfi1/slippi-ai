"""Mirrors left/right in Slippi game state data structures."""

import numpy as np

import melee

from slippi_ai import types

def mirror_stick(stick: types.Stick) -> types.Stick:
  return types.Stick(
      x=1.0 - stick.x,  #
      y=stick.y,
  )

def mirror_controller(controller: types.Controller) -> types.Controller:
  return controller._replace(
      main_stick=mirror_stick(controller.main_stick),
      c_stick=mirror_stick(controller.c_stick),
  )

def mirror_nana(nana: types.Nana) -> types.Nana:
  return nana._replace(
      facing=np.logical_not(nana.facing),
      x=-nana.x,
  )

ActionDType = np.uint16
DEAD_LEFT = ActionDType(melee.Action.DEAD_LEFT.value)
DEAD_RIGHT = ActionDType(melee.Action.DEAD_RIGHT.value)

def mirror_action(action: ActionDType) -> ActionDType:
  # This one isn't super important.
  if isinstance(action, np.ndarray):
    dead_left = action == DEAD_LEFT
    dead_right = action == DEAD_RIGHT
    action = action.copy()
    action[dead_left] = DEAD_RIGHT
    action[dead_right] = DEAD_LEFT
    return action

  if action == DEAD_LEFT:
    return DEAD_RIGHT
  elif action == DEAD_RIGHT:
    return DEAD_LEFT
  else:
    return action

def mirror_player(player: types.Player) -> types.Player:
  return player._replace(
      facing=np.logical_not(player.facing),
      x=-player.x,
      action=mirror_action(player.action),
      controller=mirror_controller(player.controller),
      nana=mirror_nana(player.nana),
  )

def mirror_randall(randall: types.Randall) -> types.Randall:
  return randall._replace(
      x=-randall.x,
  )

def mirror_fod_platforms(platforms: types.FoDPlatforms) -> types.FoDPlatforms:
  return types.FoDPlatforms(
      left=platforms.right,
      right=platforms.left,
  )

def mirror_item(item: types.Item) -> types.Item:
  return item._replace(
      # facing=np.logical_not(item.facing),
      x=-item.x,
  )

def mirror_items(items: types.Items) -> types.Items:
  return types.Items(*map(mirror_item, items))

def mirror_game(game: types.Game) -> types.Game:
  return types.Game(
      p0=mirror_player(game.p0),
      p1=mirror_player(game.p1),
      stage=game.stage,
      randall=mirror_randall(game.randall),
      fod_platforms=mirror_fod_platforms(game.fod_platforms),
      items=mirror_items(game.items),
  )
