import abc
import atexit
from dataclasses import dataclass
import logging
from typing import Dict, Mapping, Tuple

import melee


class Player(abc.ABC):

  @abc.abstractmethod
  def controller_type(self) -> melee.ControllerType:
    pass

  @abc.abstractmethod
  def menuing_kwargs(self) -> Dict:
    pass


class Human(Player):

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.GCN_ADAPTER
  
  def menuing_kwargs(self) -> Dict:
      return {}

@dataclass
class CPU(Player):
  character: melee.Character = melee.Character.FOX
  level: int = 9

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.STANDARD

  def menuing_kwargs(self) -> Dict:
      return dict(character_selected=self.character, cpu_level=self.level)

@dataclass
class AI(Player):
  character: melee.Character = melee.Character.FOX

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.STANDARD

  def menuing_kwargs(self) -> Dict:
      return dict(character_selected=self.character)

def _is_menu_state(gamestate: melee.GameState) -> bool:
  return gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]


class Dolphin:

  def __init__(
      self,
      dolphin_path: str,
      iso_path: str,
      players: Mapping[int, Player],
      stage: melee.Stage = melee.Stage.YOSHIS_STORY,
      online_delay=0,
      blocking_input=True,
      slippi_port=51441,
      render=True,
      exe_name=None,
      **console_kwargs,
  ) -> None:
    self._players = players
    self._stage = stage

    console = melee.Console(
        path=dolphin_path,
        online_delay=online_delay,
        blocking_input=blocking_input,
        slippi_port=slippi_port,
        gfx_backend='' if render else 'Null',
        emulation_speed=0,
        dolphin_home_path='./data/dolphin_user/',
        **console_kwargs,
    )
    atexit.register(console.stop)
    self.console = console

    self.controllers = {}
    self._menuing_controllers = []
    self._autostart = True

    for port, player in players.items():
      controller = melee.Controller(
          console, port, player.controller_type())
      self.controllers[port] = controller
      if isinstance(player, Human):
        self._autostart = False
      else:
        self._menuing_controllers.append((controller, player))

    console.run(
        exe_name=exe_name,
        iso_path=iso_path,
        environment_vars=dict(vblank_mode='0'),
    )

    logging.info('Connecting to console...')
    if not console.connect():
      raise RuntimeError("Failed to connect to the console.")

    for controller in self.controllers.values():
      if not controller.connect():
        raise RuntimeError("Failed to connect the controller.")

  def next_gamestate(self) -> melee.GameState:
    gamestate = None
    while gamestate is None:
      gamestate = self.console.step()
    return gamestate

  def step(self) -> Tuple[melee.GameState, bool]:
    gamestate = self.next_gamestate()

    # The console object keeps track of how long your bot is taking to process frames
    #   And can warn you if it's taking too long
    if self.console.processingtime * 1000 > 12:
        print("WARNING: Last frame took " + str(self.console.processingtime*1000) + "ms to process.")

    menu_frames = 0
    while _is_menu_state(gamestate):
      for i, (controller, player) in enumerate(self._menuing_controllers):

        melee.MenuHelper.menu_helper_simple(
            gamestate, controller,
            stage_selected=self._stage,
            connect_code=None,
            autostart=self._autostart and i == 0 and menu_frames > 180,
            swag=False,
            costume=i,
            **player.menuing_kwargs())

      gamestate = self.next_gamestate()
      menu_frames += 1

    new_game = gamestate.frame == -123
    return gamestate, new_game

  def stop(self):
    self.console.stop()

  def multi_step(self, n: int):
    for _ in range(n):
      self.step()
