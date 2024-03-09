import abc
import atexit
from dataclasses import dataclass
import logging
from typing import Dict, Mapping

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

class ConnectFailed(Exception):
  """Raised when we fail to connect to the console."""

class Dolphin:

  def __init__(
      self,
      path: str,
      iso: str,
      players: Mapping[int, Player],
      stage: melee.Stage = melee.Stage.FINAL_DESTINATION,
      online_delay=0,
      blocking_input=True,
      slippi_port=51441,
      render=True,
      save_replays=False,  # Override default in Console
      env_vars=None,
      headless=False,
      **console_kwargs,
  ) -> None:
    self._players = players
    self._stage = stage

    if headless:
      render = False
      console_kwargs.update(
          disable_audio=True,
          use_exi_inputs=True,
          enable_ffw=True,
      )

    console = melee.Console(
        path=path,
        online_delay=online_delay,
        blocking_input=blocking_input,
        slippi_port=slippi_port,
        gfx_backend='' if render else 'Null',
        copy_home_directory=False,
        setup_gecko_codes=True,
        save_replays=save_replays,
        **console_kwargs,
    )
    atexit.register(console.stop)
    self.console = console

    self.controllers: Mapping[int, melee.Controller] = {}
    self._menuing_controllers: list[tuple[melee.Controller, Player]] = []
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
        iso_path=iso,
        environment_vars=env_vars,
    )

    logging.info('Connecting to console...')
    if not console.connect():
      import os
      logging.error(
          f"PID {os.getpid()}: failed to connect to the console"
          f" {console.temp_dir} on port {slippi_port}")

      import time; time.sleep(1000)
      self.stop()
      raise ConnectFailed(f"Failed to connect to the console on port {slippi_port}.")

    for controller in self.controllers.values():
      if not controller.connect():
        self.stop()
        raise ConnectFailed("Failed to connect the controller.")

  def next_gamestate(self) -> melee.GameState:
    gamestate = self.console.step()
    assert gamestate is not None
    return gamestate

  def step(self) -> melee.GameState:
    gamestate = self.next_gamestate()

    # The console object keeps track of how long your bot is taking to process frames
    #   And can warn you if it's taking too long
    # if self.console.processingtime * 1000 > 12:
    #     print("WARNING: Last frame took " + str(self.console.processingtime*1000) + "ms to process.")

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

    return gamestate

  def stop(self):
    self.console.stop()

  def __del__(self):
    self.stop()

  def multi_step(self, n: int):
    for _ in range(n):
      self.step()
