import abc
import atexit
import dataclasses
import logging
from typing import Dict, Mapping, Optional

import fancyflags as ff
import melee
from melee.console import is_mainline_dolphin, DumpConfig



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

@dataclasses.dataclass
class CPU(Player):
  character: melee.Character = melee.Character.FOX
  level: int = 9

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.STANDARD

  def menuing_kwargs(self) -> Dict:
      return dict(character_selected=self.character, cpu_level=self.level)

@dataclasses.dataclass
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
      save_replays=False,  # Override default in Console
      env_vars=None,
      headless: bool = False,
      render: Optional[bool] = None,  # Render even when running headless.
      **console_kwargs,
  ) -> None:
    self._players = players
    self._stage = stage

    platform = None
    is_mainline = is_mainline_dolphin(path)

    if render is None:
      render = not headless

    if headless:
      console_kwargs.update(
          disable_audio=True,
      )
      if is_mainline:
        platform = 'headless'
        console_kwargs.update(emulation_speed=0)
      else:
        console_kwargs.update(
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
        platform=platform,
    )

    logging.info('Connecting to console...')
    if not console.connect():
      import os
      logging.error(
          f"PID {os.getpid()}: failed to connect to the console"
          f" {console.temp_dir} on port {slippi_port}")

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

_field = lambda f: dataclasses.field(default_factory=f)

@dataclasses.dataclass
class DolphinConfig:
  """Configure dolphin for evaluation."""
  path: Optional[str] = None  # Path to folder containing the dolphin executable
  iso: Optional[str] = None  # Path to melee 1.02 iso.
  stage: melee.Stage = melee.Stage.RANDOM_STAGE  # Which stage to play on.
  online_delay: int = 0  # Simulate online delay.
  blocking_input: bool = True  # Have game wait for AIs to send inputs.
  slippi_port: int = 51441  # Local ip port to communicate with dolphin.
  fullscreen: bool = False # Run dolphin in full screen mode
  render: Optional[bool] = None  # Render frames. Only disable if using vladfi1\'s slippi fork.
  save_replays: bool = False  # Save slippi replays to the usual location.
  replay_dir: Optional[str] = None  # Directory to save replays to.
  headless: bool = True  # Headless configuration: exi + ffw, no graphics or audio.
  infinite_time: bool = True  # Infinite time no stocks.
  log_level: int = 3  # WARN; 0 to disable
  dump: DumpConfig = _field(DumpConfig)  # For framedumping.

  def to_kwargs(self) -> dict:
    kwargs = dataclasses.asdict(self)
    del kwargs['dump']
    kwargs['dump_config'] = self.dump
    return kwargs

  @classmethod
  def kwargs_from_flags(cls, flags: dict) -> dict:
    kwargs = flags.copy()
    del kwargs['dump']
    kwargs['dump_config'] = DumpConfig(**flags['dump'])
    return kwargs

# TODO: replace usage with the above dataclass
DOLPHIN_FLAGS = dict(
    path=ff.String(None, 'Path to folder containing the dolphin executable.'),
    iso=ff.String(None, 'Path to melee 1.02 iso.'),
    stage=ff.EnumClass(melee.Stage.RANDOM_STAGE, melee.Stage, 'Which stage to play on.'),
    online_delay=ff.Integer(0, 'Simulate online delay.'),
    blocking_input=ff.Boolean(True, 'Have game wait for AIs to send inputs.'),
    slippi_port=ff.Integer(51441, 'Local ip port to communicate with dolphin.'),
    fullscreen=ff.Boolean(False, 'Run dolphin in full screen mode.'),
    render=ff.Boolean(None, 'Render frames. Only disable if using vladfi1\'s slippi fork.'),
    save_replays=ff.Boolean(False, 'Save slippi replays to the usual location.'),
    replay_dir=ff.String(None, 'Directory to save replays to.'),
    headless=ff.Boolean(
        False, 'Headless configuration: exi + ffw, no graphics or audio.'),
    infinite_time=ff.Boolean(False, 'Infinite time no stocks.'),
)
