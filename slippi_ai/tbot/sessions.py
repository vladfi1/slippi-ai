"""Dolphin sessions, run as ray actors."""

import collections
import dataclasses
import enum
import json
import logging
import random
import threading
from typing import Optional

import ray

import melee

from slippi_ai import eval_lib
from slippi_ai import dolphin as dolphin_lib
from slippi_ai.tbot.agents import Agent, AgentConfig

# Action states in which the player is dead or on the respawn platform.
# All action states up to ON_HALO_WAIT are death/respawn states.
_MAX_RESPAWN_ACTION = melee.Action.ON_HALO_WAIT.value

def is_dead_or_respawning(player: melee.PlayerState) -> bool:
  return player.action.value <= _MAX_RESPAWN_ACTION

class SDPhase(enum.Enum):
  """Phases of the crew-battle style self-destruct at the start of a game."""
  # Holding away from center until we have SD'd down to the desired stocks.
  SD = 0
  # Standing still until our respawn invincibility has worn off.
  WAIT = 1
  # The agent is in control.
  DONE = 2

@dataclasses.dataclass
class SessionStatus:
  num_menu_frames: int
  is_alive: bool
  games_completed: int = 0

class LogBuffer(logging.Handler):
  """Keeps a session's latest log lines for the frontends to show.

  Each session is its own ray actor process, so the buffer is attached to the
  root logger and sees everything the session logs.
  """

  def __init__(self, max_lines: int = 1000):
    super().__init__(level=logging.INFO)
    self.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S'))
    self._lines: collections.deque[tuple[int, str]] = collections.deque(
        maxlen=max_lines)
    self._next = 0
    self._lock = threading.Lock()

  def emit(self, record: logging.LogRecord):
    try:
      line = self.format(record)
    except Exception:
      self.handleError(record)
      return
    with self._lock:
      self._lines.append((self._next, line))
      self._next += 1

  def since(self, start: int) -> tuple[int, list[str]]:
    """Lines numbered start and above, and the number of the next line."""
    with self._lock:
      return self._next, [line for i, line in self._lines if i >= start]

def set_memory_growth():
  """Set memory growth for all available GPUs."""
  import tensorflow as tf

  gpus = tf.config.experimental.list_physical_devices('GPU')

  for gpu in gpus:
    try:
      tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      logging.warning(f'Could not set memory growth for GPU {gpu}: {e}')

class BotSession:
  """Session between two bots playing locally."""

  def __init__(
      self,
      dolphin_config: dolphin_lib.DolphinConfig,
      agent_configs: dict[int, AgentConfig],
      extra_dolphin_kwargs: dict = {},
      run_on_cpu: bool = False,
      max_games: Optional[int] = None,
  ):
    logging.basicConfig(level=logging.INFO)
    self._log_buffer = LogBuffer()
    logging.getLogger().addHandler(self._log_buffer)

    if run_on_cpu:
      eval_lib.disable_gpus()
    else:
      set_memory_growth()

    self.dolphin_config = dolphin_config
    self.stop_requested = threading.Event()
    self._games_completed = 0

    ports = agent_configs.keys()
    dolphin_kwargs = dolphin_config.to_kwargs()
    dolphin_kwargs.update(extra_dolphin_kwargs)

    players = {
        port: dolphin_lib.AI(character=config.character)
        for port, config in agent_configs.items()
    }
    dolphin = dolphin_lib.Dolphin(
        players=players,
        **dolphin_kwargs,
    )

    agents: dict[int, Agent] = {}
    for port, opponent_port in zip(ports, reversed(ports)):
      agents[port] = agent_configs[port].build(
          controller=dolphin.controllers[port],
          port=port,
          opponent_port=opponent_port,
          console_delay=dolphin.console.online_delay,
      )

    def run():
      # Don't block in the menu so that we can stop if asked to.
      gamestates = dolphin.iter_gamestates(skip_menu_frames=False)

      # Main loop
      for agent in agents.values():
        agent.start()
      try:
        was_in_game = False
        while not self.stop_requested.is_set():
          gamestate = next(gamestates)
          in_game = not dolphin_lib.is_menu_state(gamestate)
          if in_game:
            for agent in agents.values():
              agent.step(gamestate)
          # Detect game-end transition: was in game, now in menu.
          if was_in_game and not in_game:
            self._games_completed += 1
            if max_games is not None:
              logging.info(f'Bot session: game {self._games_completed} completed.')
              if self._games_completed >= max_games:
                logging.info(f'Bot session: reached max_games={max_games}, stopping.')
                break
          was_in_game = in_game
      finally:
        for agent in agents.values():
          agent.stop()
        dolphin.stop()

    self._thread = threading.Thread(target=run)
    self._thread.start()

  def stop(self):
    self.stop_requested.set()
    self._thread.join()

  def logs(self, since: int) -> tuple[int, list[str]]:
    return self._log_buffer.since(since)

  def status(self) -> SessionStatus:
    return SessionStatus(
        num_menu_frames=0,
        is_alive=self._thread.is_alive(),
        games_completed=self._games_completed,
    )

# Lets status and logs be read while the session is stopping.
MAX_CONCURRENCY = 4

GPU_KWARGS = dict(
  num_gpus=0.1,
  max_concurrency=MAX_CONCURRENCY,
  runtime_env={
    'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
  }
)

RemoteBotSession = ray.remote(num_gpus=0, max_concurrency=MAX_CONCURRENCY)(BotSession)
RemoteGpuBotSession = ray.remote(**GPU_KWARGS)(BotSession)

def get_ports(gamestate: melee.GameState, display_name: str):
  name_to_port = {
      player.displayName: port for port, player in gamestate.players.items()
  }
  if display_name not in name_to_port:
    raise ValueError(f'Could not find display name {display_name} in {name_to_port}')
  actual_port = name_to_port[display_name]
  ports = list(gamestate.players)
  ports.remove(actual_port)
  opponent_port = ports[0]
  return actual_port, opponent_port


class Session:

  def __init__(
      self,
      dolphin_config: dolphin_lib.DolphinConfig,
      agent_config: AgentConfig,
      extra_dolphin_kwargs: dict = {},
      stages: Optional[list[melee.Stage]] = None,
      run_on_cpu: bool = False,
      costume: Optional[int] = None,
      sd_stocks: Optional[int] = None,
  ):
    """
    Args:
      sd_stocks: If set, the agent will self-destruct at the start of each game
        until it has at most this many stocks left.
    """
    logging.basicConfig(level=logging.INFO)
    self._log_buffer = LogBuffer()
    logging.getLogger().addHandler(self._log_buffer)

    if run_on_cpu:
      eval_lib.disable_gpus()
    else:
      set_memory_growth()

    self.dolphin_config = dolphin_config
    self.agent_config = agent_config
    self.stop_requested = threading.Event()
    stages = stages or [dolphin_config.stage]
    self._sd_stocks = sd_stocks
    self._sd_phase = SDPhase.DONE

    dolphin_config.online_delay = agent_config.target_console_delay
    logging.info(f'Setting console delay to {agent_config.target_console_delay}')

    port = 1
    dolphin_kwargs = dolphin_config.to_kwargs()
    dolphin_kwargs.update(extra_dolphin_kwargs)

    player = dolphin_lib.AI(
        character=agent_config.character,
        costume=costume,
    )
    dolphin = dolphin_lib.Dolphin(
        players={port: player},
        **dolphin_kwargs,
    )
    controller = dolphin.controllers[port]

    agent = agent_config.build(
        controller=controller,
        console_delay=agent_config.target_console_delay,
        # Ports will be set once we're in game.
        port=0, opponent_port=0,
    )

    # In netplay the ports are going to be randomized. We use the displayName
    # to figure out which port we actually are. The connect code would be better
    # but doesn't appear to work :(
    with open(dolphin_config.user_json_path) as f:
      user_json = json.load(f)
    display_name = user_json['displayName']

    logging.info(
        f'Playing as {agent_config.name}; waiting for'
        f' {dolphin_config.connect_code} to connect.')
    self._num_menu_frames = 0

    def run():
      # Don't block in the menu so that we can stop if asked to.
      gamestates = dolphin.iter_gamestates(skip_menu_frames=False)

      # The agent's actual (netplay) port, set at the start of each game.
      actual_port: Optional[int] = None
      # Set at the start of each game and cleared once the agent has been
      # stepped, so that the agent sees a reset even if we skipped stepping
      # it on the first frame (e.g. while self-destructing).
      agent_needs_reset = False

      # Main loop
      agent.start()
      try:
        while not self.stop_requested.is_set():
          gamestate = next(gamestates)

          if not dolphin_lib.is_menu_state(gamestate):
            # Ports may change if opponent disconnects and reconnects.
            if gamestate.frame == -123:
              actual_port, opponent_port = get_ports(gamestate, display_name)
              logging.info(f'Game started on {gamestate.stage.name}.')
              agent.set_ports(actual_port, opponent_port)
              agent_needs_reset = True
              if self._sd_stocks is not None:
                self._sd_phase = SDPhase.SD

            if self._sd_step(gamestate, actual_port, controller):
              # We are self-destructing or waiting out our invincibility; the
              # agent is not stepped so that it doesn't see any of this.
              pass
            else:
              agent.step(gamestate, needs_reset=agent_needs_reset)
              agent_needs_reset = False

            self._num_menu_frames = 0
          else:
            controller.release_all()

            if self._num_menu_frames == 0:
              dolphin.stage = random.choice(stages)
              logging.info(f'Setting stage to {dolphin.stage.name}')

            self._num_menu_frames += 1
      finally:
        agent.stop()
        dolphin.stop()

    self._thread = threading.Thread(target=run)
    self._thread.start()

  def _sd_step(
      self, gamestate: melee.GameState, port: Optional[int],
      controller: melee.Controller,
  ) -> bool:
    """Crew-battle style start of game: SD down to the desired stock count.

    Returns True if we took control of the controller (and the agent should
    therefore not be stepped), False if the agent is in control.
    """
    if self._sd_phase is SDPhase.DONE or port is None:
      return False

    player = gamestate.players.get(port)
    if player is None:
      return False

    if self._sd_phase is SDPhase.SD:
      if player.stock > self._sd_stocks:
        # Hold the control stick away from the center of the stage so that we
        # walk/fall off and die. All legal stages are centered at x=0.
        away_x = 1.0 if player.position.x >= 0 else 0.0
        controller.release_all()
        controller.tilt_analog(melee.Button.BUTTON_MAIN, away_x, 0.5)
        return True
      logging.info('SD complete, waiting out invincibility.')
      self._sd_phase = SDPhase.WAIT

    if self._sd_phase is SDPhase.WAIT:
      # As in a crew battle, do nothing until our respawn invincibility is
      # over. Note that libmelee's `invulnerable` also covers intangibility
      # (e.g. ledge grabs), which is why this is only checked right after
      # respawning rather than any time we have the desired number of stocks.
      if is_dead_or_respawning(player) or player.invulnerable:
        controller.release_all()
        return True
      logging.info('Invincibility over, handing control to the agent.')
      self._sd_phase = SDPhase.DONE

    return False

  def num_menu_frames(self) -> int:
    return self._num_menu_frames

  def status(self) -> SessionStatus:
    # Return current_model from EnsembleAgent
    return SessionStatus(
        num_menu_frames=self._num_menu_frames,
        is_alive=self._thread.is_alive(),
    )

  def stop(self):
    self.stop_requested.set()
    self._thread.join()

  def logs(self, since: int) -> tuple[int, list[str]]:
    return self._log_buffer.since(since)

RemoteSession = ray.remote(num_gpus=0, max_concurrency=MAX_CONCURRENCY)(Session)
RemoteGpuSession = ray.remote(**GPU_KWARGS)(Session)
