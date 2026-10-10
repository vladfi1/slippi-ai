"""Plays agents, humans and CPUs against each other in Dolphin.

This is the loop behind scripts/eval_two.py, as a library so that other front
ends (benchmarks, a GUI) can run it too.
"""

import dataclasses
import contextlib
import logging
import os
import threading
import time
import typing as tp

import fancyflags as ff
import melee
from melee.slippstream import EnetDisconnected

from slippi_ai import eval_lib, flag_utils, utils
from slippi_ai import dolphin as dolphin_lib

PORTS = (1, 2)


def player_flags() -> dict:
  """Flags for one player, as values for `SessionConfig.players`."""
  flags = utils.map_nt(lambda x: x, eval_lib.PLAYER_FLAGS)
  flags['ai']['async_inference'] = ff.Boolean(True)
  return flags


def default_dolphin_config() -> dolphin_lib.DolphinConfig:
  """Dolphin at 1x speed with graphics, for playing and watching."""
  return dolphin_lib.DolphinConfig(
      headless=False,
      infinite_time=False,
      online_delay=2,
      emulation_speed=1,
      path=os.environ.get('DOLPHIN_PATH'),
      iso=os.environ.get('ISO_PATH'),
      instant_match_restart=False,
  )


def dolphin_flags() -> dict:
  return flag_utils.get_flags_from_default(default_dolphin_config())


@dataclasses.dataclass
class SessionConfig:
  # Values of `player_flags()` for each port.
  players: dict[int, dict]
  dolphin: dolphin_lib.DolphinConfig = dataclasses.field(
      default_factory=default_dolphin_config)
  # Stop after this many games; None to play until stopped.
  num_games: tp.Optional[int] = None


class StopEvent(tp.Protocol):
  """A threading.Event or multiprocessing.Event."""

  def is_set(self) -> bool:
    ...

  def wait(self, timeout: tp.Optional[float] = None) -> bool:
    ...


class Frame(tp.NamedTuple):
  gamestate: melee.GameState
  step_time: float  # Seconds the agents took to step on this frame.


class Session:
  """Starts the agents and Dolphin; close it to stop them."""

  def __init__(self, config: SessionConfig):
    self.config = config
    self._exit_stack = contextlib.ExitStack()
    try:
      self._start()
    except BaseException:
      self.close()
      raise

  def _start(self):
    eval_lib.disable_gpus()

    players = {
        port: eval_lib.get_player(**player)
        for port, player in self.config.players.items()
    }
    ports = sorted(players)
    if len(ports) != 2:
      raise ValueError(f'Need two players, got ports {ports}.')

    self.agents: list[eval_lib.Agent] = []

    for port, opponent_port in zip(ports, reversed(ports)):
      player = players[port]
      if isinstance(player, dolphin_lib.AI):
        agent = eval_lib.build_agent(
            port=port,
            opponent_port=opponent_port,
            console_delay=self.config.dolphin.online_delay,
            **self.config.players[port]['ai'],
        )
        agent.start()
        self.agents.append(agent)
        self._exit_stack.callback(agent.stop)

        eval_lib.update_character(player, agent.config)

    # TODO: use an envs.Environment like in RL
    self.dolphin = dolphin_lib.Dolphin(
        players=players,
        **self.config.dolphin.to_kwargs(),
    )
    self._exit_stack.callback(self.dolphin.stop)

    for agent in self.agents:
      agent.set_controller(self.dolphin.controllers[agent._port])

  def frames(
      self,
      stop_event: tp.Optional[StopEvent] = None,
  ) -> tp.Iterator[Frame]:
    """Steps the agents on each in-game frame, then yields it.

    Ends after `config.num_games` games, or when `stop_event` is set.
    """
    num_games = 0

    for gamestate in self.dolphin.iter_gamestates(skip_menu_frames=False):
      if stop_event is not None and stop_event.is_set():
        break

      if dolphin_lib.is_menu_state(gamestate):
        if num_games == self.config.num_games:
          break
        continue

      if gamestate.frame == dolphin_lib.INITIAL_FRAME:
        num_games += 1
        logging.info(f'Game {num_games}')

      start = time.perf_counter()
      for agent in self.agents:
        agent.step(gamestate)
      yield Frame(gamestate, time.perf_counter() - start)

  def close(self):
    self._exit_stack.close()

  def __enter__(self) -> 'Session':
    return self

  def __exit__(self, *_):
    self.close()


def run_session(
    config: SessionConfig,
    stop_event: tp.Optional[StopEvent] = None,
) -> None:
  """Plays until `config.num_games` games are done or `stop_event` is set.

  The stop event is checked once per frame and, in case Dolphin stops
  sending frames (e.g. the game was exited), also by a thread that then
  interrupts the wait for the next frame.
  """
  with Session(config) as session:
    done = threading.Event()
    if stop_event is not None:
      threading.Thread(
          target=_interrupt_on_stop, args=(session, stop_event, done),
          daemon=True).start()
    try:
      _play(session, stop_event)
    except EnetDisconnected:
      if stop_event is None or not stop_event.is_set():
        raise
      logging.info('Stopped while waiting for Dolphin.')
    finally:
      done.set()


def _interrupt_on_stop(session: Session, stop_event: StopEvent, done: threading.Event):
  while not done.is_set():
    if stop_event.wait(0.5):
      session.dolphin.interrupt()
      return


def _play(session: Session, stop_event: tp.Optional[StopEvent]):
  # Skip the first step, which may include compilation.
  total_step_time = 0.
  num_steps = 0

  for i, frame in enumerate(session.frames(stop_event)):
    if i > 0:
      total_step_time += frame.step_time
      num_steps += 1
    gamestate = frame.gamestate

    if gamestate.frame > 0 and gamestate.frame % (30 * 60) == 15 * 60:
      step_time = total_step_time / num_steps
      logging.info(f'step_time: {step_time:.3f}')
      if step_time > 0.016:
        logging.error('running too slow to keep up with the game!')
      elif step_time > 0.012:
        logging.warning('running slow, performance may be degraded')
