import multiprocessing as mp
from typing import Mapping, Tuple

from slippi_ai import dolphin

from slippi_ai.eval_lib import send_controller
from slippi_ai.types import Controller, Game
from slippi_ai.reward import get_reward
from slippi_db.parse_libmelee import get_game

class Environment(dolphin.Dolphin):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    assert len(self._players) == 2

    self._opponents: Mapping[int, int] = {}
    ports = list(self._players)

    for port, opponent_port in zip(ports, reversed(ports)):
      if isinstance(self._players[port], dolphin.AI):
        self._opponents[port] = opponent_port

    # Note that this may do some menuing.
    self._prev_state = super().step()

  def current_state(self) -> Mapping[int, Game]:
    results = {}
    for port, opponent_port in self._opponents.items():
      results[port] = get_game(self._prev_state, (port, opponent_port))
    return results

  def step(
    self,
    controllers: Mapping[int, Controller],
  ) -> Mapping[int, Tuple[Game, float]]:
    """Send controllers for each AI. Return the next state and reward."""
    for port, controller in controllers.items():
      send_controller(self.controllers[port], controller)

    gamestate = super().step()

    results = {}

    for port, opponent_port in self._opponents.items():
      game = get_game(gamestate, (port, opponent_port))
      reward = get_reward(
          self._prev_state, gamestate,
          own_port=port, opponent_port=opponent_port)
      results[port] = (game, reward)  # Frames?

    # TODO: check whether the game was over.
    self._prev_state = gamestate
    return results


def run_env(init_kwargs, conn):
  env = Environment(**init_kwargs)

  while True:
    controllers = conn.recv()
    if controllers is None:
      break

    conn.send(env.step(controllers))

  env.stop()
  conn.close()

class AsyncEnv:

  def __init__(self, **kwargs):
    self._parent_conn, child_conn = mp.Pipe()
    self._process = mp.Process(target=run_env, args=(kwargs, child_conn))
    self._process.start()

  def stop(self):
    self._parent_conn.send(None)
    self._process.join()
    self._parent_conn.close()

  def send(self, controllers):
    self._parent_conn.send(controllers)

  def recv(self):
    return self._parent_conn.recv()