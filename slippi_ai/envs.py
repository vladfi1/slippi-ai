import itertools
import multiprocessing as mp
from typing import Mapping, Optional

import portpicker
import ray
from melee import GameState

from slippi_ai import dolphin, utils
from slippi_ai.eval_lib import send_controller
from slippi_ai.types import Controller, Game
from slippi_ai.reward import get_reward
from slippi_db.parse_libmelee import get_game

def is_initial_frame(gamestate: GameState) -> bool:
  return gamestate.frame == -123

class Environment:
  """Wraps dolphin to provide an RL interface."""

  def __init__(self, dolphin_kwargs: dict):
    self._dolphin = dolphin.Dolphin(**dolphin_kwargs)
    self.players = self._dolphin._players

    assert len(self.players) == 2

    self._opponents: Mapping[int, int] = {}
    ports = list(self.players)

    for port, opponent_port in zip(ports, reversed(ports)):
      if isinstance(self.players[port], dolphin.AI):
        self._opponents[port] = opponent_port

    self._prev_state: Optional[GameState] = None

  def current_state(self) -> tuple[Mapping[int, Game], bool]:
    if self._prev_state is None:
      self._prev_state = self._dolphin.step()

    needs_reset = is_initial_frame(self._prev_state)

    games = {}
    for port, opponent_port in self._opponents.items():
      games[port] = get_game(self._prev_state, (port, opponent_port))

    return games, needs_reset

  def step(
    self,
    controllers: Mapping[int, Controller],
  ) -> tuple[Mapping[int, Game], bool]:
    """Send controllers for each AI. Return the next state."""
    for port, controller in controllers.items():
      send_controller(self._dolphin.controllers[port], controller)

    # TODO: compute reward?
    self._prev_state = self._dolphin.step()
    return self.current_state()

    # results = {}

    # for port, opponent_port in self._opponents.items():
    #   game = get_game(gamestate, (port, opponent_port))
    #   # TODO: configure damage ratio
    #   reward = get_reward(
    #       self._prev_state, gamestate,
    #       own_port=port, opponent_port=opponent_port)
    #   results[port] = (game, reward)

    # self._prev_state = gamestate
    # return results

def get_free_ports(n: int) -> list[int]:
  ports = [portpicker.pick_unused_port() for _ in range(n)]
  if len(ports) < n:
    raise ValueError('Not enough free ports')
  return ports

class BatchedEnvironment:
  """A set of synchronous environments with batched input/output."""

  def __init__(self, num_envs: int, env_kwargs: dict):
    self._env_kwargs = env_kwargs

    envs: list[Environment] = []
    for slippi_port in get_free_ports(num_envs):
      kwargs = env_kwargs.copy()
      kwargs.update(slippi_port=slippi_port)
      env = Environment(kwargs)
      envs.append(env)

    self._envs = envs

  def current_state(self) -> tuple[Mapping[int, Game], bool]:
    current_states = [env.current_state() for env in self._envs]
    return utils.batch_nest_nt(current_states)

  def step(
    self,
    controllers: Mapping[int, Controller],
  ) -> tuple[Mapping[int, Game], bool]:
    get_action = lambda i: utils.map_nt(lambda x: x[i], controllers)

    results = [
        env.step(get_action(i))
        for i, env in enumerate(self._envs)
    ]
    return utils.batch_nest_nt(results)

# This would raise an annoying exception when Environment subclassed
# Dolphin due to ray erroneously calling __del__ on the wrong object.
# See https://github.com/ray-project/ray/issues/32952
RemoteEnvironment = ray.remote(Environment)

class AsyncBatchedEnvironment:
  """A set of asynchronous environments with batched input/output."""

  def __init__(self, num_envs: int, dophin_kwargs: dict):
    self._env_kwargs = dophin_kwargs

    envs = []
    for slippi_port in get_free_ports(num_envs):
      kwargs = dophin_kwargs.copy()
      kwargs.update(slippi_port=slippi_port)
      env = RemoteEnvironment.remote(kwargs)
      envs.append(env)

    self._envs = envs

  def current_state(self) -> tuple[Mapping[int, Game], bool]:
    current_states = ray.get(
        [env.current_state.remote() for env in self._envs])
    return utils.batch_nest_nt(current_states)

  def step(
    self,
    controllers: Mapping[int, Controller],
  ) -> tuple[Mapping[int, Game], bool]:
    get_action = lambda i: utils.map_nt(lambda x: x[i], controllers)

    results = ray.get([
        env.step.remote(get_action(i))
        for i, env in enumerate(self._envs)
    ])
    return utils.batch_nest_nt(results)

def run_env(init_kwargs, conn):
  env = Environment(init_kwargs)

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