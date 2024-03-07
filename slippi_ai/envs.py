import collections
import multiprocessing as mp
from multiprocessing.connection import Connection
import typing as tp
from typing import Mapping, Optional
from queue import Queue
import logging

import portpicker
import ray

from melee.slippstream import EnetDisconnected
from melee import GameState

from slippi_ai import dolphin, utils
from slippi_ai.controller_lib import send_controller
from slippi_ai.types import Controller, Game
from slippi_ai.reward import get_reward
from slippi_db.parse_libmelee import get_game

def is_initial_frame(gamestate: GameState) -> bool:
  return gamestate.frame == -123

EnvOutput = tuple[Mapping[int, Game], bool]

class Environment:
  """Wraps dolphin to provide an RL interface."""

  def __init__(self, dolphin_kwargs: dict):
    # raise RuntimeError('test')

    self._dolphin = dolphin.Dolphin(**dolphin_kwargs)
    self.players = self._dolphin._players

    assert len(self.players) == 2

    self._opponents: Mapping[int, int] = {}
    ports = list(self.players)

    for port, opponent_port in zip(ports, reversed(ports)):
      if isinstance(self.players[port], dolphin.AI):
        self._opponents[port] = opponent_port

    self._prev_state: Optional[GameState] = None

  def stop(self):
    self._dolphin.stop()

  def current_state(self) -> EnvOutput:
    if self._prev_state is None:
      self._prev_state = self._dolphin.step()

    needs_reset = is_initial_frame(self._prev_state)

    games = {}
    for port, opponent_port in self._opponents.items():
      games[port] = get_game(self._prev_state, (port, opponent_port))

    return games, needs_reset

  def multi_current_state(self) -> list[EnvOutput]:
    return [self.current_state()]

  def step(
    self,
    controllers: Mapping[int, Controller],
    batch_index: Optional[int] = None,
  ) -> EnvOutput:
    """Send controllers for each AI. Return the next state."""

    if batch_index is not None:
      controllers = utils.map_single_structure(
          lambda x: x[batch_index], controllers)

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

  def multi_step(
    self,
    controllers: list[Mapping[int, Controller]],
  ) -> list[EnvOutput]:
    """Batched step to reduce communication overhead."""
    return [self.step(c) for c in controllers]

def get_free_ports(n: int) -> list[int]:
  ports = [portpicker.pick_unused_port() for _ in range(n)]
  if len(set(ports)) < n:
    raise ValueError('Not enough free ports')
  return ports

class BatchedEnvironment:
  """A set of synchronous environments with batched input/output."""

  def __init__(
      self,
      num_envs: int,
      env_kwargs: dict,
      slippi_ports: Optional[list[int]] = None,
      num_steps: int = 1,  # For compatibility
  ):
    del num_steps
    self._env_kwargs = env_kwargs

    slippi_ports = slippi_ports or get_free_ports(num_envs)
    envs: list[Environment] = []
    for slippi_port in slippi_ports:
      kwargs = env_kwargs.copy()
      kwargs.update(slippi_port=slippi_port)
      env = Environment(kwargs)
      envs.append(env)

    self._envs = envs

    # Optional "async" interface for compatibility with the Async* Envs.
    self._controller_queue = collections.deque()
    self._controller_queue.appendleft(None)  # initial state

  def stop(self):
    for env in self._envs:
      env.stop()

  def current_state(self) -> EnvOutput:
    current_states = [env.current_state() for env in self._envs]
    return utils.batch_nest_nt(current_states)

  def multi_current_state(self) -> list[EnvOutput]:
    return [self.current_state()]

  def step(
    self,
    controllers: Mapping[int, Controller],
  ) -> EnvOutput:
    get_action = lambda i: utils.map_single_structure(
        lambda x: x[i], controllers)

    results = [
        env.step(get_action(i))
        for i, env in enumerate(self._envs)
    ]
    return utils.batch_nest_nt(results)

  def multi_step(
    self,
    controllers: list[Mapping[int, Controller]],
  ) -> list[EnvOutput]:
    """Batched step to reduce communication overhead."""
    return [self.step(c) for c in controllers]

  def push(self, controllers: Mapping[int, Controller]):
    self._controller_queue.appendleft(controllers)

  def pop(self) -> EnvOutput:
    controllers = self._controller_queue.pop()
    if controllers is None:
      return self.current_state()
    return self.step(controllers)

def build_environment(
    num_envs: int,
    env_kwargs: dict,
    slippi_ports: Optional[list[int]] = None,
) -> tp.Union[Environment, BatchedEnvironment]:
  if num_envs == 0:
    if slippi_ports:
      assert len(slippi_ports) == 1
      env_kwargs = env_kwargs.copy()
      env_kwargs['slippi_port'] = slippi_ports[0]
    return Environment(env_kwargs)

  return BatchedEnvironment(num_envs, env_kwargs, slippi_ports)

T = tp.TypeVar('T')
E = tp.TypeVar('E', bound=Exception)

def retry(
    f: tp.Callable[[], T],
    on_exception: tp.Mapping[type, tp.Callable[[], tp.Any]],
    num_retries: int = 4,
) -> T:
  for _ in range(num_retries-1):
    try:
      return f()
    except tuple(on_exception) as e:
      logging.warning(f'Caught "{repr(e)}". Retrying...')
      on_exception[type(e)]()

  # Let any exception pass through on the last attempt.
  return f()

def safe_build_environment(
    build_env_kwargs: dict,
    num_retries=10,
) -> tp.Union[Environment, BatchedEnvironment]:
  return retry(
      lambda: build_environment(**build_env_kwargs),
      on_exception={dolphin.ConnectFailed: lambda: None},
      num_retries=num_retries)

class SafeEnvironment:

  def __init__(self, build_env_kwargs: dict, num_retries: int = 4):
    self._build_env_kwargs = build_env_kwargs
    self._num_retries = num_retries
    self._reset_env()

  def _reset_env(self):
    if hasattr(self, '_env'):
      self._env.stop()  # closes associated dolphin instances, freeing up ports
    self._env = safe_build_environment(
        self._build_env_kwargs, self._num_retries)

  # def _retry(self, method: str, *args):
  #   return retry(
  #       lambda: getattr(self._env, method)(*args),
  #       on_exception={dolphin.ConnectFailed: self._reset_env},
  #       num_retries=self._num_retries)

  def _retry(self, f: tp.Callable[[], T]) -> T:
    return retry(
        f,
        on_exception={EnetDisconnected: self._reset_env},
        num_retries=self._num_retries)

  def current_state(self) -> EnvOutput:
    return self._retry(lambda: self._env.current_state())

  def multi_current_state(self) -> list[EnvOutput]:
    return self._retry(lambda: self._env.multi_current_state())

  def step(
    self,
    controllers: Mapping[int, Controller],
  ) -> EnvOutput:
    return self._retry(lambda: self._env.step(controllers))

  def multi_step(
    self,
    controllers: list[Mapping[int, Controller]],
  ) -> list[EnvOutput]:
    return self._retry(lambda: self._env.multi_step(controllers))

def _run_env(
    build_env_kwargs: dict,
    conn: Connection,
    batch_time: bool = False,
):
  try:
    env = SafeEnvironment(build_env_kwargs)

    # Push initial env state.
    initial_state = env.current_state()
    if batch_time:
      initial_state = [initial_state]
    conn.send(initial_state)

    env_step = env.multi_step if batch_time else env.step

    while True:
      controllers = conn.recv()
      if controllers is None:
        break
      conn.send(env_step(controllers))

    # conn.close()
  except Exception as e:
    conn.send(e)
    # conn.close()
  except KeyboardInterrupt:
    # exit quietly without spamming stderr
    return

class EnvError(Exception):
  pass

class AsyncEnvMP:
  """An asynchronous environment using multiprocessing."""

  def __init__(
      self,
      env_kwargs: dict,
      num_envs: int = 0,
      slippi_ports: Optional[list[int]] = None,
      batch_time: bool = False,
  ):
    context = mp.get_context('forkserver')
    self._parent_conn, child_conn = context.Pipe()
    builder_kwargs = dict(
        num_envs=num_envs,
        env_kwargs=env_kwargs,
        slippi_ports=slippi_ports,
    )
    self._process = context.Process(
        target=_run_env,
        args=(builder_kwargs, child_conn),
        kwargs=dict(batch_time=batch_time))
    self._process.start()

  def stop(self):
    self.begin_stop()
    self.ensure_stopped()

  def begin_stop(self):
    """Non-blocking stop."""
    if self._process is not None and self._process.is_alive():
      self._parent_conn.send(None)

  def ensure_stopped(self):
    if self._process is not None:
      self._process.join()
      self._process.close()
      self._process = None
      # self._parent_conn.close()

  def __del__(self):
    self.stop()

  def send(self, controllers: Mapping[int, Controller]):
    self._parent_conn.send(controllers)

  def recv(self) -> EnvOutput:
    output = self._parent_conn.recv()
    if isinstance(output, Exception):
      # Maybe rebuild the environment and start over?
      raise output
    return output

class AsyncBatchedEnvironmentMP:
  """A set of asynchronous environments with batched input/output."""

  def __init__(
      self,
      num_envs: int,
      dophin_kwargs: dict,
      num_steps: int = 0,
      inner_batch_size: int = 1,
  ):
    self._env_kwargs = dophin_kwargs

    envs: list[AsyncEnvMP] = []
    for slippi_port in get_free_ports(num_envs):
      kwargs = dophin_kwargs.copy()
      kwargs.update(slippi_port=slippi_port)
      env = AsyncEnvMP(env_kwargs=kwargs, batch_time=(num_steps > 0))
      envs.append(env)

    self._envs = envs

    self._num_steps = num_steps
    self._action_queue: list[Mapping[int, Controller]] = []
    self._state_queue = collections.deque()

    # Break circular references...
    # self.push = functools.partial(
    #     self._static_push,
    #     envs=self._envs,
    #     action_queue=self._action_queue,
    #     num_steps=self._num_steps,
    # )

  def stop(self):
    # First initiate stop asynchronously for all envs.
    for env in self._envs:
      env.begin_stop()

    # Then ensure all envs are stopped.
    for env in self._envs:
      env.ensure_stopped()

  @staticmethod
  def _static_push(
      controllers: Mapping[int, Controller],
      envs: list[AsyncEnvMP],
      action_queue: list[Mapping[int, Controller]],
      num_steps: int,
  ):
    if num_steps == 0:
      get_action = lambda i: utils.map_single_structure(lambda x: x[i], controllers)
      for i, env in enumerate(envs):
        env.send(get_action(i))
      return

    action_queue.append(controllers)

    if len(action_queue) == num_steps:
      # Returns a time-indexed list of controller dictionaries.
      get_action = lambda i: utils.map_single_structure(lambda x: x[i], action_queue)

      for i, env in enumerate(envs):
        env.send(get_action(i))

      action_queue.clear()

  def _flush(self):
    # Returns a time-indexed list of controller dictionaries.
    get_action = lambda i: utils.map_single_structure(lambda x: x[i], self._action_queue)

    for i, env in enumerate(self._envs):
      env.send(get_action(i))

    self._action_queue.clear()

  def push(self, controllers: Mapping[int, Controller]):
    if self._num_steps == 0:
      get_action = lambda i: utils.map_single_structure(lambda x: x[i], controllers)
      for i, env in enumerate(self._envs):
        env.send(get_action(i))
      return

    self._action_queue.append(controllers)

    if len(self._action_queue) == self._num_steps:
      self._flush()

  def pop(self) -> EnvOutput:
    if self._num_steps == 0:
      outputs = [env.recv() for env in self._envs]
      return utils.batch_nest_nt(outputs)

    if not self._state_queue:
      batch_major = [env.recv() for env in self._envs]
      time_major = zip(*batch_major)
      for batch in time_major:
        self._state_queue.appendleft(utils.batch_nest_nt(batch))

    return self._state_queue.pop()

  def __del__(self):
    print(type(self), 'del')
    self.stop()

# This would raise an annoying exception when Environment subclassed
# Dolphin due to ray erroneously calling __del__ on the wrong object.
# See https://github.com/ray-project/ray/issues/32952
RemoteEnvironment = ray.remote(SafeEnvironment)

class AsyncBatchedEnvironmentRay:
  """A set of asynchronous environments with batched input/output."""

  def __init__(
      self,
      num_envs: int,
      dophin_kwargs: dict,
      num_steps: int = 1,
  ):
    self._env_kwargs = dophin_kwargs

    envs = []
    for slippi_port in get_free_ports(num_envs):
      env_kwargs = dophin_kwargs.copy()
      env_kwargs.update(slippi_port=slippi_port)
      build_kwargs = dict(
          num_envs=0,
          env_kwargs=env_kwargs,
      )
      env = RemoteEnvironment.remote(build_kwargs)
      envs.append(env)

    self._envs = envs

    self._num_steps = num_steps
    self._action_queue: list[Mapping[int, Controller]] = []
    self._state_queue = collections.deque()

    self._futures_queue = Queue()

    # Push initial states which are resetting.
    if self._num_steps == 0:
      self._futures_queue.put(
          [env.current_state.remote() for env in self._envs])
    else:
      self._futures_queue.put(
          [env.multi_current_state.remote() for env in self._envs])

  def _flush(self):
    # Returns a time-indexed list of controller dictionaries.
    get_action = lambda i: utils.map_single_structure(lambda x: x[i], self._action_queue)

    self._futures_queue.put([
        env.multi_step.remote(get_action(i))
        for i, env in enumerate(self._envs)])

    self._action_queue.clear()

  def push(self, controllers: Mapping[int, Controller]):
    if self._num_steps == 0:
      controllers_ref = ray.put(controllers)
      self._futures_queue.put([
          env.step.remote(controllers_ref, batch_index=i)
          for i, env in enumerate(self._envs)])
      # get_action = lambda i: utils.map_single_structure(lambda x: x[i], controllers)
      # self._futures_queue.put([
      #     env.step.remote(get_action(i))
      #     for i, env in enumerate(self._envs)])
      return

    self._action_queue.append(controllers)

    if len(self._action_queue) == self._num_steps:
      self._flush()

  def pop(self, timeout: Optional[float] = None) -> EnvOutput:
    if self._num_steps == 0:
      futures = self._futures_queue.get(timeout=timeout)
      batch = ray.get(futures, timeout=timeout)
      return utils.batch_nest_nt(batch)

    if not self._state_queue:
      futures = self._futures_queue.get(timeout=timeout)
      batch_major = ray.get(futures, timeout=timeout)
      time_major = zip(*batch_major)
      for batch in time_major:
        self._state_queue.appendleft(utils.batch_nest_nt(batch))

    return self._state_queue.pop()
