import collections
import contextlib
import multiprocessing as mp
from multiprocessing.connection import Connection
import typing as tp
from typing import Mapping, Optional
from queue import Queue

import portpicker
import ray

from melee.slippstream import EnetDisconnected
from melee import GameState

from slippi_ai import dolphin, utils
from slippi_ai.controller_lib import send_controller
from slippi_ai.types import Controller, Game
from slippi_db.parse_libmelee import get_game

def is_initial_frame(gamestate: GameState) -> bool:
  return gamestate.frame == -123

class EnvOutput(tp.NamedTuple):
  gamestates: Mapping[int, Game]
  needs_reset: bool

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

    return EnvOutput(games, needs_reset)

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

  def multi_step(
    self,
    controllers: list[Mapping[int, Controller]],
  ) -> list[EnvOutput]:
    """Batched step to reduce communication overhead."""
    return [self.step(c) for c in controllers]

def safe_environment(
    dolphin_kwargs: dict,
    num_retries=2,
) -> Environment:
  """Create an environment, retrying with different ports on failure."""
  dolphin_kwargs = dolphin_kwargs.copy()

  def reset_port():
    dolphin_kwargs['slippi_port'] = portpicker.pick_unused_port()

  return utils.retry(
      lambda: Environment(dolphin_kwargs),
      on_exception={dolphin.ConnectFailed: reset_port},
      num_retries=num_retries)

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
      env = safe_environment(kwargs)
      envs.append(env)

    self._envs = envs

    # Optional "async" interface for compatibility with the Async* Envs.
    self._controller_queue = collections.deque()
    self._controller_queue.appendleft(None)  # denotes initial state

  @property
  def num_steps(self) -> int:
    return 1

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
    return safe_environment(env_kwargs)

  # BatchedEnvironment calls safe_environment internally
  return BatchedEnvironment(num_envs, env_kwargs, slippi_ports)

T = tp.TypeVar('T')

class SafeEnvironment:

  def __init__(self, build_env_kwargs: dict, num_retries: int = 2):
    self._build_env_kwargs = build_env_kwargs
    self._num_retries = num_retries
    self._env = build_environment(**self._build_env_kwargs)

  def _reset_env(self):
    self._env.stop()  # closes associated dolphin instances, freeing up ports
    self._env = build_environment(**self._build_env_kwargs)

  # def _retry(self, method: str, *args):
  #   return retry(
  #       lambda: getattr(self._env, method)(*args),
  #       on_exception={dolphin.ConnectFailed: self._reset_env},
  #       num_retries=self._num_retries)

  def _retry(self, f: tp.Callable[[], T]) -> T:
    return utils.retry(
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

  def stop(self):
    self._env.stop()

def _run_env(
    build_env_kwargs: dict,
    conn: Connection,
    batch_time: bool = False,
):
  env = None
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
  except KeyboardInterrupt:
    # exit quietly without spamming stderr
    return
  except Exception as e:
    conn.send(e)
    # conn.close()
  finally:
    if env:
      env.stop()

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
    if num_envs % inner_batch_size != 0:
      raise ValueError(
          f'num_envs={num_envs} must be divisible by '
          f'inner_batch_size={inner_batch_size}')

    self._total_batch_size = num_envs
    self._outer_batch_size = num_envs // inner_batch_size
    self._inner_batch_size = inner_batch_size
    self._slice = lambda i, x: x[i * inner_batch_size:(i + 1) * inner_batch_size]

    self._env_kwargs = dophin_kwargs

    self._envs: list[AsyncEnvMP] = []
    slippi_ports = get_free_ports(num_envs)
    for i in range(self._outer_batch_size):
      env = AsyncEnvMP(
          env_kwargs=dophin_kwargs,
          num_envs=inner_batch_size,
          batch_time=(num_steps > 0),
          slippi_ports=self._slice(i, slippi_ports),
      )
      self._envs.append(env)

    self._num_steps = num_steps
    self._action_queue: list[Mapping[int, Controller]] = []
    self._state_queue = collections.deque()
    self._num_in_transit = 1  # take into account initial state

  def qsize(self):
    return self._num_in_transit + len(self._state_queue)

  @property
  def num_steps(self) -> int:
    return self._num_steps or 1  # 0 means 1

  def stop(self):
    # First initiate stop asynchronously for all envs.
    for env in self._envs:
      env.begin_stop()

    # Then ensure all envs are stopped.
    for env in self._envs:
      env.ensure_stopped()

  def __del__(self):
    self.stop()

  @contextlib.contextmanager
  def run(self):
    try:
      yield self
    finally:
      print('AsyncBatchedEnvironmentMP.__exit__')
      self.stop()

  def _flush(self):
    # Returns a time-indexed list of controller dictionaries.
    get_action = lambda i: utils.map_single_structure(
        lambda x: self._slice(i, x), self._action_queue)

    for i, env in enumerate(self._envs):
      env.send(get_action(i))

    self._action_queue.clear()
    self._num_in_transit += self._num_steps

  def push(self, controllers: Mapping[int, Controller]):
    if self._num_steps == 0:
      get_action = lambda i: utils.map_single_structure(
          lambda x: self._slice(i, x), self._action_queue)
      for i, env in enumerate(self._envs):
        env.send(get_action(i))
      self._num_in_transit += 1
      return

    self._action_queue.append(controllers)

    if len(self._action_queue) == self._num_steps:
      self._flush()

  def _receive(self):
    if self._num_steps == 0:
      outputs = [env.recv() for env in self._envs]
      output = utils.concat_nest_nt(outputs)
      self._state_queue.appendleft(output)
      self._num_in_transit -= 1
    else:
      batch_major = [env.recv() for env in self._envs]
      time_major = zip(*batch_major)
      for batch in time_major:
        self._state_queue.appendleft(utils.concat_nest_nt(batch))
      self._num_in_transit -= self._num_steps

  def pop(self) -> EnvOutput:
    if not self._state_queue:
      self._receive()
    return self._state_queue.pop()

  def peek_n(self, n: int) -> list[EnvOutput]:
    while len(self._state_queue) < n:
      self._receive()
    return utils.peek_deque(self._state_queue, n)

  def peek(self) -> EnvOutput:
    if not self._state_queue:
      self._receive()
    return self._state_queue[-1]

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
