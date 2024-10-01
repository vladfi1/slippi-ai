import collections
import contextlib
import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
import traceback
import typing as tp
from typing import Mapping, Optional

import numpy as np
import portpicker

from melee.slippstream import EnetDisconnected
from melee import GameState, Stage

from slippi_ai import dolphin, utils
from slippi_ai.controller_lib import send_controller
from slippi_ai.types import Controller, Game
from slippi_db.parse_libmelee import get_game

Port = int
Controllers = Mapping[Port, Controller]

def is_initial_frame(gamestate: GameState) -> bool:
  return gamestate.frame == -123

class EnvOutput(tp.NamedTuple):
  gamestates: Mapping[int, Game]
  needs_reset: bool

class Environment:
  """Wraps dolphin to provide an RL interface."""

  def __init__(
      self,
      dolphin_kwargs: dict,
      check_controller_outputs: bool = False,
  ):
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
    controllers: Controllers,
  ) -> EnvOutput:
    """Send controllers for each AI. Return the next state."""

    for port, controller in controllers.items():
      send_controller(self._dolphin.controllers[port], controller)

    # TODO: compute reward?
    self._prev_state = self._dolphin.step()
    return self.current_state()

  def multi_step(
    self,
    controllers: list[Controllers],
  ) -> list[EnvOutput]:
    """Batched step to reduce communication overhead."""
    return [self.step(c) for c in controllers]

T = tp.TypeVar('T')


class SafeEnvironment:
  """Wraps an environment with retries on disconnect."""

  def __init__(self, dolphin_kwargs: dict, num_retries: int = 2):
    self._dolphin_kwargs = dolphin_kwargs.copy()
    self._num_retries = num_retries
    self._build_environment()

  def _reset_port(self):
    old_port = self._dolphin_kwargs['slippi_port']
    new_port = portpicker.pick_unused_port()
    logging.warning('Switching from port %d to port %d.', old_port, new_port)
    self._dolphin_kwargs['slippi_port'] = new_port

  def _build_environment(self):
    self._env = utils.retry(
        lambda: Environment(self._dolphin_kwargs),
        on_exception={dolphin.ConnectFailed: self._reset_port},
        num_retries=2)

  def _reset_env(self):
    self._env.stop()  # closes associated dolphin instances, freeing up ports
    self._build_environment()

  # def _retry(self, method: str, *args):
  #   return retry(
  #       lambda: getattr(self._env, method)(*args),
  #       on_exception={dolphin.ConnectFailed: self._reset_env},
  #       num_retries=self._num_retries)

  def _retry(self, f: tp.Callable[[], T]) -> T:
    return utils.retry(
        f,
        on_exception={
            EnetDisconnected: self._reset_env,
            TimeoutError: self._reset_env,
        },
        num_retries=self._num_retries)

  def current_state(self) -> EnvOutput:
    return self._retry(lambda: self._env.current_state())

  def multi_current_state(self) -> list[EnvOutput]:
    return self._retry(lambda: self._env.multi_current_state())

  def step(
    self,
    controllers: Controllers,
  ) -> EnvOutput:
    return self._retry(lambda: self._env.step(controllers))

  def multi_step(
    self,
    controllers: list[Controllers],
  ) -> list[EnvOutput]:
    return self._retry(lambda: self._env.multi_step(controllers))

  def stop(self):
    self._env.stop()


class BatchedEnvironment:
  """A set of synchronous environments with batched input/output."""

  def __init__(
      self,
      num_envs: int,
      dolphin_kwargs: dict,
      slippi_ports: Optional[list[int]] = None,
      num_retries: int = 2,
  ):
    self._dolphin_kwargs = dolphin_kwargs

    slippi_ports = slippi_ports or utils.find_open_udp_ports(num_envs)
    envs: list[SafeEnvironment] = []
    for slippi_port in slippi_ports:
      kwargs = dolphin_kwargs.copy()
      kwargs.update(slippi_port=slippi_port)
      env = SafeEnvironment(kwargs, num_retries=num_retries)
      envs.append(env)

    self._envs = envs

    # Optional "async" interface for compatibility with the Async* Envs.
    self._output_queue = collections.deque()
    self._output_queue.appendleft(self.current_state())

  @property
  def num_steps(self) -> int:
    return 1

  def stop(self):
    for env in self._envs:
      env.stop()

  @contextlib.contextmanager
  def run(self):
    try:
      yield self
    finally:
      self.stop()

  def current_state(self) -> EnvOutput:
    current_states = [env.current_state() for env in self._envs]
    return utils.batch_nest_nt(current_states)

  def multi_current_state(self) -> list[EnvOutput]:
    return [self.current_state()]

  def step(
    self,
    controllers: Controllers,
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
    controllers: list[Controllers],
  ) -> list[EnvOutput]:
    """Batched step to reduce communication overhead."""
    return [self.step(c) for c in controllers]

  def push(self, controllers: Controllers):
    self._output_queue.appendleft(self.step(controllers))

  def pop(self) -> EnvOutput:
    return self._output_queue.pop()

  def peek(self) -> EnvOutput:
    return self._output_queue[-1]

def build_environment(
    num_envs: int,
    dolphin_kwargs: dict,
    slippi_ports: Optional[list[int]] = None,
    num_retries: int = 2,
) -> tp.Union[SafeEnvironment, BatchedEnvironment]:
  if num_envs == 0:
    if slippi_ports:
      assert len(slippi_ports) == 1
      dolphin_kwargs = dolphin_kwargs.copy()
      dolphin_kwargs['slippi_port'] = slippi_ports[0]
    return SafeEnvironment(dolphin_kwargs, num_retries=num_retries)

  # BatchedEnvironment uses SafeEnvironment internally
  return BatchedEnvironment(num_envs, dolphin_kwargs, slippi_ports, num_retries)

def _run_env(
    build_env_kwargs: dict,
    conn: Connection,
    # output_queue: mp.Queue,
    # stop: Event,
    batch_time: bool = False,
):
  send = conn.send
  # send = output_queue.put

  env = None
  try:
    env = build_environment(**build_env_kwargs)

    # Push initial env state.
    initial_state = env.current_state()
    if batch_time:
      initial_state = [initial_state]
    send(initial_state)

    env_step = env.multi_step if batch_time else env.step

    while True:
      controllers = conn.recv()
      if controllers is None:
        send(None)  # signal end of outputs
        return
      send(env_step(controllers))

    # conn.close()
  except KeyboardInterrupt:
    # exit quietly without spamming stderr
    return
  except BrokenPipeError:
    # The other end closed the connection.
    return
  except Exception:
    send(EnvError(traceback.format_exc()))
    send(None)  # signal end of outputs
  finally:
    if env:
      env.stop()

class EnvError(Exception):
  pass

class AsyncEnvMP:
  """An asynchronous environment using multiprocessing."""

  def __init__(
      self,
      dolphin_kwargs: dict,
      num_envs: int = 0,  # zero means non-batched env
      slippi_ports: Optional[list[int]] = None,
      num_retries: int = 2,
      batch_time: bool = False,
  ):
    context = mp.get_context('forkserver')
    self._parent_conn, child_conn = context.Pipe()
    self._recv = self._parent_conn.recv

    builder_kwargs = dict(
        num_envs=num_envs,
        dolphin_kwargs=dolphin_kwargs,
        slippi_ports=slippi_ports,
        num_retries=num_retries,
    )
    # self._stop = mp.Event()
    self._process = context.Process(
        name=f'_run_env({slippi_ports})',
        target=_run_env,
        args=(builder_kwargs, child_conn),
        kwargs=dict(batch_time=batch_time))
    self._process.start()

  def stop(self):
    self.begin_stop()
    self.ensure_stopped()

  def begin_stop(self):
    """Non-blocking stop."""
    if self._process is not None:
      # self._stop.set()
      try:
        self._parent_conn.send(None)
      except BrokenPipeError:
        pass

  def ensure_stopped(self):
    if self._process is None:
      return

    # The _run_env process might be blocked on pushing data into the Pipe.
    # To unblock it, we pull all the pending data from the pipe.
    while self._process.is_alive():
      # _run_env pushes None to signal execution has finished.
      if self._parent_conn.poll(1) and self._parent_conn.recv() is None:
        break

    # logging.info('Joining process %d', self._process.pid)
    self._process.join()
    self._process.close()
    self._process = None

  # def __del__(self):
  #   self.stop()

  def send(self, controllers: tp.Union[Controllers, list[Controllers]]):
    try:
      self._parent_conn.send(controllers)
    except BrokenPipeError:
      # TODO: attempt to retrieve exception from pipe?
      raise EnvError("run_env process died")

  def recv(self) -> EnvOutput:
    # TODO: ensure that enough data has been pushed?
    output = self._recv()
    if isinstance(output, Exception):
      # Maybe rebuild the environment and start over?
      raise output
    return output

class AsyncBatchedEnvironmentMP:
  """A set of asynchronous environments with batched input/output."""

  def __init__(
      self,
      num_envs: int,
      dolphin_kwargs: dict,
      num_steps: int = 0,
      inner_batch_size: int = 1,
      num_retries: int = 2,
  ):
    if num_envs % inner_batch_size != 0:
      raise ValueError(
          f'num_envs={num_envs} must be divisible by '
          f'inner_batch_size={inner_batch_size}')

    self._total_batch_size = num_envs
    self._outer_batch_size = num_envs // inner_batch_size
    self._inner_batch_size = inner_batch_size
    self._slice = lambda i, x: x[i * inner_batch_size:(i + 1) * inner_batch_size]

    self._dolphin_kwargs = dolphin_kwargs

    self._envs: list[AsyncEnvMP] = []
    slippi_ports = utils.find_open_udp_ports(num_envs)
    for i in range(self._outer_batch_size):
      env = AsyncEnvMP(
          dolphin_kwargs=dolphin_kwargs,
          num_envs=inner_batch_size,
          batch_time=(num_steps > 0),
          slippi_ports=self._slice(i, slippi_ports),
          num_retries=num_retries,
      )
      self._envs.append(env)

    self._num_steps = num_steps
    self._action_queue: list[Controllers] = []
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
      self.stop()

  def _flush(self):
    # Returns a time-indexed list of controller dictionaries.
    get_action = lambda i: utils.map_single_structure(
        lambda x: self._slice(i, x), self._action_queue)

    for i, env in enumerate(self._envs):
      env.send(get_action(i))

    self._num_in_transit += len(self._action_queue)
    self._action_queue.clear()

  def push(self, controllers: Controllers):
    if self._num_steps == 0:
      get_action = lambda i: utils.map_single_structure(
          lambda x: self._slice(i, x), controllers)
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
        self._num_in_transit -= 1

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

reified_game = utils.reify_tuple_type(Game)

class FakeBatchedEnvironment:
  def __init__(
      self,
      num_envs: int,
      players: tp.Collection[int],
  ):
    game = utils.map_nt(
        lambda t: np.full([num_envs], 0, dtype=t), reified_game)
    game.stage[:] = Stage.FINAL_DESTINATION.value  # make the stage valid
    self._output = EnvOutput(
        gamestates={p: game for p in players},
        needs_reset=np.full([num_envs], False),
    )
    self.num_steps = 0

  def stop(self):
    pass

  def pop(self) -> EnvOutput:
    return self._output

  def push(self, controllers: Controllers):
    del controllers

  def step(self, controllers: Controllers):
    del controllers
    return self._output

  def multi_step(
    self,
    controllers: list[Controllers],
  ) -> list[EnvOutput]:
    return [self._output] * len(controllers)

  def peek(self) -> EnvOutput:
    return self._output
