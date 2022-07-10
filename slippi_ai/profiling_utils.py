import atexit
import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
import time
import typing as tp

import psutil
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

import melee
from slippi_ai import dolphin as dolphin_lib

def all_children(proc: psutil.Process) -> tp.Iterator[psutil.Process]:
  yield proc
  for child in proc.children():
    yield from all_children(child)
  for pthread in proc.threads():
    yield psutil.Process(pthread.id)

class SerialEnv:

  def __init__(
      self,
      n: int,
      dolphin_kwargs: dict,
      cpu: tp.Optional[int] = 0,
  ):
    players = {1: dolphin_lib.AI(), 2: dolphin_lib.CPU()}

    self._dolphins = [
        dolphin_lib.Dolphin(players=players, **dolphin_kwargs)
        for _ in range(n)]

    time.sleep(1)

    if cpu is not None:
      logging.info(f'Setting cpu affinity to [{cpu}].')
      for proc in all_children(psutil.Process()):
        proc.cpu_affinity([cpu])

      # main_proc = psutil.Process()
      # main_proc.cpu_affinity([cpu])

      # for d in self._dolphins:
      #   proc = psutil.Process(d.console._process.pid)
      #   proc.cpu_affinity([cpu])

      #   for pthread in proc.threads():
      #     logging.info(pthread.id)
      #     psutil.Process(pthread.id).cpu_affinity([cpu])

  def step(self):
    # start_time = time.perf_counter()
    # while time.perf_counter() - start_time < 100:
    #   [d.step() for d in self._dolphins]

    # intentionally serial
    return [d.step() for d in self._dolphins]

  def stop(self):
    for d in self._dolphins:
      d.stop()

RemoteSerialEnv = ray.remote(SerialEnv)

class RayMultiSerialEnv:

  def __init__(
      self,
      n: int,
      cpus: int,
      set_affinity: bool,
      dolphin_kwargs: dict,
  ):
    self._envs = []
    make_env = RemoteSerialEnv.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().node_id,
            soft=False,
        )
    ).remote

    for i in range(cpus):
      env = make_env(
        n=n, dolphin_kwargs=dolphin_kwargs,
        cpu=i if set_affinity else None)
      self._envs.append(env)
    atexit.register(self.stop)

  def step(self):
    steps = ray.get([env.step.remote() for env in self._envs])
    states = []
    for step in steps:
      states.extend(step)
    return states

  def stop(self):
    ray.wait([env.stop.remote() for env in self._envs])

_STEP = 'step'
_STOP = 'stop'

def _run_serial_env(conn: Connection, init_kwargs):
  env = SerialEnv(**init_kwargs)

  while True:
    cmd = conn.recv()
    if cmd == _STEP:
      conn.send(env.step())
    elif cmd == _STOP:
      env.stop()
      break
    else:
      raise ValueError(f'Invalid command {cmd}')


class AsyncSerialEnv:

  def __init__(self, **kwargs):
    self._conn, child_conn = mp.Pipe()
    self._process = mp.Process(
        target=_run_serial_env,
        args=(child_conn, kwargs),
    )
    self._process.start()

  def step(self):
    self._conn.send(_STEP)

  def get(self) -> tp.List[melee.GameState]:
    return self._conn.recv()

  def stop(self):
    if not self._conn.closed:
      self._conn.send(_STOP)
      self._process.join()
      self._conn.close()

class MultiSerialEnv:
  def __init__(
      self,
      n: int,
      cpus: int,
      set_affinity: bool,
      dolphin_kwargs: dict,
  ):
    self._envs: tp.List[AsyncSerialEnv] = []

    for i in range(cpus):
      env = AsyncSerialEnv(
        n=n, dolphin_kwargs=dolphin_kwargs,
        cpu=i if set_affinity else None)
      self._envs.append(env)

    atexit.register(self.stop)

  def step(self) -> tp.List[melee.GameState]:
    for env in self._envs:
      env.step()

    gamestates = []
    for env in self._envs:
      gamestates.extend(env.get())

    return gamestates

  def stop(self):
    for env in self._envs:
      env.stop()
