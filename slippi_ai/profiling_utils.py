import atexit
import typing as tp

import psutil
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from slippi_ai import dolphin as dolphin_lib


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

    if cpu is not None:
      main_proc = psutil.Process()
      main_proc.cpu_affinity([cpu])

      for d in self._dolphins:
        proc = psutil.Process(d.console._process.pid)
        proc.cpu_affinity([cpu])

  def step(self):
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
