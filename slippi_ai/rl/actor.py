import typing as tp
import tree

import numpy as np

from slippi_ai import (
    embed,
    evaluators,
    saving,
    utils,
)
from slippi_ai import envs as env_lib
from slippi_ai.networks import RecurrentState

Port = int

class Trajectory(tp.NamedTuple):
  initial_state: RecurrentState
  observations: embed.StateActionReward  # column-major

def from_row_major(trajectory: evaluators.Trajectory) -> Trajectory:
  return Trajectory(
      initial_state=trajectory.initial_state,
      observations=utils.batch_nest(trajectory.observations),
  )

class RolloutResults(tp.NamedTuple):
  timings: evaluators.RolloutTiming
  trajectories: tp.Mapping[Port, Trajectory]

class Actor:

  def __init__(
      self,
      agent_kwargs: tp.Mapping[Port, dict],
      dolphin_kwargs: dict,
      num_envs: int,
      async_envs: bool = False,
      ray_envs: bool = False,
      env_kwargs: dict = {},
      async_inference: bool = False,
      use_gpu: bool = False,
  ):
    self._rollout_worker = evaluators.RolloutWorker(
        agent_kwargs=agent_kwargs,
        dolphin_kwargs=dolphin_kwargs,
        num_envs=num_envs,
        async_envs=async_envs,
        ray_envs=ray_envs,
        env_kwargs=env_kwargs,
        async_inference=async_inference,
        use_gpu=use_gpu,
    )

  def rollout(
      self,
      policy_vars: tp.Mapping[Port, tp.Sequence[np.ndarray]],
  ) -> RolloutResults:
    self._rollout_worker.update_variables(policy_vars)
    trajectories, timings = self._rollout_worker.rollout()
    trajectories = {
        port: from_row_major(trajectory)
        for port, trajectory in trajectories.items()}
    return RolloutResults(timings, trajectories)

  def stop(self):
    self._rollout_worker.stop()