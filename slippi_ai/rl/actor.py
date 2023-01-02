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
      policy_states: tp.Mapping[Port, dict],
      env_kwargs: dict,
      num_steps_per_rollout: int,
  ):
    self._policies = {
        port: saving.load_policy_from_state(state)
        for port, state in policy_states.items()
    }

    env = env_lib.Environment(**env_kwargs)

    self._rollout_worker = evaluators.RolloutWorker(
        self._policies, env, num_steps_per_rollout)

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