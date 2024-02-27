"""Evaluates a policy."""

import threading
import typing as tp

import numpy as np

from slippi_ai import envs as env_lib
from slippi_ai import (
    embed,
    data,
    eval_lib,
    reward,
    utils,
)

Port = int

# Mimics data.Batch
class Trajectory(tp.NamedTuple):
  frames: data.Frames
  is_resetting: bool
  # initial_state: policies.RecurrentState

class RolloutWorker:

  def __init__(
    self,
    agents: tp.Mapping[Port, eval_lib.RawAgent],
    env: env_lib.Environment,  # TODO: support multiple envs
    num_steps_per_rollout: int,
  ) -> None:
    self._env = env
    self._num_steps_per_rollout = num_steps_per_rollout

    self._agents = agents

  def rollout(self) -> tuple[tp.Mapping[Port, Trajectory], dict]:
    gamestates, needs_reset = self._env.current_state()
    state_actions: dict[Port, list[embed.StateAction]] = {
        port: [] for port in self._agents
    }
    is_resetting: list[bool] = []

    step_profiler = utils.Profiler()
    # Maybe use separate profilers for each agent?
    agent_profiler = utils.Profiler()

    for _ in range(self._num_steps_per_rollout):
      is_resetting.append(needs_reset)
      with agent_profiler:
        actions: dict[Port, embed.Action] = {}
        for port, agent in self._agents.items():
          game = gamestates[port]
          state_action = embed.StateAction(
              state=game,
              action=agent._prev_controller,
          )
          state_actions[port].append(state_action)
          actions[port] = agent.step_unbatched(game, needs_reset)

      with step_profiler:
        gamestates, needs_reset = self._env.step(actions)

    # TODO: overlap trajectories by one frame
    is_resetting = np.array(is_resetting)

    trajectories = {}
    for port, state_action_list in state_actions.items():
      state_action = utils.batch_nest_nt(state_action_list)
      rewards = reward.compute_rewards(state_action.state)
      frames = data.Frames(state_action=state_action, reward=rewards)
      trajectories[port] = Trajectory(frames, is_resetting)

    timings = {
        'step': step_profiler.mean_time(),
        'agent': agent_profiler.mean_time(),
    }

    return trajectories, timings

  def update_variables(
      self, updates: tp.Mapping[Port, tp.Sequence[np.ndarray]],
  ):
    for port, values in updates.items():
      policy = self._agents[port]._policy
      for var, val in zip(policy.variables, values):
        var.assign(val)

class RolloutMetrics(tp.NamedTuple):
  reward: float

  @classmethod
  def from_trajectory(cls, trajectory: Trajectory) -> 'RolloutMetrics':
    return cls(reward=np.sum(trajectory.frames.reward))

class RemoteEvaluator:

  def __init__(
      self,
      agent_kwargs: tp.Mapping[Port, dict],
      env_kwargs: dict,
      num_steps_per_rollout: int,
      use_gpu: bool = False,
  ):
    if not use_gpu:
      eval_lib.disable_gpus()

    env = env_lib.Environment(**env_kwargs)
    opponents = env._opponents
    assert set(agent_kwargs) == set(opponents)

    agents = {
        port: eval_lib.build_raw_agent(
            console_delay=env_kwargs['online_delay'],
            batch_size=1,
            **kwargs,
        )
        for port, kwargs in agent_kwargs.items()
    }

    for port, kwargs in agent_kwargs.items():
      eval_lib.update_character(
          env._players[port], kwargs['state']['config'])

    self._rollout_worker = RolloutWorker(
        agents, env, num_steps_per_rollout)

    self._lock = threading.Lock()

  def rollout(
      self,
      policy_vars: tp.Mapping[Port, tp.Sequence[np.ndarray]],
  ) -> tuple[tp.Mapping[Port, RolloutMetrics], dict]:
    with self._lock:
      self._rollout_worker.update_variables(policy_vars)
      trajectories, timings = self._rollout_worker.rollout()
      metrics = {
          port: RolloutMetrics.from_trajectory(trajectory)
          for port, trajectory in trajectories.items()
      }
      return metrics, timings
