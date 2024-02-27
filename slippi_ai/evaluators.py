"""Evaluates a policy."""

import threading
import typing as tp

import numpy as np

import melee

from slippi_ai import (
    embed,
    data,
    eval_lib,
    dolphin,
    reward,
    utils,
)

Port = int

class Trajectory(tp.NamedTuple):
  frames: data.Frames
  is_resetting: bool
  # initial_state: policies.RecurrentState

class RolloutWorker:

  def __init__(
    self,
    agents: tp.Mapping[Port, eval_lib.Agent],
    env: dolphin.Dolphin,  # TODO: support multiple envs
    num_steps_per_rollout: int,
  ) -> None:
    self._env = env
    self._num_steps_per_rollout = num_steps_per_rollout

    self._agents = agents
    self._last_gamestate: tp.Optional[melee.GameState] = None

  def rollout(self) -> tuple[tp.Mapping[Port, Trajectory], dict]:
    if self._last_gamestate is None:
      self._last_gamestate = self._env.step()

    gamestate = self._last_gamestate
    state_actions: dict[Port, list[embed.StateAction]] = {
        port: [] for port in self._agents
    }
    is_resetting: list[bool] = []

    step_profiler = utils.Profiler()
    # Maybe use separate profilers for each agent?
    agent_profiler = utils.Profiler()

    for _ in range(self._num_steps_per_rollout):
      is_resetting.append(gamestate.frame == -123)
      with agent_profiler:
        for port, agent in self._agents.items():
          prev_action = agent._prev_controller
          state = agent.step(gamestate).state
          state_actions[port].append(embed.StateAction(state, prev_action))

      with step_profiler:
        gamestate = self._env.step()

    # TODO: overlap trajectories by one frame
    self._last_gamestate = gamestate
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
      dolphin_kwargs: dict,
      num_steps_per_rollout: int,
      use_gpu: bool = False,
  ):
    if not use_gpu:
      eval_lib.disable_gpus()

    env = dolphin.Dolphin(**dolphin_kwargs)

    players = env._players
    assert len(players) == 2

    # maps port to opponent port
    opponents = dict(zip(players, reversed(players)))
    ai_ports = [
        port for port, player in players.items()
        if isinstance(player, dolphin.AI)]
    assert set(agent_kwargs) == set(ai_ports)

    agents = {
        port: eval_lib.build_agent(
            controller=env.controllers[port],
            opponent_port=opponents[port],
            console_delay=dolphin_kwargs['online_delay'],
            **kwargs,
        )
        for port, kwargs in agent_kwargs.items()
    }

    for port, agent in agents.items():
      eval_lib.update_character(players[port], agent.config)

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
