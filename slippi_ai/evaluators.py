"""Evaluates a policy."""

import threading
import typing as tp

import numpy as np

import melee

from slippi_ai.policies import Policy
from slippi_ai import (
    embed,
    saving,
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
    policies: tp.Mapping[Port, Policy],
    configs: tp.Mapping[Port, dict],
    env: dolphin.Dolphin,  # TODO: support multiple envs
    num_steps_per_rollout: int,
    # compile: bool = True,
  ) -> None:
    players = env._players
    assert len(players) == 2

    # maps port to opponent port
    opponents = dict(zip(players, reversed(players)))
    ai_ports = [
        port for port, player in players.items()
        if isinstance(player, dolphin.AI)]
    assert set(policies) == set(ai_ports)

    self._policies = policies
    self._env = env
    self._num_steps_per_rollout = num_steps_per_rollout

    self._agents = {
        port: eval_lib.Agent(
            controller=env.controllers[port],
            opponent_port=opponents[port],
            policy=policy,
            # TODO: configure compile
            config=configs[port],
            # This is redundant.
            embed_controller=policy.controller_embedding,
        )
        for port, policy in policies.items()
    }

    self._last_gamestate: tp.Optional[melee.GameState] = None

  def rollout(self) -> tp.Mapping[Port, Trajectory]:
    if self._last_gamestate is None:
      self._last_gamestate = self._env.step()

    gamestate = self._last_gamestate
    state_actions: dict[Port, list[embed.StateAction]] = {
        port: [] for port in self._policies
    }
    is_resetting: list[bool] = []

    for _ in range(self._num_steps_per_rollout):
      is_resetting.append(gamestate.frame == -123)
      for port, agent in self._agents.items():
        prev_action = agent._prev_controller
        state = agent.step(gamestate).state
        state_actions[port].append(embed.StateAction(state, prev_action))

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

    return trajectories

  def update_variables(
      self, updates: tp.Mapping[Port, tp.Sequence[np.ndarray]],
  ):
    for port, values in updates.items():
      for var, val in zip(self._policies[port].variables, values):
        var.assign(val)

class RolloutMetrics(tp.NamedTuple):
  reward: float

  @classmethod
  def from_trajectory(cls, trajectory: Trajectory) -> 'RolloutMetrics':
    return cls(reward=np.sum(trajectory.frames.reward))

class RemoteEvaluator:

  def __init__(
      self,
      policy_configs: tp.Mapping[Port, dict],
      dolphin_kwargs: dict,
      num_steps_per_rollout: int,
  ):
    policies = {
        port: saving.policy_from_config(config)
        for port, config in policy_configs.items()
    }

    env = dolphin.Dolphin(**dolphin_kwargs)

    self._rollout_worker = RolloutWorker(
        policies, policy_configs, env, num_steps_per_rollout)

    self._lock = threading.Lock()

  def rollout(
      self,
      policy_vars: tp.Mapping[Port, tp.Sequence[np.ndarray]],
  ) -> tp.Mapping[Port, RolloutMetrics]:
    with self._lock:
      self._rollout_worker.update_variables(policy_vars)
      trajectories = self._rollout_worker.rollout()
      return {
          port: RolloutMetrics.from_trajectory(trajectory)
          for port, trajectory in trajectories.items()
      }
