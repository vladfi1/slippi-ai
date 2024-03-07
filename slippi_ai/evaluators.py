"""Evaluates a policy."""

import collections
import threading
import typing as tp
import queue
import logging
import cProfile

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
Timings = dict

# Mimics data.Batch
class Trajectory(tp.NamedTuple):
  frames: data.Frames
  is_resetting: bool
  # initial_state: policies.RecurrentState

def feed_environments(
    ports: list[Port],
    get_action: tp.Callable[[Port], embed.Game],
    push_actions: tp.Callable[[tp.Mapping[Port, embed.Game]], None],
    # env_profiler: cProfile.Profile,
    push_profiler: utils.Profiler,
):
  # TODO: run only for a fixed number of steps.
  while True:
    actions = {}
    for port in ports:
      actions[port] = get_action(port)

      # Agents output None once they're done.
      if actions[port] is None:
        return

    try:
      with push_profiler:
        push_actions(actions)
    except:
      logging.warn('Error pushing actions, shutting down env feeder thread.')
      return

class RolloutWorker:

  def __init__(
    self,
    agents: tp.Mapping[Port, eval_lib.AsyncDelayedAgent],
    batched_env: env_lib.AsyncBatchedEnvironmentMP,
    num_steps_per_rollout: int,
    damage_ratio: float = 0,
  ) -> None:
    self._env = batched_env
    self._num_steps_per_rollout = num_steps_per_rollout
    self._agents = agents
    self._damage_ratio = damage_ratio

    self._env_push_profiler = cProfile.Profile()
    self._env_push_profiler = utils.Profiler()

    agent_profilers = {
        port: utils.Profiler() for port in agents}
    # Careful to break circular references.
    self._agent_profilers = agent_profilers
    get_actions = {port: agents[port].pop for port in agents}

    def get_action(i):
      with agent_profilers[i]:
        return get_actions[i]()

    self._env_feeder = threading.Thread(
        target=feed_environments,
        kwargs=dict(
            ports=list(agents),
            get_action=get_action,
            push_actions=batched_env.push,
            push_profiler=self._env_push_profiler,
        ))
    assert not utils.ref_path_exists([self._env_feeder], [self])
    self._env_feeder.start()

  def rollout(self, num_steps: int) -> tuple[tp.Mapping[Port, Trajectory], Timings]:
    state_actions: dict[Port, list[embed.StateAction]] = {
        port: [] for port in self._agents
    }
    is_resetting: list[bool] = []

    step_profiler = utils.Profiler()
    # Maybe use separate profilers for each agent?
    # agent_profiler = utils.Profiler()

    def record_state(port: Port, game: embed.Game):
      state_action = embed.StateAction(
          state=game,
          # TODO: use actual controller from agent instead?
          action=game.p0.controller,
          name=self._agents[port]._agent._name_code,
      )
      state_actions[port].append(state_action)

    agent_state_queue_sizes = {port: [] for port in self._agents}
    env_queue_sizes = []

    for _ in range(num_steps):
      # Actions are auto-fed into the environment.
      # We just need to pull gamestates and feed them back into the agents.
      with step_profiler:
        # env_queue_sizes.append(self._env._futures_queue.qsize())
        error = False
        gamestates, needs_reset = self._env.pop()
        # if error:
        #   import gc
        #   gc.collect()
        #   referrers = gc.get_referrers(self._env)
        #   import ipdb; ipdb.set_trace()

      is_resetting.append(needs_reset)

      for port, agent in self._agents.items():
        # agent_state_queue_sizes[port].append(agent._state_queue.qsize())
        game = gamestates[port]
        agent.push(game, needs_reset)
        record_state(port, game)

    # # TODO: record the last gamestate and delayed actions.
    # for gamestates, needs_reset in self._env.peek():
    #   is_resetting.append(needs_reset)
    #   for port, game in gamestates.items():
    #     record_state(port, game)

    # Batch everything up to be time-major.
    is_resetting = np.array(is_resetting)

    trajectories = {}
    for port, state_action_list in state_actions.items():
      # Trajectories are time-major.
      state_action = utils.batch_nest_nt(state_action_list)
      rewards = reward.compute_rewards(state_action.state, self._damage_ratio)
      frames = data.Frames(state_action=state_action, reward=rewards)
      trajectories[port] = Trajectory(frames, is_resetting)

    qs = np.array([1, 5, 10, 25, 50, 75, 100])
    def percentile(a):
      return dict(zip(qs, np.percentile(a, qs)))

    # agent_state_queue_size_stats = {
    #     port: percentile(sizes)
    #     for port, sizes in agent_state_queue_sizes.items()
    # }

    timings = {
        'env_pop': step_profiler.mean_time(),
        # TODO: handle concurrency
        'env_push': self._env_push_profiler.mean_time(),
        'agent_pop': {
            port: profiler.mean_time()
            for port, profiler in self._agent_profilers.items()},
        # 'agent_state_queue_size': agent_state_queue_size_stats,
        # 'agent_state_queue_pop': {
        #     port: agent.state_queue_profiler.mean_time()
        #     for port, agent in self._agents.items()
        # },
        'agent_step': {
            port: agent.step_profiler.mean_time()
            for port, agent in self._agents.items()
        },
        # 'env_queue': {q: p for q, p in zip(qs, ps)},
    }

    # self._env_push_profiler.dump_stats('env_push.prof')

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
      num_envs: int,
      num_steps_per_rollout: int,
      async_envs: bool = False,
      ray_envs: bool = False,
      async_inference: bool = False,
      use_gpu: bool = False,
      extra_env_kwargs: dict = {},
  ):
    # TODO: do this in a better way
    if not use_gpu:
      eval_lib.disable_gpus()

    agents = {
        port: eval_lib.build_delayed_agent(
            console_delay=env_kwargs['online_delay'],
            batch_size=num_envs,
            async_inference=async_inference,
            **kwargs,
        )
        for port, kwargs in agent_kwargs.items()
    }

    env_kwargs = env_kwargs.copy()
    for port, kwargs in agent_kwargs.items():
      eval_lib.update_character(
          env_kwargs['players'][port],
          kwargs['state']['config'])

    if not async_envs:
      env_class = env_lib.BatchedEnvironment
    elif ray_envs:
      env_class = env_lib.AsyncBatchedEnvironmentRay
    else:
      env_class = env_lib.AsyncBatchedEnvironmentMP

    self._env = env_class(num_envs, env_kwargs, **extra_env_kwargs)

    self._rollout_worker = RolloutWorker(
        agents, self._env, num_steps_per_rollout)

    self._lock = threading.Lock()

  def rollout(
      self,
      policy_vars: tp.Mapping[Port, tp.Sequence[np.ndarray]],
      num_steps: int,
  ) -> tuple[tp.Mapping[Port, RolloutMetrics], Timings]:
    # with self._lock:
    self._rollout_worker.update_variables(policy_vars)
    trajectories, timings = self._rollout_worker.rollout(num_steps)
    metrics = {
        port: RolloutMetrics.from_trajectory(trajectory)
        for port, trajectory in trajectories.items()
    }
    return metrics, timings
