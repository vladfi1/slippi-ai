"""Evaluates a policy."""

import contextlib
import threading
import typing as tp
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

  def rollout(self, num_steps: int) -> tuple[tp.Mapping[Port, Trajectory], Timings]:
    if num_steps % self._env.num_steps != 0:
      raise ValueError(
          f'rollout length {num_steps} must be a multiple of '
          'environment time-batch-size {self._env.num_steps}')

    state_actions: dict[Port, list[embed.StateAction]] = {
        port: [] for port in self._agents
    }
    is_resetting: list[bool] = []

    step_profiler = utils.Profiler()
    agent_profilers = {
        port: utils.Profiler() for port in self._agents}

    def record_state(port: Port, game: embed.Game):
      state_action = embed.StateAction(
          state=game,
          # TODO: use actual controller from agent instead?
          action=game.p0.controller,
          name=self._agents[port]._agent._name_code,
      )
      state_actions[port].append(state_action)

    # agent_state_queue_sizes = {port: [] for port in self._agents}
    # env_queue_sizes = []

    for t in range(num_steps):
      # Actions are auto-fed into the environment.
      # We just need to pull gamestates and feed them back into the agents.
      # Note that there will always be a first gamestate before any actions
      # are fed into the environment.
      with step_profiler:
        # env_queue_sizes.append(self._env._futures_queue.qsize())
        gamestates, needs_reset = self._env.pop()

      is_resetting.append(needs_reset)

      # Asynchronously push the gamestates to the agents.
      for port, agent in self._agents.items():
        # agent_state_queue_sizes[port].append(agent._state_queue.qsize())
        game = gamestates[port]
        agent.push(game, needs_reset)
        record_state(port, game)

      # Feed the actions into the environment.
      # If the environment batches multiple steps, we need to make sure
      # to feed many actions in ahead of time.
      if t % self._env.num_steps == 0:
        for _ in range(self._env.num_steps):
          actions = {}
          for port, agent in self._agents.items():
            with agent_profilers[port]:
              actions[port] = agent.pop()
          with self._env_push_profiler:
            self._env.push(actions)

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
            for port, profiler in agent_profilers.items()},
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
      raise NotImplementedError('Ray environments are not up to date.')
      # env_class = env_lib.AsyncBatchedEnvironmentRay
    else:
      env_class = env_lib.AsyncBatchedEnvironmentMP

    self._env = env_class(num_envs, env_kwargs, **extra_env_kwargs)

    # Make sure that the buffer sizes aren't too big.
    for agent in agents.values():
      # We get one environment state (the initial one) for free.
      slack = 1 + agent.delay

      # Maximum number of items that could get stuck.
      max_agent_buffer = agent.batch_steps - 1
      max_env_buffer = self._env.num_steps - 1

      if max_agent_buffer + max_env_buffer > slack:
        raise ValueError(
            f'Agent and environment step buffer sizes are too large: '
            f'{max_agent_buffer} + {max_env_buffer} > {slack}')

    self._agents = agents
    self._rollout_worker = RolloutWorker(
        agents, self._env, num_steps_per_rollout)

    self._lock = threading.Lock()

  @contextlib.contextmanager
  def run(self):
    with contextlib.ExitStack() as stack:
      for agent in self._agents.values():
        stack.enter_context(agent.run())
      if isinstance(self._env, env_lib.AsyncBatchedEnvironmentMP):
        stack.enter_context(self._env.run())
      yield

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
