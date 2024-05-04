"""Evaluates a policy."""

import collections
import contextlib
import typing as tp
import cProfile

import numpy as np
import ray

from slippi_ai import envs as env_lib
from slippi_ai import (
    embed,
    data,
    eval_lib,
    policies,
    reward,
    utils,
)

Port = int
Timings = dict
Params = tp.Sequence[np.ndarray]

# Mimics data.Batch
class Trajectory(tp.NamedTuple):
  frames: data.Frames
  is_resetting: bool
  initial_state: policies.RecurrentState
  delayed_actions: embed.Action


class RolloutWorker:

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
      damage_ratio: float = 0,  # For rewards.
  ):
    self._agents = {
        port: eval_lib.build_delayed_agent(
            console_delay=dolphin_kwargs['online_delay'],
            batch_size=num_envs,
            async_inference=async_inference,
            run_on_cpu=not use_gpu,
            **kwargs,
        )
        for port, kwargs in agent_kwargs.items()
    }

    self._batched_name_codes = {
        port: np.full([num_envs], agent._agent._name_code, dtype=np.int32)
        for port, agent in self._agents.items()
    }

    dolphin_kwargs = dolphin_kwargs.copy()
    for port, kwargs in agent_kwargs.items():
      eval_lib.update_character(
          dolphin_kwargs['players'][port],
          kwargs['state']['config'])

    if not async_envs:
      env_class = env_lib.BatchedEnvironment
    elif ray_envs:
      raise NotImplementedError('Ray environments are not up to date.')
      # env_class = env_lib.AsyncBatchedEnvironmentRay
    else:
      env_class = env_lib.AsyncBatchedEnvironmentMP

    self._env = env_class(num_envs, dolphin_kwargs, **env_kwargs)

    self._damage_ratio = damage_ratio

    self._agent_profilers = {
        port: utils.Profiler() for port in self._agents}
    # self._env_push_profiler = cProfile.Profile()
    self._env_push_profiler = utils.Profiler()

    self._prev_actions = collections.deque()
    self._prev_actions.append({
        port: agent.default_controller
        for port, agent in self._agents.items()
    })

    # Make sure that the buffer sizes aren't too big.
    for agent in self._agents.values():
      # We get one environment state (the initial one) for free.
      slack = 1 + agent.delay

      # Maximum number of items that could get stuck.
      max_agent_buffer = agent.batch_steps - 1
      max_env_buffer = self._env.num_steps - 1

      if max_agent_buffer + max_env_buffer > slack:
        raise ValueError(
            f'Agent and environment step buffer sizes are too large: '
            f'{max_agent_buffer} + {max_env_buffer} > {slack}')

    # Get the environment to run ahead as much as possible.
    self.min_delay = min(agent.delay for agent in self._agents.values())
    for _ in range(self.min_delay):
      self._push_actions()

  def _push_actions(self):
    """Pop actions from the agents and push them to the environment."""
    actions = {}
    for port, agent in self._agents.items():
      with self._agent_profilers[port]:
        actions[port] = agent.pop()
    self._prev_actions.append(actions)
    with self._env_push_profiler:
      self._env.push(actions)

  def rollout(self, num_steps: int) -> tuple[tp.Mapping[Port, Trajectory], Timings]:
    state_actions: dict[Port, list[embed.StateAction]] = {
        port: [] for port in self._agents
    }
    is_resetting: list[bool] = []

    initial_states = {
        port: agent._agent.hidden_state
        for port, agent in self._agents.items()
    }

    step_profiler = utils.Profiler()
    # agent_profilers = {
    #     port: utils.Profiler() for port in self._agents}
    # env_push_profiler = utils.Profiler()

    def record_state(output: env_lib.EnvOutput):
      prev_actions = self._prev_actions[0]
      for port, game in output.gamestates.items():
        state_action = embed.StateAction(
            state=game,
            # TODO: check that the prev action matches what the env has?
            # action=game.p0.controller,
            action=prev_actions[port],
            name=self._batched_name_codes[port],
        )
        state_actions[port].append(state_action)
      is_resetting.append(output.needs_reset)

    for t in range(num_steps):
      # Note that there will always be a first gamestate before any actions
      # are fed into the environment; either the initial state or the last
      # state peeked on the previous rollout.
      with step_profiler:
        # env_queue_sizes.append(self._env._futures_queue.qsize())
        output = self._env.pop()

      record_state(output)
      self._prev_actions.popleft()

      # Asynchronously push the gamestates to the agents.
      for port, agent in self._agents.items():
        game = output.gamestates[port]
        agent.push(game, output.needs_reset)

      # Feed the actions from the agents into the environment.
      self._push_actions()

    # Record the last gamestate and action, but also save them for the
    # next rollout.
    record_state(self._env.peek())

    # Record the delayed actions.
    assert len(self._prev_actions) == 1 + self.min_delay
    remaining_actions = list(self._prev_actions)[1:]
    delayed_actions: dict[Port, list[embed.Action]] = {}
    for port, agent in self._agents.items():
      delayed_actions[port] = [actions[port] for actions in remaining_actions]
      num_left = agent.delay - self.min_delay
      delayed_actions[port].extend(agent.peek_n(num_left))

      # Note: the above call to peek_n forces the agent to process all
      # of the `num_steps` states that it's been fed. This ensures that the
      # agent's hidden state is the correct one on the next rollout.
      if isinstance(agent, eval_lib.AsyncDelayedAgent):
        # Assert that the agent has in fact processed all of the states.
        assert agent._state_queue.empty()

    # Now batch everything up into time-major Trajectories.
    trajectories = {}
    is_resetting = np.array(is_resetting)
    for port, state_action_list in state_actions.items():
      state_action = utils.batch_nest_nt(state_action_list)
      rewards = reward.compute_rewards(state_action.state, self._damage_ratio)
      frames = data.Frames(state_action=state_action, reward=rewards)
      trajectories[port] = Trajectory(
          frames=frames,
          is_resetting=is_resetting,
          initial_state=initial_states[port],
          # Note that delayed actions aren't batched, mainly to
          # simplify the case where the delay is 0.
          delayed_actions=delayed_actions[port])

    timings = {
        'env_pop': step_profiler.mean_time(),
        'env_push': self._env_push_profiler.mean_time(),
        'agent_pop': {
            port: profiler.mean_time()
            for port, profiler in self._agent_profilers.items()},
        'agent_step': {
            port: agent.step_profiler.mean_time()
            for port, agent in self._agents.items()
        },
    }

    # self._env_push_profiler.dump_stats('env_push.prof')

    return trajectories, timings

  def update_variables(
      self, updates: tp.Mapping[Port, Params],
  ):
    for port, values in updates.items():
      policy = self._agents[port]._policy
      for var, val in zip(policy.variables, values):
        var.assign(val)

  @contextlib.contextmanager
  def run(self):
    try:
      self.start()
      yield
    finally:
      self.stop()

  def start(self):
    # TODO: don't allow starting more than once, or running without starting.
    for agent in self._agents.values():
      agent.start()

  def stop(self):
    for agent in self._agents.values():
      agent.stop()
    self._env.stop()

class RolloutMetrics(tp.NamedTuple):
  reward: float

  @classmethod
  def from_trajectory(cls, trajectory: Trajectory) -> 'RolloutMetrics':
    return cls(reward=np.sum(trajectory.frames.reward))


class Evaluator:

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
    self._rollout_worker = RolloutWorker(
        agent_kwargs=agent_kwargs,
        dolphin_kwargs=dolphin_kwargs,
        num_envs=num_envs,
        async_envs=async_envs,
        ray_envs=ray_envs,
        env_kwargs=env_kwargs,
        async_inference=async_inference,
        use_gpu=use_gpu,
    )

  def update_variables(
      self, updates: tp.Mapping[Port, tp.Sequence[np.ndarray]],
  ):
    self._rollout_worker.update_variables(updates)

  def rollout(
      self,
      num_steps: int,
      policy_vars: tp.Optional[tp.Mapping[Port, Params]] = None,
  ) -> tuple[tp.Mapping[Port, RolloutMetrics], Timings]:
    if policy_vars is not None:
      self._rollout_worker.update_variables(policy_vars)
    trajectories, timings = self._rollout_worker.rollout(num_steps)
    metrics = {
        port: RolloutMetrics.from_trajectory(trajectory)
        for port, trajectory in trajectories.items()
    }
    return metrics, timings

  @contextlib.contextmanager
  def run(self):
    try:
      self._rollout_worker.start()
      yield
    finally:
      self._rollout_worker.stop()

RayRolloutWorker = ray.remote(RolloutWorker)

class RayEvaluator:
  def __init__(
      self,
      agent_kwargs: tp.Mapping[Port, dict],
      dolphin_kwargs: dict,
      num_envs: int,  # per-worker
      num_workers: int = 1,
      async_envs: bool = False,
      env_kwargs: dict = {},
      ray_envs: bool = False,
      async_inference: bool = False,
      use_gpu: bool = False,
      resources: tp.Mapping[str, float] = {},
  ):
    # TODO: Allow multiple gpu workers on the same machine. For this,
    # we'll need to tell tensorflow not to reserve all gpu memory.
    build_worker = RayRolloutWorker.options(
        num_gpus=1 if use_gpu else 0,
        resources=resources)

    self._rollout_workers: list[ray.ObjectRef[RolloutWorker]] = []
    for _ in range(num_workers):
      self._rollout_workers.append(build_worker.remote(
          agent_kwargs=agent_kwargs,
          dolphin_kwargs=dolphin_kwargs,
          num_envs=num_envs,
          async_envs=async_envs,
          ray_envs=ray_envs,
          env_kwargs=env_kwargs,
          async_inference=async_inference,
          use_gpu=use_gpu,
      ))

  def update_variables(
      self, updates: tp.Mapping[Port, tp.Sequence[np.ndarray]],
  ):
    ray.wait([
        worker.update_variables.remote(updates)
        for worker in self._rollout_workers])

  def rollout(
      self,
      num_steps: int,
      policy_vars: tp.Optional[tp.Mapping[Port, Params]] = None,
  ) -> tuple[tp.Mapping[Port, RolloutMetrics], Timings]:
    if policy_vars is not None:
      for worker in self._rollout_workers:
        worker.update_variables.remote(policy_vars)

    rollout_futures = [
        worker.rollout.remote(num_steps)
        for worker in self._rollout_workers
    ]
    rollout_results = ray.get(rollout_futures)
    trajectories, timings = zip(*rollout_results)

    # Merge the results.
    trajectories = utils.concat_nest_nt(trajectories)
    # TODO: handle non-mean timings.
    timings = utils.map_nt(lambda *args: np.mean(args), *timings)

    metrics = {
        port: RolloutMetrics.from_trajectory(trajectory)
        for port, trajectory in trajectories.items()
    }
    return metrics, timings

  @contextlib.contextmanager
  def run(self):
    try:
      ray.wait([worker.start.remote() for worker in self._rollout_workers])
      yield
    except KeyboardInterrupt:
      # Properly shut down the workers. If we don't do this, the workers
      # can get stuck, not sure why.
      raise
    finally:
      ray.wait([worker.stop.remote() for worker in self._rollout_workers])
