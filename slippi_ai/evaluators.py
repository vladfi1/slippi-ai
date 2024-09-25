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
    eval_lib,
    policies,
    reward,
    utils,
)
from slippi_ai.types import Game
from slippi_ai.controller_heads import SampleOutputs

Port = int
Timings = dict
Params = tp.Sequence[np.ndarray]

# Mimics data.Batch
class Trajectory(tp.NamedTuple):
  # The [T+1, ...] arrays overlap in time by 1.
  states: Game  # [T+1, B]
  name: np.ndarray  # [T+1, B]
  actions: SampleOutputs  # [T+1, B]
  rewards: np.ndarray  # [T, B]
  is_resetting: bool  # [T+1, B]
  initial_state: policies.RecurrentState  # [B]
  delayed_actions: list[SampleOutputs]  # [D, B]

  @classmethod
  def batch(cls, trajectories: list['Trajectory']) -> 'Trajectory':
    # TODO: test?
    batch_dims = Trajectory(
        states=1,
        name=1,
        actions=1,
        rewards=1,
        is_resetting=1,
        initial_state=0,
        delayed_actions=0,
    )
    return utils.map_nt(
        lambda axis, *ts: utils.concat_nest_nt(ts, axis),
        batch_dims, *trajectories)


class RolloutWorker:

  def __init__(
      self,
      agent_kwargs: tp.Mapping[Port, dict],
      dolphin_kwargs: dict,
      num_envs: int,
      async_envs: bool = False,
      env_kwargs: dict = {},
      use_gpu: bool = False,
      damage_ratio: float = 0,  # For rewards.
      use_fake_envs: bool = False,
  ):
    self._agents = {
        port: eval_lib.build_delayed_agent(
            console_delay=dolphin_kwargs['online_delay'],
            batch_size=num_envs,
            run_on_cpu=not use_gpu,
            **kwargs,
        )
        for port, kwargs in agent_kwargs.items()
    }
    dolphin_kwargs = dolphin_kwargs.copy()
    for port, kwargs in agent_kwargs.items():
      eval_lib.update_character(
          dolphin_kwargs['players'][port],
          kwargs['state']['config'])

    self._num_envs = num_envs
    if use_fake_envs:
      self._env = env_lib.FakeBatchedEnvironment(
          num_envs, players=list(agent_kwargs))
    else:
      if not async_envs:
        env_class = env_lib.BatchedEnvironment
      else:
        env_class = env_lib.AsyncBatchedEnvironmentMP
      self._env = env_class(num_envs, dolphin_kwargs, **env_kwargs)

    self._damage_ratio = damage_ratio

    self._agent_profilers = {
        port: utils.Profiler() for port in self._agents}
    # self._env_push_profiler = cProfile.Profile()
    self._env_push_profiler = utils.Profiler()

    self._prev_agent_outputs = collections.deque()
    self._prev_agent_outputs.append({
        port: agent.dummy_sample_outputs
        for port, agent in self._agents.items()
    })

    # Make sure that the buffer sizes aren't too big.
    # TODO: do this check before env/agent creation
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
    # Because we push env states to the agents once per main loop iteration,
    # we need to leave each agent with at least batch_steps - 1 actions in its
    # buffer. This ensures that the agent will have enough env states pushed to
    # take a multi_step just as its output queue runs out.
    self.env_runahead = min(
        agent.delay - (agent.batch_steps - 1)
        for agent in self._agents.values())
    for _ in range(self.env_runahead):
      self._push_actions()

  def _push_actions(self):
    """Pop actions from the agents and push them to the environment."""
    outputs: dict[Port, SampleOutputs] = {}
    for port, agent in self._agents.items():
      with self._agent_profilers[port]:
        outputs[port] = agent.pop()
    self._prev_agent_outputs.append(outputs)

    decoded_actions = {
        port: self._agents[port].embed_controller.decode(action.controller_state)
        for port, action in outputs.items()
    }
    with self._env_push_profiler:
      self._env.push(decoded_actions)

  def rollout(self, num_steps: int) -> tuple[tp.Mapping[Port, Trajectory], Timings]:
    # This ensures that the agent can process all of the states it will be fed.
    for agent in self._agents.values():
      if num_steps % agent.batch_steps != 0:
        raise ValueError('Agent batch steps must divide rollout length.')

    # Buffers for per-frame data.
    gamestates: dict[Port, list[Game]] = {
        port: [] for port in self._agents
    }
    sample_outputs: dict[Port, list[SampleOutputs]] = {
        port: [] for port in self._agents
    }
    is_resetting: list[bool] = []

    initial_states = {
        port: agent.hidden_state
        for port, agent in self._agents.items()
    }

    step_profiler = utils.Profiler()

    def record_state(
        env_output: env_lib.EnvOutput,
        prev_agent_outputs: dict[Port, SampleOutputs],
    ):
      for port, game in env_output.gamestates.items():
        gamestates[port].append(game)
        sample_outputs[port].append(prev_agent_outputs[port])
      is_resetting.append(env_output.needs_reset)

    for _ in range(num_steps):
      # Note that there will always be a first gamestate before any actions
      # are fed into the environment; either the initial state or the last
      # state peeked on the previous rollout.
      with step_profiler:
        output = self._env.pop()

      record_state(output, self._prev_agent_outputs.popleft())

      # Asynchronously push the gamestates to the agents.
      for port, agent in self._agents.items():
        game = output.gamestates[port]
        # The agent is responsible for calling from_state on the game.
        agent.push(game, output.needs_reset)

      # Feed the actions from the agents into the environment.
      self._push_actions()

    # Record the last gamestate and action, but don't pop them as we will
    # also use them to begin next rollout.
    record_state(self._env.peek(), self._prev_agent_outputs[0])

    # Record the delayed actions.
    assert len(self._prev_agent_outputs) == 1 + self.env_runahead
    remaining_actions = list(self._prev_agent_outputs)[1:]
    delayed_actions: dict[Port, list[SampleOutputs]] = {}
    for port, agent in self._agents.items():
      delayed_actions[port] = [actions[port] for actions in remaining_actions]
      num_left = agent.delay - self.env_runahead
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
    for port, agent in self._agents.items():
      states=utils.batch_nest_nt(gamestates[port])
      trajectories[port] = Trajectory(
          # TODO: Let the learner call from_state on game
          states=agent._policy.embed_game.from_state(states),
          name=np.full(
              [num_steps + 1, self._num_envs],
              agent.name_code,
              dtype=embed.NAME_DTYPE),
          actions=utils.batch_nest_nt(sample_outputs[port]),
          rewards=reward.compute_rewards(states, self._damage_ratio),
          is_resetting=is_resetting,
          initial_state=initial_states[port],
          # Note that delayed actions aren't time-concatenated, mainly to
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
    return cls(reward=np.sum(trajectory.rewards))


class Evaluator(RolloutWorker):

  def rollout(
      self,
      num_steps: int,
      policy_vars: tp.Optional[tp.Mapping[Port, Params]] = None,
  ) -> tuple[tp.Mapping[Port, RolloutMetrics], Timings]:
    if policy_vars is not None:
      self.update_variables(policy_vars)
    trajectories, timings = super().rollout(num_steps)
    metrics = {
        port: RolloutMetrics.from_trajectory(trajectory)
        for port, trajectory in trajectories.items()
    }
    return metrics, timings

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
          env_kwargs=env_kwargs,
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
