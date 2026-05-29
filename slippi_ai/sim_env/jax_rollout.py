"""JAX rollout assembly for single-process melee-sim-light envs.

The generic evaluator path builds slippi-ai Game/controller Python objects once
per port per frame. This worker uses
`SimBatchedEnvironment.current_game_batch()` and `step_encoded()` instead: one
reusable observation tree is filled from native sim buffers, JAX samples both
port perspectives, and the resulting trajectory is split back into the port
entries expected by the learner.
"""

import collections
import contextlib
import itertools
import time
import typing as tp

import jax
import melee
import numpy as np

from slippi_ai import eval_lib
from slippi_ai import reward
from slippi_ai import sim_env
from slippi_ai import utils
from slippi_ai.sim_env.observations import TrajectoryStateBuffer
from slippi_ai.controller_heads import SampleOutputs
from slippi_ai.evaluators import Port, Timings, Trajectory, AbstractRolloutWorker
from slippi_ai.jax.jax_utils import fast_map, slice_map

T = tp.TypeVar('T')

Agent = eval_lib.DelayedAgent | eval_lib.AsyncDelayedAgent

class AgentInfo(tp.NamedTuple):
  agent: Agent
  ports: tuple[Port, ...]
  env_slice: slice
  env_to_port_slices: tuple[slice, ...]
  agent_to_port_slices: tuple[slice, ...]
  # buffer: _TrajectoryStateBuffer


class JaxSimRolloutWorker(AbstractRolloutWorker):
  """RolloutWorker-compatible adapter for JAX policies on one sim batch."""

  ports: tuple[Port, Port] = (1, 2)

  def __init__(
      self,
      *,
      agent_kwargs: tp.Mapping[int | tuple[int, ...], dict],
      dolphin_kwargs: tp.Union[dict, tp.Sequence[dict]],
      num_envs: int,
      rollout_length: int,
      batch_steps: int = 1,
      per_agent_outputs: bool = True,
      use_fake_envs: bool = False,
      async_envs: bool = False,
      inner_batch_size: tp.Optional[int] = None,
  ):
    self._num_envs = num_envs
    self._num_players = self._num_envs * len(self.ports)
    self._rollout_length = rollout_length
    del batch_steps  # Not used yet.
    self._per_agent_outputs = per_agent_outputs

    self._batch_slice_by_port = {
        port: slice(i * self._num_envs, (i + 1) * self._num_envs)
        for i, port in enumerate(self.ports)
    }
    if isinstance(dolphin_kwargs, dict):
      dolphin_kwargs = [dolphin_kwargs.copy() for _ in range(self._num_envs)]
    elif self._num_envs != len(dolphin_kwargs):
      raise ValueError(
          f'num_envs={self._num_envs} does not match '
          f'{len(dolphin_kwargs)} dolphin kwargs entries.')
    else:
      dolphin_kwargs = list(dolphin_kwargs)


    self._agents: list[AgentInfo] = []
    self._port_to_agent: dict[Port, Agent] = {}
    given_ports = set[Port]()

    for ports, kwargs in agent_kwargs.items():
      if isinstance(ports, int):
        ports = (ports,)

      for port in ports:
        if port not in self.ports:
          raise ValueError(f'Invalid port {port} in agent_kwargs, expected one of {self.ports}')
        if port in given_ports:
          raise ValueError(f'Multiple entries for port {port} in agent_kwargs')
        given_ports.add(port)

      for p1, p2 in zip(ports, ports[1:]):
        if p2 != p1 + 1:
          raise ValueError(f'Agent ports must be consecutive, got {ports}')

      # TODO: use agent's observation_config
      # TODO: check that agent's character matches the env
      agent = eval_lib.build_delayed_agent(
          console_delay=0,
          batch_size=len(ports) * self._num_envs,
          **kwargs)
      self._port_to_agent[ports[0]] = agent

      start_index = (ports[0] - 1)  # ports are 1-indexed
      end_index = start_index + len(ports)
      env_slice = slice(
          start_index * self._num_envs,
          end_index * self._num_envs,
      )

      env_to_port_slices = tuple(
          slice((port - 1) * self._num_envs, port * self._num_envs)
          for port in ports
      )

      agent_to_port_slices = tuple(
          slice(i * self._num_envs, (i + 1) * self._num_envs)
          for i in range(len(ports))
      )

      self._agents.append(AgentInfo(
          agent=agent,
          ports=ports,
          env_slice=env_slice,
          env_to_port_slices=env_to_port_slices,
          agent_to_port_slices=agent_to_port_slices,
      ))

    if given_ports != set(self.ports):
      raise ValueError(f'Got agent kwargs for ports {given_ports}, expected {self.ports}')

    # Initialize prev_agent_outputs queues with a single dummy action
    self._prev_agent_outputs = collections.deque[list[SampleOutputs]]()
    self._prev_agent_outputs.append([
        agent_info.agent.dummy_sample_outputs
        for agent_info in self._agents
    ])

    # Translate the Dolphin-style environment config into the smaller sim env
    # config. The sim path still accepts the same high-level launch config as
    # the generic rollout worker.
    dolphin_kwargs_0 = dolphin_kwargs[0]
    stages = [kwargs['stage'] for kwargs in dolphin_kwargs]
    if any(stage is melee.Stage.RANDOM_STAGE for stage in stages):
      stages = list(itertools.islice(
          itertools.cycle(sim_env.SUPPORTED_STAGES),
          self._num_envs,
      ))

    # Get the environment to run ahead as much as possible.
    # Because we push env states to the agents once per main loop iteration,
    # we need to leave each agent with at least batch_steps - 1 actions in its
    # buffer. This ensures that the agent will have enough env states pushed to
    # take a multi_step just as its output queue runs out.
    self._env_runahead = min(
        info.agent.delay - (info.agent.batch_steps - 1)
        for info in self._agents)

    self._async_envs = async_envs
    self._inner_batch_size = inner_batch_size
    self._env_kwargs = dict(
        num_envs=self._num_envs,
        players=[kwargs['players'] for kwargs in dolphin_kwargs],
        stage=stages,
        max_frame_id=(
            -1 if dolphin_kwargs_0['infinite_time'] else 8 * 60 * 60 - 123),
        fake=use_fake_envs,
        frame_buffer_length=self._rollout_length + 1,
        runahead=self._env_runahead,
    )

    self._build_env()
    self._needs_reset = np.ones(self._num_envs, dtype=np.bool_)
    # TODO: consider per-agent buffers to avoid some extra slicing
    self._state_buffer = TrajectoryStateBuffer.build(
        self._num_players, self._rollout_length + 1)

    # Start env runahead
    for _ in range(self._env_runahead):
      self._push_actions()

  def start(self):
    pass

  @contextlib.contextmanager
  def run(self):
    try:
      self.start()
      yield
    finally:
      self.stop()

  def stop(self):
    self._env.stop()

  def active_sim_games(self) -> list[dict[str, int | str]]:
    return self._env.active_games()

  def reset_env(self):
    self.stop()
    self._build_env()
    self._needs_reset[:] = True

    # Start env runahead
    # TODO: properly reset the agents with dummy actions instead of reusing
    # delayed actions from the previous rollout.
    raise NotImplementedError  # not difficult, just lazy
    # Code below is copied from evaluators.py
    assert len(self._prev_agent_outputs) == 1 + self._env_runahead
    for agent_outputs in list(self._prev_agent_outputs)[1:]:
      decoded_actions = {
          port: self._agents[port].decode_controller(output.controller_state)
          for port, output in agent_outputs.items()
      }
      with self._env_push_profiler:
        self._env.push(decoded_actions)

  def update_variables(self, updates: tp.Mapping[Port, tp.Any]):
    for ports, update in updates.items():
      self._port_to_agent[ports].policy.set_state(update)

  def _push_actions(self, timings: tp.Optional[dict] = None):
    """Pop actions from the agents and push them to the environment."""
    agent_outputs: list[SampleOutputs] = []
    controllers: sim_env.Controllers = {}

    for agent_info in self._agents:
      agent = agent_info.agent

      # Note: game_batch contains mutable views: the agent should write these
      # into its own internal state if it needs to reference them later, e.g.
      # if it batches steps across time.
      pop_start = time.perf_counter()
      output = agent.pop()
      agent_outputs.append(output)

      decoded_controllers = agent.decode_controller(output.controller_state)
      # TODO: send actions to the environment without slicing by port
      for port, port_slice in zip(agent_info.ports, agent_info.agent_to_port_slices):
        controllers[port] = slice_map(port_slice, decoded_controllers)

      if timings:
        elapsed = time.perf_counter() - pop_start
        elapsed_per_port = elapsed / len(agent_info.ports)
        for port in agent_info.ports:
          timings[f'agent_pop_{port}'] += elapsed_per_port

    self._prev_agent_outputs.append(agent_outputs)

    push_start = time.perf_counter()
    self._env.push(controllers)
    if timings:
      timings['env_push'] += time.perf_counter() - push_start

  def rollout(
      self,
      num_steps: int,
      verbose: bool = False,
  ) -> tuple[tp.Mapping[Port, Trajectory], Timings]:
    timings: dict[str, float] = collections.defaultdict(float)
    if num_steps != self._rollout_length:
      raise ValueError(
          f'JaxSimRolloutWorker was built for rollout_length={self._rollout_length}, '
          f'got rollout({num_steps}).')

    state_buffer = self._state_buffer

    action_buffers: list[list[SampleOutputs]] = [
        [] for _ in self._agents
    ]

    initial_states = [
        agent_info.agent.hidden_state
        for agent_info in self._agents
    ]

    if verbose:
      import tqdm
      step_iter = tqdm.trange(num_steps, desc='Rollout', unit='step')
    else:
      step_iter = range(num_steps)

    def record_state(
        game_batch: sim_env.GameBatch,
        prev_agent_outputs: list[SampleOutputs],
        t: int,
    ):
      copy_start = time.perf_counter()

      slot = state_buffer.slots[t]
      fast_map(np.copyto, slot.game_batch, game_batch)

      for buffer, output in zip(action_buffers, prev_agent_outputs):
        buffer.append(output)

      timings['state_copy'] += time.perf_counter() - copy_start

    for t in step_iter:
      env_pop_start = time.perf_counter()
      game_batch = self._env.pop()
      timings['env_pop'] += time.perf_counter() - env_pop_start

      record_state(game_batch, self._prev_agent_outputs.popleft(), t)

      for agent_info in self._agents:
        agent = agent_info.agent

        if len(self._agents) == 1:
          # If there's only one agent, we can skip the slicing.
          agent_inputs = game_batch
        else:
          agent_inputs = slice_map(agent_info.env_slice, game_batch)

        # Note: game_batch contains mutable views: the agent should write these
        # into its own internal state if it needs to reference them later, e.g.
        # if it batches steps across time.
        push_start = time.perf_counter()
        agent.push(agent_inputs.game, agent_inputs.needs_reset)

        elapsed = time.perf_counter() - push_start
        elapsed_per_port = elapsed / len(agent_info.ports)
        for port in agent_info.ports:
          timings[f'agent_push_{port}'] += elapsed_per_port

      # Feed the actions from the agents into the environment.
      self._push_actions(timings)

    # Capture the T+1 terminal observation and assemble the learner trajectory
    # trees expected by the existing PPO path.
    record_state(self._env.peek(), self._prev_agent_outputs[0], num_steps)

    build_start = time.perf_counter()
    time_major_states = state_buffer.states
    rewards = reward.ko_diff(time_major_states.game)

    # Record the delayed actions.
    assert len(self._prev_agent_outputs) == 1 + self._env_runahead
    remaining_actions = list(self._prev_agent_outputs)[1:]

    trajectories: dict[Port, Trajectory] = {}
    for i, agent_info in enumerate(self._agents):
      initial_state = initial_states[i]
      agent = agent_info.agent
      env_slice = agent_info.env_slice
      actions = fast_map(utils.stack, *action_buffers[i])

      delayed_actions = [actions[i] for actions in remaining_actions]
      num_left = agent_info.agent.delay - self._env_runahead
      delayed_actions.extend(agent_info.agent.peek_n(num_left))

      if self._per_agent_outputs:
        states = fast_map(
            lambda x: np.asarray(x[:, env_slice]).copy(),
            state_buffer.states)
        trajectories[agent_info.ports[0]] = Trajectory(
            states=agent.policy.encode_game(states.game),
            name=np.broadcast_to(
                agent.name_code,
                [num_steps + 1, env_slice.stop - env_slice.start]),
            actions=actions,
            rewards=rewards[:, env_slice].astype(np.float32),
            is_resetting=states.needs_reset[:, env_slice].copy(),
            initial_state=initial_state,
            delayed_actions=delayed_actions,
        )
      else:
        for port, env_to_port_slice, agent_to_port_slice in zip(
            agent_info.ports, agent_info.env_to_port_slices, agent_info.agent_to_port_slices):
          states = fast_map(
              lambda x: np.asarray(x[:, env_to_port_slice]).copy(),
              state_buffer.states)
          batch_size = agent_to_port_slice.stop - agent_to_port_slice.start

          trajectories[port] = Trajectory(
              states=agent.policy.encode_game(states.game),
              name=np.broadcast_to(
                  agent.name_code[agent_to_port_slice],
                  [num_steps + 1, batch_size]),
              actions=slice_map((slice(None), agent_to_port_slice), actions),
              rewards=rewards[:, env_to_port_slice].astype(np.float32),
              is_resetting=states.needs_reset[:, env_to_port_slice].copy(),
              initial_state=slice_map(agent_to_port_slice, initial_state),
              delayed_actions=slice_map(agent_to_port_slice, delayed_actions),
          )

    timings['trajectory_build'] = time.perf_counter() - build_start

    timing = {
        'env_pop': timings['state_copy'] / max(num_steps, 1),
        'env_push': timings['env_step'] / max(num_steps, 1),
        'agent_step': {
            port: timings[f'agent_step_{port}'] / max(num_steps, 1)
            for port in self.ports
        },
        'trajectory_build': timings['trajectory_build'],
    }
    return trajectories, {
        'timing': timing,
        'unexpected_reset': state_buffer.states.needs_reset[1:, :self._num_envs].copy(),
        'completed_games': self._env.pop_completed_games(),
    }

  def _build_env(self):
    # if self._async_envs:
    #   assert self._inner_batch_size is not None
    #   self._env = sim_env.MultiprocessSimEnvironment(
    #       **self._env_kwargs,
    #       inner_batch_size=self._inner_batch_size,
    #   )

    self._env = sim_env.AsyncSimBatchedEnvironment(**self._env_kwargs)
