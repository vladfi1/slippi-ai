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
from slippi_ai.controller_heads import SampleOutputs
from slippi_ai.data import NAME_DTYPE
from slippi_ai.evaluators import Port, Timings, Trajectory
from slippi_ai.jax import policies
from slippi_ai.types import Game

T = tp.TypeVar('T')

class MultiMap(tp.Protocol):
  def __call__(self, f: tp.Callable, *args: T) -> T:
    ...

fast_map: MultiMap = tp.cast(MultiMap, jax.tree.map)

class SliceMap(tp.Protocol):
  def __call__(self, s: slice | tuple[slice, ...], struct: T) -> T:
    ...

slice_map: SliceMap = lambda s, struct: jax.tree.map(lambda x: x[s], struct)

Agent = eval_lib.DelayedAgent | eval_lib.AsyncDelayedAgent

class AgentInfo(tp.NamedTuple):
  agent: Agent
  ports: tuple[Port, ...]
  env_slice: slice
  env_to_port_slices: tuple[slice, ...]
  agent_to_port_slices: tuple[slice, ...]


class JaxSimRolloutWorker:
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

    self._env_kwargs = dict(
        num_envs=self._num_envs,
        players=[kwargs['players'] for kwargs in dolphin_kwargs],
        stage=stages,
        max_frame_id=(
            -1 if dolphin_kwargs_0['infinite_time'] else 8 * 60 * 60 - 123),
        fake=use_fake_envs,
        frame_buffer_length=self._rollout_length + 1,
    )

    self._env = self._build_env()
    self._needs_reset = np.ones(self._num_envs, dtype=np.bool_)
    game_batch = self._env.current_game_batch(self._needs_reset)
    self._state_buffer = _make_trajectory_state_buffer(
        game_batch.game, self._rollout_length)
    self._reset_buffer = np.empty(
        (self._rollout_length + 1, self._num_players), dtype=np.bool_)

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
    self._env = self._build_env()
    self._needs_reset[:] = True

  def update_variables(self, updates: tp.Mapping[Port, tp.Any]):
    for ports, update in updates.items():
      self._port_to_agent[ports].policy.set_state(update)

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
    reset_buffer = self._reset_buffer

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
        t: int,
        prev_agent_outputs: list[SampleOutputs],
    ):
      game_batch = self._env.current_game_batch(self._needs_reset)

      copy_start = time.perf_counter()
      _copy_state_slot(state_buffer, t)
      reset_buffer[t] = game_batch.needs_reset

      for buffer, output in zip(action_buffers, prev_agent_outputs):
        buffer.append(output)

      timings['state_copy'] += time.perf_counter() - copy_start

      return game_batch

    # Step the sim immediately with already-delayed controller inputs, while
    # collecting a chunk of observations for one fused JAX actor call.
    for t in step_iter:
      game_batch = record_state(t, self._prev_agent_outputs.popleft())

      agent_outputs: list[SampleOutputs] = []
      controllers: sim_env.Controllers = {}

      for agent_info in self._agents:
        agent = agent_info.agent

        agent_inputs = slice_map(
            agent_info.env_slice,
            (game_batch.game, game_batch.needs_reset))

        step_start = time.perf_counter()
        output = agent.step(*agent_inputs)
        agent_outputs.append(output)

        decoded_controllers = agent.decode_controller(output.controller_state)
        # TODO: send actions to the environment without slicing by port
        for port, port_slice in zip(agent_info.ports, agent_info.agent_to_port_slices):
          controllers[port] = slice_map(port_slice, decoded_controllers)

        elapsed = time.perf_counter() - step_start
        elapsed_per_port = elapsed / len(agent_info.ports)
        for port in agent_info.ports:
          timings[f'agent_step_{port}'] += elapsed_per_port

      self._prev_agent_outputs.append(agent_outputs)

      env_start = time.perf_counter()
      self._needs_reset = self._env.advance(controllers)
      timings['env_step'] += time.perf_counter() - env_start

    # Capture the T+1 terminal observation and assemble the learner trajectory
    # trees expected by the existing PPO path.
    record_state(num_steps, self._prev_agent_outputs[0])

    build_start = time.perf_counter()
    time_major_states = state_buffer.states
    rewards = reward.compute_rewards(
        time_major_states, damage_ratio=0, nana_ratio=0)

    trajectories: dict[Port, Trajectory] = {}
    for agent_info, action_buffer, initial_state in zip(
        self._agents, action_buffers, initial_states):
      agent = agent_info.agent
      env_slice = agent_info.env_slice
      actions = fast_map(utils.stack, *action_buffer)

      if self._per_agent_outputs:
        states = fast_map(
            lambda x: np.asarray(x[:, env_slice]).copy(),
            state_buffer.states)
        trajectories[agent_info.ports[0]] = Trajectory(
            states=agent.policy.encode_game(states),
            name=np.broadcast_to(
                agent.name_code,
                [num_steps + 1, env_slice.stop - env_slice.start]),
            actions=actions,
            rewards=rewards[:, env_slice],
            is_resetting=reset_buffer[:, env_slice].copy(),
            initial_state=initial_state,
            delayed_actions=agent.peek_n(agent.delay),
        )
      else:
        for port, env_to_port_slice, agent_to_port_slice in zip(
            agent_info.ports, agent_info.env_to_port_slices, agent_info.agent_to_port_slices):
          states = fast_map(
              lambda x: np.asarray(x[:, env_to_port_slice]).copy(),
              state_buffer.states)
          batch_size = agent_to_port_slice.stop - agent_to_port_slice.start

          trajectories[port] = Trajectory(
              states=agent.policy.encode_game(states),
              name=np.broadcast_to(
                  agent.name_code[agent_to_port_slice],
                  [num_steps + 1, batch_size]),
              actions=slice_map((slice(None), agent_to_port_slice), actions),
              rewards=rewards[:, env_to_port_slice],
              is_resetting=reset_buffer[:, env_to_port_slice].copy(),
              initial_state=slice_map(agent_to_port_slice, initial_state),
              delayed_actions=slice_map(agent_to_port_slice, agent.peek_n(agent.delay)),
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
        'unexpected_reset': reset_buffer[1:, :self._num_envs].copy(),
        'completed_games': self._env.pop_completed_games(),
    }

  def _build_env(self):
    return sim_env.SimBatchedEnvironment(**self._env_kwargs)


class _TrajectoryStateBuffer(tp.NamedTuple):
  """Time-major storage for T+1 policy observations.

  `current_game_batch()` reuses one mutable Game tree, while the learner needs
  the whole rollout after the sim has advanced. `slots` are NumPy views into the
  time-major `states` tree, so copying into a slot writes directly into the
  corresponding frame of the final trajectory buffer.
  """

  states: Game
  slots: list[Game]  # T+1 views into states
  slot_leaves: list[tuple]
  source_leaves: tuple


def _make_trajectory_state_buffer(
    source_game: Game,
    rollout_length: int,
) -> _TrajectoryStateBuffer:
  states = utils.map_single_structure(
      lambda leaf: np.empty(
          (rollout_length + 1,) + np.asarray(leaf).shape,
          dtype=np.asarray(leaf).dtype,
      ),
      source_game,
  )
  slots = [
      utils.map_single_structure(lambda leaf, i=i: leaf[i], states)
      for i in range(rollout_length + 1)
  ]
  return _TrajectoryStateBuffer(
      states=states,
      slots=slots,
      slot_leaves=[tuple(jax.tree.leaves(slot)) for slot in slots],
      source_leaves=tuple(jax.tree.leaves(source_game)),
  )


def _copy_state_slot(state_buffer: _TrajectoryStateBuffer, index: int):
  for dst, src in zip(
      state_buffer.slot_leaves[index],
      state_buffer.source_leaves,
  ):
    dst[...] = src
