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


class JaxSimRolloutWorker:
  """RolloutWorker-compatible adapter for JAX policies on one sim batch."""

  ports: tuple[Port, Port] = (1, 2)

  def __init__(
      self,
      *,
      # policy: policies.Policy,
      agent_kwargs: tp.Mapping[int, dict],
      dolphin_kwargs: tp.Union[dict, tp.Sequence[dict]],
      num_envs: int,
      rollout_length: int,
      batch_steps: int = 1,
      # opponent_policy: policies.Policy | None = None,
      train_opponent: bool = True,
      use_fake_envs: bool = False,
  ):
    self._num_envs = num_envs
    self._num_players = self._num_envs * len(self.ports)
    self._rollout_length = rollout_length
    del batch_steps  # Not used yet.

    self._batch_slice_by_port = {
        port: slice(i * self._num_envs, (i + 1) * self._num_envs)
        for i, port in enumerate(self.ports)
    }
    self._train_ports = self.ports if train_opponent else (1,)
    if isinstance(dolphin_kwargs, dict):
      dolphin_kwargs = [dolphin_kwargs.copy() for _ in range(self._num_envs)]
    elif self._num_envs != len(dolphin_kwargs):
      raise ValueError(
          f'num_envs={self._num_envs} does not match '
          f'{len(dolphin_kwargs)} dolphin kwargs entries.')
    else:
      dolphin_kwargs = list(dolphin_kwargs)

    agent1_kwargs = agent_kwargs[1]
    agent2_kwargs = agent_kwargs[2]

    should_compile = agent1_kwargs['compile']
    state = agent1_kwargs['state']
    name_code_by_port = {
        port: np.asarray(
            eval_lib.get_name_codes(
                kwargs['state'],
                kwargs['name'],
                batch_size=self._num_envs,
            ),
            dtype=NAME_DTYPE,
        )
        for port, kwargs in ((1, agent1_kwargs), (2, agent2_kwargs))
    }
    self._name_code = np.concatenate(
        [name_code_by_port[port] for port in self.ports], axis=0)

    # TODO: use agent's observation_config
    # TODO: check that agent's character matches the env
    self._agents = {
        port: eval_lib.build_delayed_agent(
            console_delay=0,
            batch_size=self._num_envs,
            **kwargs)
        for port, kwargs in agent_kwargs.items()
    }

    self._prev_agent_outputs = collections.deque[dict[Port, SampleOutputs]]()
    self._prev_agent_outputs.append({
        port: agent.dummy_sample_outputs
        for port, agent in self._agents.items()
    })

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
        game_batch.game, self._rollout_length + 1)
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
    self._needs_reset = np.ones(self._num_envs, dtype=np.bool_)
    game_batch = self._env.current_game_batch(self._needs_reset)
    self._state_buffer = _make_trajectory_state_buffer(
        game_batch.game, self._rollout_length)

  def update_variables(self, updates: tp.Mapping[int, tp.Any]):
    for port, update in updates.items():
      self._agents[port].policy.set_state(update)

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
    action_buffers: dict[Port, list[SampleOutputs]] = {
        port: [] for port in self._agents
    }

    initial_states = {
        port: agent.hidden_state
        for port, agent in self._agents.items()
    }

    if verbose:
      import tqdm
      step_iter = tqdm.trange(num_steps, desc='Rollout', unit='step')
    else:
      step_iter = range(num_steps)

    def record_state(
        prev_agent_outputs: dict[Port, SampleOutputs],
    ):
      game_batch = self._env.current_game_batch(self._needs_reset)

      copy_start = time.perf_counter()
      _copy_state_slot(state_buffer, t)
      reset_buffer[t] = game_batch.needs_reset

      for port, output in prev_agent_outputs.items():
        action_buffers[port].append(output)

      timings['state_copy'] += time.perf_counter() - copy_start

      return game_batch

    # Step the sim immediately with already-delayed controller inputs, while
    # collecting a chunk of observations for one fused JAX actor call.
    for t in step_iter:
      game_batch = record_state(self._prev_agent_outputs.popleft())

      agent_outputs: dict[Port, SampleOutputs] = {}
      controllers: sim_env.Controllers = {}
      for port, agent in self._agents.items():
        batch_slice = self._batch_slice_by_port[port]

        agent_inputs = utils.map_single_structure(
            lambda x: x[batch_slice],
            (game_batch.game, game_batch.needs_reset))

        step_start = time.perf_counter()
        agent_outputs[port] = agent.step(*agent_inputs)
        controllers[port] = agent.decode_controller(
            agent_outputs[port].controller_state)

        elapsed = time.perf_counter() - step_start
        elapsed_per_port = elapsed / len(self.ports)
        for port in self.ports:
          timings[f'agent_step_{port}'] += elapsed_per_port

      self._prev_agent_outputs.append(agent_outputs)

      env_start = time.perf_counter()
      self._needs_reset = self._env.advance(controllers)
      timings['env_step'] += time.perf_counter() - env_start

    # Capture the T+1 terminal observation and assemble the learner trajectory
    # trees expected by the existing PPO path.
    record_state(self._prev_agent_outputs[0])

    build_start = time.perf_counter()
    time_major_states = state_buffer.states
    rewards = reward.compute_rewards(time_major_states)
    timings['trajectory_build'] = time.perf_counter() - build_start

    trajectories = {}
    for port, batch_slice in self._batch_slice_by_port.items():
      if port not in self._train_ports:
        continue

      agent = self._agents[port]

      states = fast_map(
          lambda x: np.asarray(x[:, batch_slice]).copy(),
          state_buffer.states)
      encoded_states = agent.policy.encode_game(states)

      batch_size = batch_slice.stop - batch_slice.start
      trajectories[port] = Trajectory(
          states=encoded_states,
          name=np.full(
              [num_steps + 1, batch_size],
              agent.name_code,
              dtype=NAME_DTYPE),
          actions=utils.batch_nest_nt(action_buffers[port]),
          rewards=rewards[:, batch_slice],
          is_resetting=reset_buffer[:, batch_slice].copy(),
          initial_state=initial_states[port],
          delayed_actions=agent.peek_n(agent.delay),
      )

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
