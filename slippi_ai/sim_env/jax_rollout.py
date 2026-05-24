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


class JaxSimRolloutWorker:
  """RolloutWorker-compatible adapter for JAX policies on one sim batch."""

  ports: tuple[Port, Port] = (1, 2)

  def __init__(
      self,
      *,
      policy: policies.Policy,
      agent_kwargs: tp.Mapping[int, dict],
      dolphin_kwargs: tp.Union[dict, tp.Sequence[dict]],
      num_envs: int,
      rollout_length: int,
      batch_steps: int = 1,
      opponent_policy: policies.Policy | None = None,
      train_opponent: bool = True,
      use_fake_envs: bool = False,
  ):
    self._num_envs = int(num_envs)
    self._num_players = self._num_envs * len(self.ports)
    self._rollout_length = int(rollout_length)
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

    # Same-policy self-play gets one actor over [port1 views, port2 views].
    # Fixed-opponent rollout keeps that same env/action layout, but samples the
    # two halves with separate policy objects and concatenates the outputs.
    if opponent_policy is None:
      self.actor = policy.build_agent(
          batch_size=self._num_players,
          name_code=self._name_code,
          compile=should_compile,
          pack_args=True,
      )
      self._opponent_actor = None
    else:
      self.actor = policy.build_agent(
          batch_size=self._num_envs,
          name_code=name_code_by_port[1],
          compile=should_compile,
          pack_args=True,
      )
      self._opponent_actor = opponent_policy.build_agent(
          batch_size=self._num_envs,
          name_code=name_code_by_port[2],
          compile=agent2_kwargs['compile'],
          pack_args=True,
      )

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

    controller_config = state['config']['embed']['controller']
    other_config = agent2_kwargs['state']['config']['embed']['controller']
    if other_config != controller_config:
      raise ValueError('JAX sim rollout requires matching controller configs.')
    if controller_config['type'] != 'default':
      raise ValueError('sim env only supports the default controller embedding')
    controller_config = controller_config['default']
    self._controller_spacing = (
        int(controller_config['axis_spacing']),
        int(controller_config['shoulder_spacing']),
    )
    self._batch_steps = max(1, int(batch_steps))
    self._env_kwargs = dict(
        num_envs=self._num_envs,
        players=[kwargs['players'] for kwargs in dolphin_kwargs],
        stage=stages,
        max_frame_id=(
            -1 if dolphin_kwargs_0['infinite_time'] else 8 * 60 * 60 - 123),
        fake=use_fake_envs,
        frame_buffer_length=self._rollout_length + self.actor._policy.delay + 2,
    )

    self._env = self._build_env()
    self._needs_reset = np.ones(self._num_envs, dtype=np.bool_)
    self._dummy_outputs = self.actor._policy.controller_head.dummy_sample_outputs(
        [self._num_players])
    game_batch = self._env.current_game_batch(self._needs_reset)
    self._state_buffer = _make_trajectory_state_buffer(
        game_batch.game, self._rollout_length)
    self._reset_buffer = np.empty(
        (self._rollout_length + 1, self._num_players), dtype=np.bool_)
    self._reset_delay_queues()

  def _reset_delay_queues(self):
    # Keep one controller in the queue even when policy delay is 0; then the
    # same queue update path covers both immediate and delayed control.
    self._delayed_controller_queue = collections.deque(
        [_copy_to_numpy_tree(self._dummy_outputs.controller_state)
         for _ in range(self.actor._policy.delay + 1)])
    self._delayed_outputs_queue = collections.deque(
        [_copy_to_numpy_tree(self._dummy_outputs)
         for _ in range(self.actor._policy.delay + 1)])

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
    self._reset_delay_queues()

  def update_variables(self, updates):
    if self._opponent_actor is None:
      self.actor._policy.set_state(updates.get(1, updates.get(2)))
      return
    if 1 in updates:
      self.actor._policy.set_state(updates[1])
    if 2 in updates:
      self._opponent_actor._policy.set_state(updates[2])

  def rollout(
      self,
      num_steps: int,
      verbose: bool = False,
  ) -> tuple[tp.Mapping[Port, Trajectory], Timings]:
    del verbose
    timings: dict[str, float] = collections.defaultdict(float)
    num_steps = int(num_steps)
    if num_steps != self._rollout_length:
      raise ValueError(
          f'JaxSimRolloutWorker was built for rollout_length={self._rollout_length}, '
          f'got rollout({num_steps}).')
    game_batch = self._env.current_game_batch(self._needs_reset)
    state_buffer = self._state_buffer
    reset_buffer = self._reset_buffer
    trajectory_actions = []
    initial_state = self.actor.hidden_state()
    if self._opponent_actor is None:
      initial_state_by_port = {
          port: utils.map_single_structure(lambda x: x[batch_slice], initial_state)
          for port, batch_slice in self._batch_slice_by_port.items()
      }
    else:
      initial_state_by_port = {1: initial_state}

    # Step the sim immediately with already-delayed controller inputs, while
    # collecting a chunk of observations for one fused JAX actor call.
    for chunk_start in range(0, num_steps, self._batch_steps):
      # `chunk_len` is normally `_batch_steps`; it is shorter only for a final
      # partial chunk when num_steps is not divisible by batch_steps.
      chunk_len = min(self._batch_steps, num_steps - chunk_start)
      chunk_inputs = []
      chunk_reset_masks = []
      controller_queue_start = list(self._delayed_controller_queue)

      for local_t in range(chunk_len):
        t = chunk_start + local_t
        copy_start = time.perf_counter()
        _copy_state_slot(state_buffer, t)
        reset_buffer[t] = game_batch.needs_reset
        timings['state_copy'] += time.perf_counter() - copy_start

        reset_mask = reset_buffer[t]
        chunk_inputs.append((state_buffer.slots[t], reset_mask))
        chunk_reset_masks.append(reset_mask)
        if np.any(reset_mask):
          _reset_delayed_controller_queue(
              self._delayed_controller_queue,
              self._dummy_outputs.controller_state,
              reset_mask,
          )

        delayed_controller = self._delayed_controller_queue.popleft()
        trajectory_actions.append(self._delayed_outputs_queue.popleft())

        env_start = time.perf_counter()
        self._needs_reset = self._env.step_encoded(
            delayed_controller,
            axis_spacing=self._controller_spacing[0],
            shoulder_spacing=self._controller_spacing[1],
        )
        game_batch = self._env.current_game_batch(self._needs_reset)
        timings['env_step'] += time.perf_counter() - env_start

      if self._opponent_actor is None:
        step_start = time.perf_counter()
        chunk_outputs = _sample_chunk(self.actor, chunk_inputs)
        # Pull the whole stacked chunk to host once. Later delayed-controller
        # replay and trajectory assembly slice these arrays many times, and
        # doing that slicing on JAX device arrays creates thousands of tiny ops.
        chunk_outputs = jax.tree.map(np.asarray, chunk_outputs)
        elapsed = time.perf_counter() - step_start
        elapsed_per_port = elapsed / len(self.ports)
        for port in self.ports:
          timings[f'agent_step_{port}'] += elapsed_per_port
      else:
        main_inputs = [
            (
                utils.map_single_structure(
                    lambda x: x[self._batch_slice_by_port[1]], game),
                reset[self._batch_slice_by_port[1]],
            )
            for game, reset in chunk_inputs
        ]
        opponent_inputs = [
            (
                utils.map_single_structure(
                    lambda x: x[self._batch_slice_by_port[2]], game),
                reset[self._batch_slice_by_port[2]],
            )
            for game, reset in chunk_inputs
        ]
        main_start = time.perf_counter()
        main_outputs = _sample_chunk(self.actor, main_inputs)
        main_outputs = jax.tree.map(np.asarray, main_outputs)
        timings['agent_step_1'] += time.perf_counter() - main_start

        opponent_start = time.perf_counter()
        opponent_outputs = _sample_chunk(self._opponent_actor, opponent_inputs)
        opponent_outputs = jax.tree.map(np.asarray, opponent_outputs)
        timings['agent_step_2'] += time.perf_counter() - opponent_start

        chunk_outputs = jax.tree.map(
            lambda a, b: np.concatenate([a, b], axis=1),
            main_outputs,
            opponent_outputs,
        )
      _replay_delayed_controller_queue(
          self._delayed_controller_queue,
          controller_queue_start,
          chunk_outputs,
          chunk_reset_masks,
          self._dummy_outputs.controller_state,
      )
      for index in range(chunk_len):
        self._delayed_outputs_queue.append(
            jax.tree.map(lambda x, i=index: x[i], chunk_outputs))

    # Capture the T+1 terminal observation and assemble the learner trajectory
    # trees expected by the existing PPO path.
    copy_start = time.perf_counter()
    _copy_state_slot(state_buffer, num_steps)
    reset_buffer[num_steps] = game_batch.needs_reset
    trajectory_actions.append(self._delayed_outputs_queue[0])
    timings['state_copy'] += time.perf_counter() - copy_start

    build_start = time.perf_counter()
    time_major_states = state_buffer.states
    encoded_states = self.actor._policy.network.encode_game(time_major_states)
    rewards = reward.compute_rewards(time_major_states)
    batched_actions = _batch_actions(trajectory_actions)
    delayed_actions = list(self._delayed_outputs_queue)[1:]
    timings['trajectory_build'] = time.perf_counter() - build_start

    trajectories = {}
    for port, batch_slice in self._batch_slice_by_port.items():
      if port not in self._train_ports:
        continue
      trajectories[port] = _trajectory_slice(
          states=encoded_states,
          actions=batched_actions,
          rewards=rewards,
          is_resetting=reset_buffer,
          initial_state=initial_state_by_port[port],
          delayed_actions=delayed_actions,
          name_code=self._name_code,
          batch_slice=batch_slice,
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
  slots: list[Game]
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


def _trajectory_slice(
    *,
    states: Game,
    actions: SampleOutputs,
    rewards: np.ndarray,
    is_resetting: np.ndarray,
    initial_state,
    delayed_actions: list[SampleOutputs],
    name_code: np.ndarray,
    batch_slice: slice,
) -> Trajectory:
  batch_size = batch_slice.stop - batch_slice.start
  return Trajectory(
      states=utils.map_single_structure(
          lambda x: np.asarray(x[:, batch_slice]).copy(), states),
      name=np.broadcast_to(
          name_code[batch_slice],
          (is_resetting.shape[0], batch_size),
      ).copy(),
      actions=utils.map_single_structure(lambda x: x[:, batch_slice], actions),
      rewards=rewards[:, batch_slice],
      is_resetting=is_resetting[:, batch_slice].copy(),
      initial_state=initial_state,
      delayed_actions=[
          utils.map_single_structure(lambda x: x[batch_slice], action)
          for action in delayed_actions
      ],
  )


def _batch_actions(actions: list[SampleOutputs]) -> SampleOutputs:
  return jax.tree.map(lambda *xs: np.stack(xs, axis=0), *actions)


def _sample_chunk(
    actor: tp.Any,
    chunk_inputs: list[tuple[Game, np.ndarray]],
) -> SampleOutputs:
  if len(chunk_inputs) == 1:
    return jax.tree.map(
        lambda x: x[None],
        actor.step_device(chunk_inputs[0][0], chunk_inputs[0][1]),
    )
  return actor.multi_step_stacked_device(chunk_inputs)


def _replay_delayed_controller_queue(
    queue: collections.deque,
    queue_start: list,
    sample_outputs: SampleOutputs,
    reset_masks: list[np.ndarray],
    neutral_controller,
):
  # The sim consumes controller inputs after policy delay, so while collecting a
  # chunk we have to step with the queue as it existed at chunk start. Once the
  # actor returns the whole time-stacked chunk, replay those sampled controllers
  # into the delay queue in order, applying neutral reset sentinels lane-wise.
  queue.clear()
  queue.extend(queue_start)
  for index, reset_mask in enumerate(reset_masks):
    if np.any(reset_mask):
      _reset_delayed_controller_queue(queue, neutral_controller, reset_mask)
    controller = jax.tree.map(
        lambda x: np.asarray(x[int(index)]).copy(),
        sample_outputs.controller_state,
    )
    queue.append(controller)
    queue.popleft()


def _copy_to_numpy_tree(value):
  return utils.map_single_structure(lambda x: np.asarray(x).copy(), value)


def _reset_delayed_controller_queue(
    queue: collections.deque,
    neutral_controller,
    reset_mask: np.ndarray,
):
  reset_mask = np.asarray(reset_mask, dtype=np.bool_)
  for index, value in enumerate(queue):
    queue[index] = utils.map_nt(
        lambda controller, neutral: _reset_leaf(
            controller, neutral, reset_mask),
        value,
        neutral_controller,
    )


def _reset_leaf(value, default, reset: np.ndarray):
  while reset.ndim < value.ndim:
    reset = reset[..., None]
  return np.where(reset, default, value)
