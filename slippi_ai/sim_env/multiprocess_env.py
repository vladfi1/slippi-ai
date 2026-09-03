"""Multiprocess wrapper for sharding sim env steps across CPU workers.

The main JAX process owns policy inference plus the shared observation/action
buffers. Each worker owns a normal `SimBatchedEnvironment` shard, waits for the
main process to publish encoded actions, steps its shard, and fills its slice of
the shared `GameBatchBuffers`.
"""

import multiprocessing as mp
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event
import logging
import queue
import time
import traceback
import typing as tp

import melee
import numpy as np

from slippi_ai.sim_env import env as sim_env
from slippi_ai.sim_env.observations import (
    GameBatchBuffer,
    GameBatch,
    trajectory_buffer_from_struct,
)
from slippi_ai.shared_arrays import (
    SharedArraySpec,
    SharedArrayOwner,
    SharedArrayAttacher,
    Allocation,
    allocate,
    alloc_from_specs,
)
from slippi_ai.types import Controller
from slippi_ai.jax.jax_utils import slice_map, fast_map

T = tp.TypeVar('T')

_BARRIER_TIMEOUT_S = 900.0
_ERROR_POLL_S = 1.0


def _copy_into_slice(src: np.ndarray, dst: np.ndarray, dst_slice: slice):
  dst[dst_slice] = src

def _safe_close(process: BaseProcess, timeout: float = 5):
  process.join(timeout=timeout)
  process.terminate()
  process.join(timeout=timeout)
  process.close()

class MultiprocessSimEnvironment:
  """Process-sharded `SimBatchedEnvironment` wrapper for JAX rollout."""

  def __init__(
      self,
      *,
      num_envs: int,
      inner_batch_size: int,
      players: tp.Sequence[sim_env.PlayerConfigs],
      stage: tp.Sequence[melee.Stage],
      runahead: int = 0,  # number of extra actions that can be pushed
      max_frame_id: int = -1,
      data_dir: str | None = None,
      fake: bool = False,
  ):
    self._num_envs = num_envs
    self._inner_batch_size = inner_batch_size
    # A non-positive size would give num_workers <= 0, spawning no workers at
    # all. The barriers below iterate over per-worker events, so they'd pass
    # trivially and rollouts would silently read never-written buffers.
    if self._inner_batch_size <= 0:
      raise ValueError(
          f'inner_batch_size must be positive, got {self._inner_batch_size}.')
    if self._num_envs % self._inner_batch_size:
      raise ValueError(
          f'num_envs={self._num_envs} must be divisible by '
          f'inner_batch_size={self._inner_batch_size}.')
    self._num_workers = self._num_envs // self._inner_batch_size
    self._closed = False
    self._stage_by_env = np.asarray(stage, dtype=object)

    # Shared buffers are allocated once by the parent. Workers attach to the
    # same blocks and fill only their shard, so rollout can read one global
    # GameBatch without gathering per-worker Python objects every frame.

    self._env_runahead = runahead + 1
    num_players = self._num_envs * 2
    self._player_slices = {
        1: slice(0, num_envs),
        2: slice(num_envs, 2 * num_envs),
    }

    self._obs_alloc = allocate(GameBatch, (self._env_runahead, num_players))
    self._obs_buffer = trajectory_buffer_from_struct(self._obs_alloc.arrays)

    self._action_alloc = allocate(Controller, (self._env_runahead, num_players))
    self._action_buffer = trajectory_buffer_from_struct(self._action_alloc.arrays)

    self._misc_owner = SharedArrayOwner()
    self._episode_ids = self._misc_owner.array((self._num_envs,), np.int64)

    # Parent sets the target trajectory frame before releasing workers. Workers
    # fill their shard of that frame, then parent marks it current after the
    # observation barrier completes.
    self._obs_index = 0
    self._action_index = 0
    self._pushed_minus_popped = 1

    # Two barriers define one synchronous sim frame: parent publishes actions,
    # workers step their shards, then parent reads the completed observations.
    self._context = mp.get_context('spawn')
    self._obs_written_events = [
        [self._context.Event() for _ in range(self._num_workers)]
        for _ in range(self._env_runahead)
    ]
    self._action_written_events = [
        [self._context.Event() for _ in range(self._num_workers)]
        for _ in range(self._env_runahead)
    ]
    # TODO: try using events or semaphores instead of barriers?
    self._stop_event = self._context.Event()
    self._completed_queue = self._context.Queue()
    self._error_queue = self._context.Queue()
    self._processes: list[BaseProcess] = []

    # Each worker gets a contiguous env range. The policy-facing action buffer
    # remains global: [all port-1 perspectives, all port-2 perspectives].
    for worker_id in range(self._num_workers):
      start = worker_id * self._inner_batch_size
      stop = start + self._inner_batch_size

      obs_events = [event[worker_id] for event in self._obs_written_events]
      action_events = [event[worker_id] for event in self._action_written_events]

      process = self._context.Process(
          target=_worker_main,
          name=f'sim-env-worker-{worker_id}',
          kwargs=dict(
              worker_id=worker_id,
              offset=start,
              batch_size=self._inner_batch_size,
              total_envs=self._num_envs,
              players=players[start:stop],
              stages=stage[start:stop],
              env_runahead=self._env_runahead,
              max_frame_id=max_frame_id,
              data_dir=data_dir,
              fake=fake,
              obs_specs=self._obs_alloc.specs,
              action_specs=self._action_alloc.specs,
              misc_specs=self._misc_owner.specs,
              obs_written_events=obs_events,
              action_written_events=action_events,
              stop_event=self._stop_event,
              completed_queue=self._completed_queue,
              error_queue=self._error_queue,
          ),
      )
      process.start()
      self._processes.append(process)

  def stop(self):
    logging.info('Stopping multiprocess sim environment')
    if self._closed:
      return
    self._closed = True
    self._stop_event.set()
    # free the workers from blocking on actions
    for events in self._action_written_events:
      for event in events:
        event.set()
    for process in self._processes:
      _safe_close(process)
    self._obs_alloc.close()
    self._action_alloc.close()
    self._misc_owner.close()
    self._obs_alloc.unlink()
    self._action_alloc.unlink()
    self._misc_owner.unlink()

  # @property
  # def game_batch_buffer(self) -> GameBatchBuffer:
  #   return self._trajectory_buffers.slots[self._current_index]

  def _wait_obs_written(self, index: int, *, clear: bool):
    """Wait until all workers have written their observation shard.

    Workers run up to `runahead` steps behind the pushed actions, so a slot
    may not have been written yet even though its action has been consumed.
    """
    # TODO: is there a more efficient way to do this? A semaphore could work
    # but it doesn't allow you to acquire N at once.
    # Poll in short increments: a worker that raised (error_queue) or died
    # without cleanup (native crash) would otherwise present as a silent
    # hang until the full barrier timeout.
    deadline = time.monotonic() + _BARRIER_TIMEOUT_S
    for worker_id, event in enumerate(self._obs_written_events[index]):
      while not event.wait(timeout=_ERROR_POLL_S):
        self._check_worker_errors()
        process = self._processes[worker_id]
        if not process.is_alive():
          raise RuntimeError(
              f'{process.name} died with exit code {process.exitcode}')
        if time.monotonic() > deadline:
          raise RuntimeError('timed out waiting for worker observations')
      if clear:
        event.clear()

  def pop(self) -> GameBatch:
    self._check_worker_errors()

    self._pushed_minus_popped -= 1
    if self._pushed_minus_popped < 0:
      raise RuntimeError('not enough actions pushed')

    self._wait_obs_written(self._obs_index, clear=True)

    game_batch = self._obs_buffer.slots[self._obs_index]
    self._obs_index = (self._obs_index + 1) % self._env_runahead
    return game_batch

  def push(self, controllers: sim_env.Controllers):
    self._check_worker_errors()

    self._pushed_minus_popped += 1
    if self._pushed_minus_popped > self._env_runahead:
      raise RuntimeError('too many actions pushed')

    for port, controller in controllers.items():
      player_slice = self._player_slices[port]
      fast_map(
          lambda src, dst: _copy_into_slice(src, dst, player_slice),
          controller, self._action_buffer.slots[self._action_index])

    for event in self._action_written_events[self._action_index]:
      event.set()

    self._action_index = (self._action_index + 1) % self._env_runahead

  def peek(self) -> GameBatch:
    if not (self._pushed_minus_popped > 0):
      raise RuntimeError('no observations available')
    # Don't clear the events so that a subsequent pop() also completes.
    self._wait_obs_written(self._obs_index, clear=False)
    return self._obs_buffer.slots[self._obs_index]

  def active_games(self) -> list[dict[str, int | str]]:
    return [
        {
            'env_id': env_id,
            'episode_id': int(self._episode_ids[env_id]),
            'stage': self._stage_by_env[env_id].name,
            'stage_id': int(self._stage_by_env[env_id].value),
        }
        for env_id in range(self._num_envs)
    ]

  def pop_completed_games(self) -> list[dict[str, tp.Any]]:
    games = []
    while True:
      try:
        games.extend(self._completed_queue.get_nowait())
      except queue.Empty:
        return games

  def _check_worker_errors(self):
    try:
      error = self._error_queue.get_nowait()
    except queue.Empty:
      return
    raise RuntimeError(error)


def _worker_main(
    *,
    worker_id: int,
    offset: int,
    batch_size: int,
    total_envs: int,
    players: tp.Sequence[sim_env.PlayerConfigs],
    stages: tp.Sequence[melee.Stage],
    env_runahead: int,
    max_frame_id: int,
    data_dir: str | None,
    fake: bool,
    obs_specs: GameBatch,
    action_specs: Controller,
    misc_specs: tp.Sequence[SharedArraySpec],
    obs_written_events: list[Event],
    action_written_events: list[Event],
    stop_event: Event,
    completed_queue: mp.Queue,
    error_queue: mp.Queue,
):
  obs_alloc = alloc_from_specs(obs_specs)
  action_alloc = alloc_from_specs(action_specs)
  misc_attacher = SharedArrayAttacher(misc_specs)
  env = None
  try:
    obs_buffer = trajectory_buffer_from_struct(obs_alloc.arrays)
    assert len(obs_buffer) == env_runahead
    assert len(obs_written_events) == env_runahead

    game_batch_buffers = [
        GameBatchBuffer(game_batch) for game_batch in obs_buffer.slots]

    action_buffer = trajectory_buffer_from_struct(action_alloc.arrays)
    assert len(action_buffer) == env_runahead
    assert len(action_written_events) == env_runahead

    episode_ids = misc_attacher.array((total_envs,), np.int64)
    env = sim_env.SimBatchedEnvironment(
        num_envs=batch_size,
        players=players,
        stage=stages,
        max_frame_id=max_frame_id,
        data_dir=data_dir,
        fake=fake,
        # Seed matches by global env index so sharding doesn't change games.
        seed_offset=offset,
    )
    env_slice = slice(offset, offset + batch_size)
    p1_slice = env_slice
    p2_slice = slice(total_envs + offset, total_envs + offset + batch_size)

    port_slices = {1: p1_slice, 2: p2_slice}

    # Seed the parent-visible GameBatch with the shard's starting state so the
    # first policy inference sees valid observations.
    local_reset = np.ones(batch_size, dtype=np.bool_)
    episode_ids[env_slice] = 0

    index = 0

    while not stop_event.is_set():
      # The trajectory buffer knows the full batch size, and so can figure out
      # where to write p2's data just from p1's slice.
      env.write_current_game(
          game_batch_buffers[index],
          local_reset,
          env_slice=env_slice,
      )
      # Notify the parent env that obs have been written
      obs_written_events[index].set()

      # The parent has already copied actions into shared_action. This worker
      # consumes only its port-1 and port-2 slices, steps its local EnvBatch, and
      # writes results back into the same global GameBatch buffers.

      action_written_events[index].wait()
      action_written_events[index].clear()

      if stop_event.is_set():
        break

      controllers = {
          port: slice_map(s, action_buffer.slots[index])
          for port, s in port_slices.items()
      }
      local_reset = env.advance(controllers)

      episode_ids[env_slice] = env.episode_ids
      completed_games = env.pop_completed_games()
      if completed_games:
        for game in completed_games:
          game['env_id'] += offset
        completed_queue.put(completed_games)

      index = (index + 1) % env_runahead
  except BaseException:
    error_queue.put(traceback.format_exc())
  finally:
    if env is not None:
      env.stop()
    obs_alloc.close()
    action_alloc.close()
    misc_attacher.close()
