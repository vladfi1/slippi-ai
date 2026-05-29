"""Multiprocess wrapper for sharding sim env steps across CPU workers.

The main JAX process owns policy inference plus the shared observation/action
buffers. Each worker owns a normal `SimBatchedEnvironment` shard, waits for the
main process to publish encoded actions, steps its shard, and fills its slice of
the shared `GameBatchBuffers`.
"""

import dataclasses
import math
import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing import shared_memory
import queue
import traceback
import typing as tp

import melee
import numpy as np

from slippi_ai import utils
from slippi_ai.sim_env import env as sim_env
from slippi_ai.sim_env.observations import (
    GameBatchBuffer,
    GameBatch,
    trajectory_buffer_from_struct,
)
from slippi_ai.types import Controller, reify_tuple_type
from slippi_ai.jax.jax_utils import slice_map, fast_map

T = tp.TypeVar('T')

_BARRIER_TIMEOUT_S = 900.0


@dataclasses.dataclass(frozen=True)
class SharedArraySpec:
  name: str
  shape: tuple[int, ...]
  dtype: np.dtype


class SharedArrayOwner:
  """Main process allocator, owns shared NumPy arrays and unlinks them during
  wrapper shutdown. Every call to array() creates a shared memory block and
  returns a NumPy view."""

  def __init__(self):
    self.specs: list[SharedArraySpec] = []
    self._blocks: list[shared_memory.SharedMemory] = []

  def array(self, shape: tuple[int, ...], dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    size = math.prod(shape) * dtype.itemsize
    block = shared_memory.SharedMemory(create=True, size=size)
    array = np.ndarray(shape, dtype=dtype, buffer=block.buf)
    array.fill(0)
    self.specs.append(SharedArraySpec(block.name, shape, dtype))
    self._blocks.append(block)
    return array

  def close(self):
    for block in self._blocks:
      block.close()

  def unlink(self):
    for block in self._blocks:
      try:
        block.unlink()
      except FileNotFoundError:
        pass


class SharedArrayAttacher:
  """Worker-side mirror of SharedArrayOwner. Attaches to arrays in the same
  order the owner allocated them."""

  def __init__(self, specs: tp.Sequence[SharedArraySpec]):
    self._specs = tuple(specs)
    self._index = 0
    self._blocks: list[shared_memory.SharedMemory] = []

  def array(self, shape: tuple[int, ...], dtype) -> np.ndarray:
    spec = self._specs[self._index]
    self._index += 1
    dtype = np.dtype(dtype)
    if shape != spec.shape or dtype != spec.dtype:
      raise ValueError(
          f'shared array mismatch: expected {spec.shape}/{spec.dtype}, '
          f'got {shape}/{dtype}')
    block = shared_memory.SharedMemory(name=spec.name)
    self._blocks.append(block)
    return np.ndarray(spec.shape, dtype=dtype, buffer=block.buf)

  def close(self):
    for block in self._blocks:
      block.close()

class Allocation(tp.NamedTuple, tp.Generic[T]):
  arrays: T
  specs: T
  blocks: list[shared_memory.SharedMemory]

  def close(self):
    for block in self.blocks:
      block.close()

  def unlink(self):
    for block in self.blocks:
      try:
        block.unlink()
      except FileNotFoundError:
        pass

def allocate(
    struct_t: type[T],
    shape: tuple[int, ...],
) -> Allocation[T]:
  reified = reify_tuple_type(struct_t)
  blocks: list[shared_memory.SharedMemory] = []

  def _shared_memory(dtype):
    dtype = np.dtype(dtype)
    size = math.prod(shape) * dtype.itemsize
    block = shared_memory.SharedMemory(create=True, size=size)
    blocks.append(block)
    return block

  block_struct = utils.map_nt(_shared_memory, reified)

  def _array(dtype, block: shared_memory.SharedMemory):
    array = np.ndarray(shape, dtype=dtype, buffer=block.buf)
    array.fill(0)
    return array

  arrays = utils.map_nt(_array, reified, block_struct)

  def _spec(dtype, block: shared_memory.SharedMemory):
    return SharedArraySpec(block.name, shape, dtype)

  specs = utils.map_nt(_spec, reified, block_struct)

  return Allocation(
      arrays=arrays,
      specs=specs,
      blocks=blocks,
  )

def alloc_from_specs(specs: T) -> Allocation[T]:
  blocks: list[shared_memory.SharedMemory] = []

  def _array(spec: SharedArraySpec):
    block = shared_memory.SharedMemory(name=spec.name)
    blocks.append(block)
    return np.ndarray(spec.shape, dtype=spec.dtype, buffer=block.buf)

  arrays = utils.map_nt(_array, specs)
  return Allocation(arrays, specs, blocks)

def _copy_into_slice(src: np.ndarray, dst: np.ndarray, dst_slice: slice):
  dst[dst_slice] = src

class MultiprocessSimEnvironment:
  """Process-sharded `SimBatchedEnvironment` wrapper for JAX rollout."""

  def __init__(
      self,
      *,
      num_envs: int,
      inner_batch_size: int,
      players: tp.Sequence[sim_env.PlayerConfigs],
      stage: tp.Sequence[melee.Stage],
      frame_buffer_length: int = 128,
      runahead: int = 0,  # number of extra actions that can be pushed
      max_frame_id: int = -1,
      data_dir: str | None = None,
      fake: bool = False,
  ):
    self._num_envs = num_envs
    self._inner_batch_size = inner_batch_size
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
    self._processes = []

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
              frame_buffer_length=frame_buffer_length,
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
    if self._closed:
      return
    self._closed = True
    self._stop_event.set()
    # free the workers from blocking on actions
    for events in self._action_written_events:
      for event in events:
        event.set()
    for process in self._processes:
      process.join(timeout=5.0)
      if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
      process.close()
    self._obs_alloc.close()
    self._action_alloc.close()
    self._misc_owner.close()
    self._obs_alloc.unlink()
    self._action_alloc.unlink()
    self._misc_owner.unlink()

  # @property
  # def game_batch_buffer(self) -> GameBatchBuffer:
  #   return self._trajectory_buffers.slots[self._current_index]

  def pop(self) -> GameBatch:
    self._check_worker_errors()

    self._pushed_minus_popped -= 1
    if self._pushed_minus_popped < 0:
      raise RuntimeError('not enough actions pushed')

    # Wait until all workers have written their observations
    # TODO: is there a more efficient way to do this? A semaphore could work
    # but it doesn't allow you to acquire N at once.
    for event in self._obs_written_events[self._obs_index]:
      event.wait(timeout=_BARRIER_TIMEOUT_S)
      event.clear()

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
    frame_buffer_length: int,
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
        frame_buffer_length=frame_buffer_length,
        max_frame_id=max_frame_id,
        data_dir=data_dir,
        fake=fake,
    )
    del frame_buffer_length  # don't confuse with env runahead
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
