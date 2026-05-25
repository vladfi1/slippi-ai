"""Multiprocess wrapper for sharding sim env steps across CPU workers.

The main JAX process owns policy inference plus the shared observation/action
buffers. Each worker owns a normal `SimBatchedEnvironment` shard, waits for the
main process to publish encoded actions, steps its shard, and fills its slice of
the shared `GameBatchBuffers`.
"""

import dataclasses
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import traceback
import typing as tp

import melee
import numpy as np

from slippi_ai.sim_env import env as sim_env
from slippi_ai.sim_env.observations import (
    ArrayAllocator,
    GameBatch,
    GameBatchBuffers,
)
from slippi_ai.types import Buttons, Controller, Stick


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


class MultiprocessSimEnvironment:
  """Process-sharded `SimBatchedEnvironment` wrapper for JAX rollout."""

  def __init__(
      self,
      *,
      num_envs: int,
      inner_batch_size: int,
      players: tp.Sequence[sim_env.PlayerConfigs],
      stage: tp.Sequence[melee.Stage],
      frame_buffer_length: int,
      max_frame_id: int = -1,
      data_dir: str | None = None,
      fake: bool = False,
  ):
    self._num_envs = int(num_envs)
    self._inner_batch_size = int(inner_batch_size)
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
    self._obs_owner = SharedArrayOwner()
    self._action_owner = SharedArrayOwner()
    self._misc_owner = SharedArrayOwner()
    self._game_batch = GameBatchBuffers.with_allocator(
        self._num_envs, self._obs_owner.array)
    self._shared_action = shared_action_buffer(
        self._num_envs * 2, self._action_owner.array)
    self._needs_reset = self._misc_owner.array((self._num_envs,), np.bool_)
    self._episode_ids = self._misc_owner.array((self._num_envs,), np.int64)
    self._controller_spacing = self._misc_owner.array((2,), np.int32)

    # Two barriers define one synchronous sim frame: parent publishes actions,
    # workers step their shards, then parent reads the completed observations.
    self._context = mp.get_context('spawn')
    self._actions_ready = self._context.Barrier(self._num_workers + 1)
    self._observations_ready = self._context.Barrier(self._num_workers + 1)
    self._stop_event = self._context.Event()
    self._completed_queue = self._context.Queue()
    self._error_queue = self._context.Queue()
    self._processes = []

    # Each worker gets a contiguous env range. The policy-facing action buffer
    # remains global: [all port-1 perspectives, all port-2 perspectives].
    for worker_id in range(self._num_workers):
      start = worker_id * self._inner_batch_size
      stop = start + self._inner_batch_size
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
              max_frame_id=max_frame_id,
              data_dir=data_dir,
              fake=fake,
              obs_specs=self._obs_owner.specs,
              action_specs=self._action_owner.specs,
              misc_specs=self._misc_owner.specs,
              actions_ready=self._actions_ready,
              observations_ready=self._observations_ready,
              stop_event=self._stop_event,
              completed_queue=self._completed_queue,
              error_queue=self._error_queue,
          ),
      )
      process.start()
      self._processes.append(process)

    # Workers publish their initial observations before the first policy call.
    self._wait_for_observations('initial observations')

  def stop(self):
    if self._closed:
      return
    self._closed = True
    self._stop_event.set()
    try:
      self._actions_ready.wait(timeout=1.0)
    except Exception:
      pass
    for process in self._processes:
      process.join(timeout=5.0)
      if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
      process.close()
    self._obs_owner.close()
    self._action_owner.close()
    self._misc_owner.close()
    self._obs_owner.unlink()
    self._action_owner.unlink()
    self._misc_owner.unlink()

  def current_game_batch(self, needs_reset: np.ndarray | None = None) -> GameBatch:
    del needs_reset
    return GameBatch(
        game=self._game_batch.game,
        needs_reset=self._game_batch.needs_reset,
    )

  def step_encoded(
      self,
      controller_state: Controller,
      *,
      axis_spacing: int,
      shoulder_spacing: int,
  ) -> np.ndarray:
    # Copy JAX outputs into shared host buffers, release all workers for one sim
    # frame, then wait until every shard has filled its observation slice.
    self._controller_spacing[:] = (axis_spacing, shoulder_spacing)
    copy_action_to_shared_buffer(self._shared_action, controller_state)
    self._wait_for_actions()
    self._wait_for_observations('step observations')
    return self._needs_reset

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

  def _wait_for_actions(self):
    self._check_worker_errors()
    try:
      _barrier_wait(self._actions_ready, 'action release')
    except RuntimeError:
      self._check_worker_errors()
      raise

  def _wait_for_observations(self, label: str):
    self._check_worker_errors()
    try:
      _barrier_wait(self._observations_ready, label)
    except RuntimeError:
      self._check_worker_errors()
      raise
    self._check_worker_errors()

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
    max_frame_id: int,
    data_dir: str | None,
    fake: bool,
    obs_specs: tp.Sequence[SharedArraySpec],
    action_specs: tp.Sequence[SharedArraySpec],
    misc_specs: tp.Sequence[SharedArraySpec],
    actions_ready,
    observations_ready,
    stop_event,
    completed_queue,
    error_queue,
):
  # Recreate NumPy views over the parent's shared-memory blocks in the exact
  # allocation order used during parent setup.
  obs_attacher = SharedArrayAttacher(obs_specs)
  action_attacher = SharedArrayAttacher(action_specs)
  misc_attacher = SharedArrayAttacher(misc_specs)
  env = None
  try:
    game_batch = GameBatchBuffers.with_allocator(total_envs, obs_attacher.array)
    shared_action = shared_action_buffer(total_envs * 2, action_attacher.array)
    needs_reset = misc_attacher.array((total_envs,), np.bool_)
    episode_ids = misc_attacher.array((total_envs,), np.int64)
    controller_spacing = misc_attacher.array((2,), np.int32)
    env = sim_env.SimBatchedEnvironment(
        num_envs=batch_size,
        players=players,
        stage=stages,
        frame_buffer_length=frame_buffer_length,
        max_frame_id=max_frame_id,
        data_dir=data_dir,
        fake=fake,
    )
    env_slice = slice(offset, offset + batch_size)
    p1_slice = env_slice
    p2_slice = slice(total_envs + offset, total_envs + offset + batch_size)

    # Seed the parent-visible GameBatch with the shard's starting state so the
    # first policy inference sees valid observations.
    local_reset = np.ones(batch_size, dtype=np.bool_)
    needs_reset[env_slice] = local_reset
    episode_ids[env_slice] = 0
    game_batch.fill_slice(
        env.buffers.gamestate_view[env.cursor],
        local_reset,
        env_slice,
        env._last_controllers,
        controller_slice=slice(None),
    )
    _barrier_wait(observations_ready, f'worker {worker_id} initial observations')

    while not stop_event.is_set():
      # The parent has already copied actions into shared_action. This worker
      # consumes only its port-1 and port-2 slices, steps its local EnvBatch, and
      # writes results back into the same global GameBatch buffers.
      _barrier_wait(actions_ready, f'worker {worker_id} action wait')
      if stop_event.is_set():
        break
      local_reset = env.step_encoded_slices(
          shared_action,
          player_slices=(p1_slice, p2_slice),
          axis_spacing=controller_spacing[0],
          shoulder_spacing=controller_spacing[1],
      )
      needs_reset[env_slice] = local_reset
      episode_ids[env_slice] = env._episode_ids
      game_batch.fill_slice(
          env.buffers.gamestate_view[env.cursor],
          local_reset,
          env_slice,
          env._last_controllers,
          controller_slice=slice(None),
      )
      completed_games = env.pop_completed_games()
      if completed_games:
        for game in completed_games:
          game['env_id'] += offset
        completed_queue.put(completed_games)
      _barrier_wait(observations_ready, f'worker {worker_id} observations')
  except BaseException:
    error_queue.put(traceback.format_exc())
    _abort_barrier(actions_ready)
    _abort_barrier(observations_ready)
  finally:
    if env is not None:
      env.stop()
    obs_attacher.close()
    action_attacher.close()
    misc_attacher.close()


def shared_action_buffer(
    total_players: int,
    allocate_array: ArrayAllocator,
) -> Controller:
  # Encoded controller buckets are compact uint8/bool values produced by the
  # JAX controller head, not libmelee float stick coordinates.
  shape = (total_players,)
  return Controller(
      main_stick=Stick(
          x=allocate_array(shape, np.uint8),
          y=allocate_array(shape, np.uint8),
      ),
      c_stick=Stick(
          x=allocate_array(shape, np.uint8),
          y=allocate_array(shape, np.uint8),
      ),
      shoulder=allocate_array(shape, np.uint8),
      buttons=Buttons(**{
          name: allocate_array(shape, np.bool_)
          for name in Buttons._fields
      }),
  )


def copy_action_to_shared_buffer(dst: Controller, src: Controller):
  np.copyto(dst.main_stick.x, np.asarray(src.main_stick.x), casting='unsafe')
  np.copyto(dst.main_stick.y, np.asarray(src.main_stick.y), casting='unsafe')
  np.copyto(dst.c_stick.x, np.asarray(src.c_stick.x), casting='unsafe')
  np.copyto(dst.c_stick.y, np.asarray(src.c_stick.y), casting='unsafe')
  np.copyto(dst.shoulder, np.asarray(src.shoulder), casting='unsafe')
  for name in Buttons._fields:
    np.copyto(getattr(dst.buttons, name), np.asarray(getattr(src.buttons, name)))

def _barrier_wait(barrier, label: str):
  try:
    barrier.wait(timeout=_BARRIER_TIMEOUT_S)
  except Exception as exc:
    raise RuntimeError(f'timed out or broke barrier during {label}') from exc


def _abort_barrier(barrier):
  try:
    barrier.abort()
  except Exception:
    pass
