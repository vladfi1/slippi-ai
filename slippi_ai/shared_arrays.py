"""Helpers for sharing structured NumPy arrays between processes.

The parent process allocates shared-memory blocks for a NamedTuple structure
of arrays (dtypes derived from type annotations via `reify_tuple_type`), and
workers attach to the same blocks by spec.
"""

import dataclasses
import math
from multiprocessing import shared_memory
import typing as tp

import numpy as np

from slippi_ai import utils
from slippi_ai.types import reify_tuple_type

T = tp.TypeVar('T')


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
