# TODO: implement WDS interfaces?

import abc
import itertools
import random
import threading
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventMP
import queue
import typing as tp


class Dataset[T](abc.ABC, tp.Iterator[T]):

  def stop(self):
    """Cleans up any resources used by the dataset."""

  def map[U](self, map_fn: tp.Callable[[T], U]) -> 'Dataset[U]':
    """Maps a function over the dataset."""
    return MapDataset(self, map_fn)

  def map_iter[U](self, map_fn: tp.Callable[[T], tp.Iterable[U]]) -> 'Dataset[U]':
    """Maps a function that returns an iterable over the dataset, flattening the result."""
    return MapIter(self, map_fn)

  def map_mp[U](self, map_fn: tp.Callable[[T], U], num_workers: int, buffer: int) -> 'Dataset[U]':
    """Maps a function over the dataset using multiprocessing."""
    return MPMap(self, map_fn, num_workers=num_workers, buffer=buffer)

  def shuffle(self, buffer: int, seed: int = 0) -> 'Dataset[T]':
    """Shuffles the dataset using a buffer."""
    return ShuffleDataset(self, buffer, seed)

class IteratorDataset[T](Dataset[T]):

  def __init__(self, iterator: tp.Iterator[T]):
    self.iterator = iterator

  def __next__(self) -> T:
    return next(self.iterator)

  def stop(self):
    pass

class PrefetchMT[T](Dataset[T]):
  """Prefetches items from a dataset using multithreading."""

  def __init__(self, dataset: Dataset[T], buffer: int):
    self.dataset = dataset
    self.buffer = buffer

    self.queue = queue.Queue(buffer)
    self.should_stop = False
    self.exhausted = False
    self.process = threading.Thread(
        target=self._prefetch_worker,
        daemon=True,
    )
    self.process.start()

  def _prefetch_worker(
      self,
      timeout: float = 1.0,
  ):
    for item in self.dataset:
      if self.should_stop:
        return

      while True:
        try:
          self.queue.put(item, timeout=timeout)
          break
        except queue.Full:
          if self.should_stop:
            return

    self.exhausted = True

  def __next__(self) -> T:
    while True:
      try:
        return self.queue.get(timeout=0.1)
      except queue.Empty:
        if self.exhausted:
          raise StopIteration

  def stop(self):
    self.should_stop = True
    self.process.join()
    self.dataset.stop()

class FilterDataset[T](Dataset[T]):
  """Filters items from a dataset based on a predicate."""

  def __init__(self, dataset: Dataset[T], predicate: tp.Callable[[T], bool]):
    self.dataset = dataset
    self.predicate = predicate

  def __next__(self) -> T:
    while True:
      item = next(self.dataset)
      if self.predicate(item):
        return item

  def stop(self):
    self.dataset.stop()

class MapDataset[T, U](Dataset[U]):
  """Maps a function over a dataset."""

  def __init__(self, dataset: Dataset[T], map_fn: tp.Callable[[T], U]):
    self.dataset = dataset
    self.map_fn = map_fn

  def __next__(self) -> U:
    item = next(self.dataset)
    return self.map_fn(item)

  def stop(self):
    self.dataset.stop()

class MapIter[T, U](Dataset[U]):
  """Maps a function that returns an iterable over a dataset, flattening the result."""

  def __init__(self, dataset: Dataset[T], map_fn: tp.Callable[[T], tp.Iterable[U]]):
    self.dataset = dataset
    self.map_fn = map_fn
    self._iterator = self._iterate()

  def _iterate(self) -> tp.Iterator[U]:
    for item in self.dataset:
      yield from self.map_fn(item)

  def __next__(self) -> U:
    return next(self._iterator)

  def stop(self):
    self.dataset.stop()

def _mp_map_worker[T, U](
    map_fn: tp.Callable[[T], U],
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    stop_event: EventMP,
    timeout: float = 1.0,
):
  try:
    while not stop_event.is_set():
      while True:
        try:
          item = input_queue.get(timeout=timeout)
          break
        except queue.Empty:
          if stop_event.is_set():
            return

      result = map_fn(item)

      while True:
        try:
          output_queue.put(result, timeout=timeout)
          break
        except queue.Full:
          if stop_event.is_set():
            return
  finally:
    # Note: output queue's join_thread must be cancelled, otherwise the worker
    # will hang on exit if the output queue is full.
    output_queue.cancel_join_thread()

class MPMap[T, U](Dataset[U]):
  """Maps a function over a dataset using multiprocessing."""

  def __init__(self, dataset: Dataset[T], map_fn: tp.Callable[[T], U], num_workers: int, buffer: int):
    self.dataset = dataset
    self.map_fn = map_fn
    self.buffer_size = buffer
    context = mp.get_context('forkserver')
    self.queues = [(context.Queue(), context.Queue()) for _ in range(num_workers)]
    self.stop_event = context.Event()
    self.workers = [
        context.Process(
            target=_mp_map_worker,
            args=(
                map_fn,
                input_queue,
                output_queue,
                self.stop_event,
            ),
            daemon=True,
        ) for input_queue, output_queue in self.queues
    ]

    # Queue up buffered items
    for _ in range(buffer):
      for input_queue, _ in self.queues:
        input_queue.put(next(self.dataset))

    for worker in self.workers:
      worker.start()

    self._iterator = self._iter()

  def _iter(self) -> tp.Iterator[U]:
    queue_iter = itertools.cycle(self.queues)

    for item in self.dataset:
      input_queue, output_queue = next(queue_iter)
      input_queue.put(item)
      yield output_queue.get()

    # Drain remaining items from output queues
    for _ in range(self.buffer_size):
      for _, output_queue in queue_iter:
        yield output_queue.get()

  def __next__(self) -> U:
    return next(self._iterator)

  def stop(self):
    self.stop_event.set()
    for worker in self.workers:
      worker.join()

    for input_queue, output_queue in self.queues:
      output_queue.close()
      input_queue.cancel_join_thread()
      input_queue.close()

    self.dataset.stop()

class ShuffleDataset[T](Dataset[T]):
  """Shuffles items from a dataset using a buffer."""

  def __init__(self, dataset: Dataset[T], buffer: int, seed: int = 0):
    self.dataset = dataset
    self.buffer = buffer
    self.rng = random.Random(seed)
    self._iterator = self._iterate()

  def _iterate(self) -> tp.Iterator[T]:
    buffer = []
    for item in self.dataset:
      if len(buffer) < self.buffer:
        buffer.append(item)
      else:
        idx = self.rng.randint(0, self.buffer - 1)
        yield buffer[idx]
        buffer[idx] = item

    self.rng.shuffle(buffer)
    yield from buffer

  def __next__(self) -> T:
    return next(self._iterator)

  def stop(self):
    self.dataset.stop()

class ChildDataset[T](Dataset[T]):

  def __init__(self, dataset: Dataset[T]):
    self.dataset = dataset

  def __next__(self) -> T:
    return next(self.dataset)

  def stop(self):
    self.dataset.stop()

def split_dataset[T](dataset: Dataset[T], num_splits: int) -> list[Dataset[T]]:
  """Splits a dataset into multiple datasets that share the underlying dataset."""
  return [ChildDataset(dataset) for _ in range(num_splits)]

class InterleaveDataset[T](Dataset[T]):
  """Interleaves multiple datasets into a single dataset."""

  def __init__(self, datasets: list[Dataset[T]]):
    self.datasets = datasets
    self.dataset_iter = itertools.cycle(self.datasets)

  def __next__(self) -> T:
    return next(next(self.dataset_iter))

  def stop(self):
    for dataset in self.datasets:
      dataset.stop()

class FlattenDataset[T](Dataset[T]):
  """Flattens a dataset of iterables into a single dataset."""

  def __init__(self, dataset: Dataset[tp.Iterable[T]]):
    self.dataset = dataset
    self._iterator = self._iterate()

  def _iterate(self) -> tp.Iterator[T]:
    for iterable in self.dataset:
      for item in iterable:
        yield item

  def __next__(self) -> T:
    return next(self._iterator)

  def stop(self):
    self.dataset.stop()

class ZipDataset[T](Dataset[tuple[T, ...]]):
  """Zips multiple datasets together."""

  def __init__(self, datasets: list[Dataset]):
    self.datasets = datasets

  def __next__(self) -> tuple[T, ...]:
    return tuple(next(dataset) for dataset in self.datasets)

  def stop(self):
    for dataset in self.datasets:
      dataset.stop()