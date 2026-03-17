import abc
import itertools
import threading
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventMP
import queue
import typing as tp


class Dataset[T](abc.ABC, tp.Iterator[T]):

  def stop(self):
    """Cleans up any resources used by the dataset."""

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
    context = mp.get_context('forkserver')
    self.input_queue = context.Queue()
    self.output_queue = context.Queue()
    self.stop_event = context.Event()
    self.workers = [
        context.Process(
            target=_mp_map_worker,
            args=(
                map_fn,
                self.input_queue,
                self.output_queue,
                self.stop_event,
            ),
            daemon=True,
        ) for _ in range(num_workers)
    ]

    for _ in range(buffer):
      self.input_queue.put(next(self.dataset))

    for worker in self.workers:
      worker.start()

  def __next__(self) -> U:
    item = next(self.dataset)
    self.input_queue.put(item)
    return self.output_queue.get()

  def stop(self):
    self.stop_event.set()
    for worker in self.workers:
      worker.join()

    self.output_queue.close()
    self.input_queue.cancel_join_thread()
    self.input_queue.close()

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