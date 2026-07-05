"""Tests for slippi_ai.datasets."""

from contextlib import contextmanager
import signal
import time
import unittest

from slippi_ai import datasets


@contextmanager
def time_limit(seconds: float):
  """Fails the test instead of hanging forever if the body takes too long."""
  def handler(signum, frame):
    raise TimeoutError(f'Timed out after {seconds}s')
  old_handler = signal.signal(signal.SIGALRM, handler)
  signal.setitimer(signal.ITIMER_REAL, seconds)
  try:
    yield
  finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, old_handler)


class RecordingDataset(datasets.Dataset[int]):
  """A dataset wrapping an iterator that records whether stop() was called."""

  def __init__(self, iterator):
    self.iterator = iterator
    self.stopped = False

  def __next__(self):
    return next(self.iterator)

  def stop(self):
    self.stopped = True


class IteratorDatasetTest(unittest.TestCase):

  def test_basic(self):
    ds = datasets.IteratorDataset(iter(range(5)))
    self.assertEqual(list(ds), list(range(5)))

  def test_exhausted_raises_stop_iteration(self):
    ds = datasets.IteratorDataset(iter([]))
    with self.assertRaises(StopIteration):
      next(ds)


class MapDatasetTest(unittest.TestCase):

  def test_map(self):
    ds = datasets.IteratorDataset(iter(range(5))).map(lambda x: x * 2)
    self.assertEqual(list(ds), [0, 2, 4, 6, 8])

  def test_stop_delegates(self):
    inner = RecordingDataset(iter(range(3)))
    ds = inner.map(lambda x: x + 1)
    ds.stop()
    self.assertTrue(inner.stopped)


class MapIterTest(unittest.TestCase):

  def test_flattens(self):
    ds = datasets.IteratorDataset(iter(range(3))).map_iter(lambda x: [x, x])
    self.assertEqual(list(ds), [0, 0, 1, 1, 2, 2])

  def test_empty_iterables_are_skipped(self):
    ds = datasets.IteratorDataset(iter(range(4))).map_iter(
        lambda x: [x] if x % 2 == 0 else [])
    self.assertEqual(list(ds), [0, 2])


class FilterDatasetTest(unittest.TestCase):

  def test_filter(self):
    ds = datasets.FilterDataset(
        datasets.IteratorDataset(iter(range(10))),
        lambda x: x % 2 == 0)
    self.assertEqual(list(ds), [0, 2, 4, 6, 8])

  def test_stop_delegates(self):
    inner = RecordingDataset(iter(range(3)))
    ds = datasets.FilterDataset(inner, lambda x: True)
    ds.stop()
    self.assertTrue(inner.stopped)


class ShuffleDatasetTest(unittest.TestCase):

  def test_buffer_one_is_identity(self):
    # With a buffer of 1 the algorithm can't reorder anything.
    ds = datasets.IteratorDataset(iter(range(20))).shuffle(buffer=1)
    self.assertEqual(list(ds), list(range(20)))

  def test_preserves_multiset(self):
    for buffer in (1, 3, 10, 1000):
      ds = datasets.IteratorDataset(iter(range(50))).shuffle(buffer=buffer, seed=0)
      self.assertCountEqual(list(ds), list(range(50)))

  def test_deterministic_given_seed(self):
    def shuffled():
      ds = datasets.IteratorDataset(iter(range(50))).shuffle(buffer=5, seed=123)
      return list(ds)

    self.assertEqual(shuffled(), shuffled())

  def test_stop_delegates(self):
    inner = RecordingDataset(iter(range(3)))
    ds = datasets.ShuffleDataset(inner, buffer=2)
    list(ds)
    ds.stop()
    self.assertTrue(inner.stopped)


class ChildDatasetTest(unittest.TestCase):

  def test_split_shares_underlying_iterator(self):
    parent = datasets.IteratorDataset(iter(range(6)))
    left, right = datasets.split_dataset(parent, 2)

    # Alternately pulling from each child should walk the shared sequence.
    seen = [next(left), next(right), next(left), next(right),
            next(left), next(right)]
    self.assertEqual(seen, list(range(6)))

    with self.assertRaises(StopIteration):
      next(left)

  def test_stop_delegates(self):
    inner = RecordingDataset(iter(range(3)))
    [child] = datasets.split_dataset(inner, 1)
    child.stop()
    self.assertTrue(inner.stopped)


class InterleaveDatasetTest(unittest.TestCase):

  def test_round_robin(self):
    a = datasets.IteratorDataset(iter(['a0', 'a1', 'a2']))
    b = datasets.IteratorDataset(iter(['b0', 'b1', 'b2']))
    ds = datasets.InterleaveDataset([a, b])
    result = [next(ds) for _ in range(6)]
    self.assertEqual(result, ['a0', 'b0', 'a1', 'b1', 'a2', 'b2'])

  def test_stop_stops_all_children(self):
    a = RecordingDataset(iter(range(1)))
    b = RecordingDataset(iter(range(1)))
    ds = datasets.InterleaveDataset([a, b])
    ds.stop()
    self.assertTrue(a.stopped)
    self.assertTrue(b.stopped)


class FlattenDatasetTest(unittest.TestCase):

  def test_flattens_nested_iterables(self):
    source = datasets.IteratorDataset(iter([[1, 2], [], [3], [4, 5]]))
    ds = datasets.FlattenDataset(source)
    self.assertEqual(list(ds), [1, 2, 3, 4, 5])


class ZipDatasetTest(unittest.TestCase):

  def test_zips_elementwise(self):
    a = datasets.IteratorDataset(iter(range(3)))
    b = datasets.IteratorDataset(iter(['a', 'b', 'c']))
    ds = datasets.ZipDataset([a, b])
    self.assertEqual(list(ds), [(0, 'a'), (1, 'b'), (2, 'c')])

  def test_stop_stops_all_children(self):
    a = RecordingDataset(iter(range(1)))
    b = RecordingDataset(iter(range(1)))
    ds = datasets.ZipDataset([a, b])
    ds.stop()
    self.assertTrue(a.stopped)
    self.assertTrue(b.stopped)


class PrefetchMTTest(unittest.TestCase):

  def test_preserves_order(self):
    inner = datasets.IteratorDataset(iter(range(50)))
    ds = datasets.PrefetchMT(inner, buffer=4)
    with time_limit(10):
      result = list(ds)
    self.assertEqual(result, list(range(50)))

  def test_stop_before_exhausted_does_not_hang(self):
    inner = RecordingDataset(iter(range(10_000)))
    ds = datasets.PrefetchMT(inner, buffer=4)
    next(ds)
    next(ds)
    with time_limit(10):
      ds.stop()
    self.assertTrue(inner.stopped)


# Module-level so that MPMap's multiprocessing workers (forkserver) can
# pickle a reference to them.
def _double(x: int) -> int:
  return x * 2


def _raise(x: int) -> int:
  raise ValueError(f'bad item: {x}')


class MPMapTest(unittest.TestCase):
  """Regression tests for MPMap, including its finite-dataset drain path.

  MPMap's _iter() has two phases: the main loop (round-robins items to
  workers while the underlying dataset still has items) and a drain loop
  (retrieves the num_workers * buffer results still in flight once the
  dataset is exhausted). The drain loop previously iterated an unbounded
  itertools.cycle and hung forever on any finite dataset; these tests pin
  down correct behavior across sizes that exercise both loops.
  """

  def _check_order_preserved(self, n: int, num_workers: int, buffer: int):
    ds = datasets.IteratorDataset(iter(range(n)))
    mapped = ds.map_mp(_double, num_workers=num_workers, buffer=buffer)
    try:
      with time_limit(20):
        result = list(mapped)
    finally:
      mapped.stop()
    self.assertEqual(result, [_double(x) for x in range(n)])

  def test_dataset_exactly_fills_buffer(self):
    # n == num_workers * buffer: the main loop never runs, only the drain.
    self._check_order_preserved(n=6, num_workers=2, buffer=3)

  def test_dataset_larger_than_buffer(self):
    # Main loop runs for the leftover items, then drain empties the rest.
    self._check_order_preserved(n=37, num_workers=5, buffer=3)

  def test_single_worker_single_buffer(self):
    self._check_order_preserved(n=1, num_workers=1, buffer=1)

  def test_uneven_split(self):
    self._check_order_preserved(n=17, num_workers=4, buffer=2)

  def test_dataset_smaller_than_buffer_raises(self):
    # MPMap eagerly primes num_workers * buffer items during construction,
    # so it requires the dataset to have at least that many items upfront.
    ds = datasets.IteratorDataset(iter(range(3)))
    with self.assertRaises(StopIteration):
      ds.map_mp(_double, num_workers=2, buffer=2)

  def test_worker_error_is_reported(self):
    # An exception in map_fn should surface promptly as that same exception
    # in the consumer, rather than leaving the consumer blocked forever
    # waiting on a worker that silently died.
    ds = datasets.IteratorDataset(iter(range(2)))
    mapped = ds.map_mp(_raise, num_workers=1, buffer=1)
    try:
      with time_limit(5):
        with self.assertRaises(ValueError) as ctx:
          next(mapped)
        self.assertIn('bad item: 0', str(ctx.exception))

        # The worker catches the exception internally and keeps running, so
        # it can still be joined cleanly.
        self.assertTrue(all(w.is_alive() for w in mapped.workers))
    finally:
      mapped.stop()

  def test_stop_terminates_workers(self):
    ds = datasets.IteratorDataset(iter(range(100)))
    mapped = ds.map_mp(_double, num_workers=3, buffer=4)
    with time_limit(20):
      next(mapped)
      next(mapped)
      mapped.stop()

    for worker in mapped.workers:
      worker.join(timeout=5)
      self.assertFalse(worker.is_alive())

  def test_stop_without_consuming_does_not_hang(self):
    ds = datasets.IteratorDataset(iter(range(100)))
    mapped = ds.map_mp(_double, num_workers=4, buffer=8)
    start = time.perf_counter()
    with time_limit(20):
      mapped.stop()
    self.assertLess(time.perf_counter() - start, 10)


if __name__ == '__main__':
  unittest.main()
