import collections
import dataclasses
import gc
import logging
import platform
import queue
import random
import subprocess
import time
import typing as tp

import tree

import numpy as np

T = tp.TypeVar('T')

def field(default_factory: tp.Callable[[], T]) -> T:
  return dataclasses.field(default_factory=default_factory)

def stack(*vals):
  return np.stack(vals)

def concat(*vals):
  return np.concatenate(vals)

def batch_nest(nests: tp.Sequence[T]) -> T:
  return tree.map_structure(stack, *nests)


class Profiler:
  def __init__(self, burnin: int = 1):
    self.cumtime = 0
    self.num_calls = 0
    self.burnin = burnin
    self.needs_reset = False

  def __enter__(self):
    if self.needs_reset:
      self.cumtime = 0
      self.num_calls = 0
      self.needs_reset = False

    self._enter_time = time.perf_counter()

  def __exit__(self, type, value, traceback):
    self.num_calls += 1
    self.cumtime += time.perf_counter() - self._enter_time

    if self.burnin > 0:
      self.burnin -= 1

      if self.burnin == 0:
        self.needs_reset = True

  def mean_time(self):
    return self.cumtime / self.num_calls

T = tp.TypeVar('T')

class Periodically:
  def __init__(self, f: tp.Callable[..., T], interval: float):
    self.f = f
    self.interval = interval
    self.last_call = None

  def __call__(self, *args, **kwargs) -> tp.Optional[T]:
    now = time.time()
    if self.last_call is None or now - self.last_call > self.interval:
      self.last_call = now
      return self.f(*args, **kwargs)

def periodically(interval: int):
  """Decorator for running a function periodically."""
  def wrap(f):
    return Periodically(f, interval)
  return wrap

class Tracker(tp.Generic[T]):

  def __init__(self, initial: T):
    self.last = initial

  def update(self, latest: T) -> T:
    delta = latest - self.last
    self.last = latest
    return delta

class EMA:
  """Exponential moving average."""

  def __init__(self, window: float):
    self.decay = 1. / window
    self.value = None

  def update(self, value):
    if self.value is None:
      self.value = value
    else:
      self.value += self.decay * (value - self.value)

def map_single_structure(f, nest: T) -> T:
  """Map over a single nest."""
  t = type(nest)
  if t is tuple:
    return tuple([map_single_structure(f, v) for v in nest])
  if issubclass(t, tuple):
    return t(*[map_single_structure(f, v) for v in nest])
  if issubclass(t, dict):
    return {k: map_single_structure(f, v) for k, v in nest.items()}
  if t is list:
    return [map_single_structure(f, v) for v in nest]
  # Not a nest.
  return f(nest)

def map_nt(f, *nt: T) -> T:
  """Map over nested tuples and dicts.

  More efficient than tf/tree map_structure.
  """
  t = type(nt[0])
  if t is tuple:
    return tuple([map_nt(f, *vs) for vs in zip(*nt)])
  if issubclass(t, tuple):
    return t(*[map_nt(f, *vs) for vs in zip(*nt)])
  if issubclass(t, dict):
    return {k: map_nt(f, *[v[k] for v in nt]) for k in nt[0].keys()}
  if t is list:
    return [map_nt(f, *vs) for vs in zip(*nt)]
  # Not a nest.
  return f(*nt)

def batch_nest_nt(nests: tp.Sequence[T]) -> T:
  # More efficient than batch_nest
  return map_nt(stack, *nests)

def concat_nest_nt(nests: tp.Sequence[T], axis: int = 0) -> T:
  # More efficient than batch_nest
  return map_nt(lambda *xs: np.concatenate(xs, axis), *nests)

def unbatch_nest(nest: T) -> list[T]:
  return [
      tree.unflatten_as(nest, flat_slice)
      for flat_slice in zip(*tree.flatten(nest))
  ]

def reify_tuple_type(t: type[T]) -> T:
  """Takes a tuple type and returns a structure with types at the leaves."""
  # TODO: support typing.Tuple

  if issubclass(t, tuple):  # namedtuple
    return t(*[
        reify_tuple_type(t.__annotations__[name])
        for name in t._fields])

  # A leaf type
  return t

def peek_deque(d: collections.deque, n: int) -> list:
  """Peek at the last n elements of a deque."""
  assert len(d) >= n
  items = [d.pop() for _ in range(n)]
  d.extend(reversed(items))
  return items

class PeekableQueue(tp.Generic[T]):

  def __init__(self):
    self.queue = queue.Queue()
    self._peeked = collections.deque()

  def put(self, item: T):
    self.queue.put(item)

  def get(self) -> T:
    if self._peeked:
      return self._peeked.pop()
    return self.queue.get()

  def peek_n(self, n: int) -> list[T]:
    while len(self._peeked) < n:
      self._peeked.appendleft(self.queue.get())
    return peek_deque(self._peeked, n)

  def peek(self) -> T:
    return self.peek_n(1)[0]

  def qsize(self) -> int:
    return self.queue.qsize() + len(self._peeked)

  def empty(self) -> bool:
    return self.qsize() == 0

E = tp.TypeVar('E', bound=Exception)

def retry(
    f: tp.Callable[[], T],
    on_exception: tp.Mapping[type, tp.Callable[[], tp.Any]],
    num_retries: int = 4,
) -> T:
  for _ in range(num_retries):
    try:
      return f()
    except tuple(on_exception) as e:
      logging.warning(f'Caught "{repr(e)}". Retrying...')
      on_exception[type(e)]()

  # Let any exception pass through on the last attempt.
  return f()

def is_structure(obj_or_type) -> bool:
  check_fn = issubclass if isinstance(obj_or_type, type) else isinstance
  return check_fn(obj_or_type, (tp.Mapping, tp.Sequence))

def is_namedtuple(t: type) -> bool:
  return hasattr(t, '_fields')

def _check_same_structure(
    s1, s2, equal: bool = False) -> list[tuple[list, str]]:

  t1 = type(s1)
  t2 = type(s2)
  st1 = is_structure(t1)
  st2 = is_structure(t2)

  if (st1 != st2) or (st1 and st2 and (t1 != t2)):
    return [([], f'type mismatch: {t1} and {t2}')]

  errors: list[tuple[list, str]] = []

  # TODO: handle arbitrary mappings?
  if isinstance(s1, dict):
    assert isinstance(s2, dict)

    keys1 = set(s1)
    keys2 = set(s2)

    for k1 in keys1 - keys2:
      errors.append(([k1], 'only in first'))
    for k2 in keys2 - keys1:
      errors.append(([k2], 'only in second'))

    for k in keys1.union(keys2):
      sub_errors = _check_same_structure(s1[k], s2[k], equal)
      for path, _ in sub_errors:
        path.append(k)
      errors.extend(sub_errors)
    return errors

  if is_namedtuple(t1):
    assert is_namedtuple(t2)

    for k in t1._fields:
      sub_errors = _check_same_structure(
          getattr(s1, k), getattr(s2, k), equal)
      for path, _ in sub_errors:
        path.append(k)
      errors.extend(sub_errors)
    return errors

  if isinstance(s1, tp.Sequence):
    assert isinstance(s2, tp.Sequence)

    if len(s1) != len(s2):
      errors.append(([], f'different lengths: {len(s1)} != {len(s2)}'))
      return errors

    for i, (x1, x2) in enumerate(zip(s1, s2)):
      sub_errors = _check_same_structure(x1, x2, equal)
      for path, _ in sub_errors:
        path.append(i)
      errors.extend(sub_errors)
    return errors

  if equal and (s1 != s2):
    return [([], f'{s1} != {s2}')]

  return []

def check_same_structure(
    s1, s2, equal: bool = False) -> list[tuple[list, str]]:
  errors = _check_same_structure(s1, s2, equal)
  for path, _ in errors:
    path.reverse()
  return errors

def find_open_udp_ports(num: int):
  min_port = 10_000
  max_port = 2 ** 16

  system = platform.system()
  if system == 'Linux':
    netstat_command = ['ss', '-an', '--udp']
    port_delimiter = ':'
  elif system == 'Darwin':
    netstat_command = ['netstat', '-an', '-p', 'udp']
    port_delimiter = '.'
  else:
    raise NotImplementedError(f'Unsupported system "{system}"')

  netstat = subprocess.check_output(netstat_command)
  lines = netstat.decode().split('\n')[2:]

  used_ports = set()
  for line in lines:
    words = line.split()
    if not words:
      continue

    address, port = words[3].rsplit(port_delimiter, maxsplit=1)
    if port == '*':
      # TODO: what does this mean? Seems to only happen on Darwin.
      continue

    if address in ('::', 'localhost', '0.0.0.0', '*'):
      used_ports.add(int(port))

  available_ports = set(range(min_port, max_port)) - used_ports

  if len(available_ports) < num:
    raise RuntimeError('Not enough available ports.')

  return random.sample(available_ports, num)


def ref_path_exists(
    srcs: list[object],
    dsts: list[object],
) -> bool:
  todo = srcs.copy()
  dsts = set(id(obj) for obj in dsts)
  done = set()

  while todo:
    next = todo.pop()
    key = id(next)

    if key in dsts:
      return True

    if key in done:
      continue

    for obj in gc.get_referents(next):
      todo.append(obj)

    done.add(key)

  return False

def has_ref_cycle(obj: object) -> bool:
  return ref_path_exists(gc.get_referents(obj), [obj])

def gc_run(main):
  """Run main and then run garbage collection.

  This is necessary because any exceptions raised by `main` can prevent
  objects from being garbage collected.
  """
  from absl import app
  try:
    app.run(main)
  except (Exception, KeyboardInterrupt) as e:
    import traceback; traceback.print_exc()

    # release stack frames that might be preventing garbage collection
    new_exception = type(e)(*e.args)

  # For some reason we have to manually call gc.collect
  import gc; gc.collect()
  raise new_exception
