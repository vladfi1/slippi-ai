import abc
import typing

class Type(abc.ABC):

  # @abc.abstractmethod
  def check(self, value) -> bool:
    """Checks that a value is of this type."""
    return False

  @abc.abstractmethod
  def map(self, f, *args):
    """Maps a function over arguments of this type."""

  @abc.abstractmethod
  def flatten(self, obj) -> typing.Iterator:
    """Turns an object into a flat sequence of components."""

  @abc.abstractmethod
  def unflatten(self, seq: typing.Iterator):
    """Turns a flat sequence into a structured object."""

class Base(Type):

  def __init__(self, t):
    self._type = t

  def check(self, value) -> bool:
    return isinstance(value, self._type)

  def map(self, f, *args):
    return f(*args)

  def flatten(self, obj):
    yield obj

  def unflatten(self, seq):
    return next(seq)

class Dict(Type):
  """Ordered key-value mapping."""

  def __init__(self, mapping: typing.Iterable[typing.Tuple[str, Type]]):
    self._mapping = mapping
  
  def map(self, f, *args):
    get_args = lambda k: tuple(arg[k] for arg in args)
    return {k: t.map(f, *get_args(k)) for k, t in self._mapping}

  def flatten(self, struct):
    for k, t in self._mapping:
      yield from t.flatten(struct[k])

  def unflatten(self, seq):
    return {k: t.unflatten(seq) for k, t in self._mapping}

class Tuple(Type):

  def __init__(self, types: typing.Iterable[Type]):
    self._types = types

  def map(self, f, *args):
    get_args = lambda i: tuple(arg[i] for arg in args)
    return tuple(t.map(f, *get_args(i)) for i, t in enumerate(self._types))

  def flatten(self, tup):
    for i, t in enumerate(self._types):
      yield from t.flatten(tup[i])

  def unflatten(self, seq):
    return tuple(t.unflatten(seq) for t in self._types)

class Vector(Tuple):
  """Fixed-length list."""

  def __init__(self, t: Type, length: int):
    super().__init__([t] * length)
