import enum
import dataclasses
import typing as tp

from absl import flags
from absl import logging
import fancyflags as ff
import tree

T = tp.TypeVar('T')

TYPE_TO_ITEM = {
    bool: ff.Boolean,
    int: ff.Integer,
    str: ff.String,
    float: ff.Float,
    list: ff.Sequence,
}

def maybe_undo_optional(t: type) -> type:
  if (
      hasattr(t, '__origin__') and
      t.__origin__ is tp.Union and
      len(t.__args__) == 2 and
      t.__args__[1] is type(None)
  ):
    return t.__args__[0]
  return t


def get_flags_from_default(default) -> tp.Optional[tree.Structure[ff.Item]]:
  if isinstance(default, dict):
    result = {}
    for k, v in default.items():
      flag = get_flags_from_default(v)
      if flag is not None:
        result[k] = flag
    return result

  item_constructor = TYPE_TO_ITEM.get(type(default))
  if item_constructor is not None:
    return item_constructor(default)

  return None


def _get_default(field: dataclasses.Field):
  return None if field.default is dataclasses.MISSING else field.default


def get_flags_from_dataclass(cls: type) -> tree.Structure[ff.Item]:
  if not dataclasses.is_dataclass(cls):
    raise TypeError(f'{cls} is not a dataclass')
  
  result = {}

  for field in dataclasses.fields(cls):
    field_type = maybe_undo_optional(field.type)
    item_constructor = TYPE_TO_ITEM.get(field_type)
    if item_constructor is not None:
      result[field.name] = item_constructor(_get_default(field))
    elif dataclasses.is_dataclass(field_type):
      result[field.name] = get_flags_from_dataclass(field_type)
    elif issubclass(field_type, enum.Enum):
      result[field.name] = ff.EnumClass(
          default=_get_default(field),
          enum_class=field_type,
      )
    else:
      logging.warn(f'Unsupported field of type {field_type}')

  return result


def define_dict_from_dataclass(name: str, cls: type) -> flags.FlagHolder:
  return ff.DEFINE_dict(name, **get_flags_from_dataclass(cls))


def dataclass_from_dict(cls: tp.Type[T], nest: dict) -> T:
  recursed = {}

  for field in dataclasses.fields(cls):
    value = nest[field.name]
    if dataclasses.is_dataclass(field.type):
      value = dataclass_from_dict(field.type, value)
    recursed[field.name] = value
  
  return cls(**recursed)
