import enum
import dataclasses
import typing as tp

from absl import flags
from absl import logging
import fancyflags as ff
from fancyflags._definitions import MultiItem
import tree

T = tp.TypeVar('T')
Item = tp.Union[ff.Item, MultiItem]
ItemConstructor = tp.Callable[[tp.Any], Item]

TYPE_TO_ITEM: dict[type, ItemConstructor] = {
    bool: ff.Boolean,
    int: ff.Integer,
    str: ff.String,
    float: ff.Float,
    list[str]: ff.StringList,
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

def undo_list(t: type) -> tp.Optional[type]:
  if (
      hasattr(t, '__origin__') and
      t.__origin__ is list
  ):
    return t.__args__[0]
  return None

def handle_list(field_type: type, default: tp.Any) -> tp.Optional[Item]:
  base_type = undo_list(field_type)
  if base_type is None:
    return None

  if issubclass(base_type, enum.Enum):
    return ff.MultiEnumClass(
        default=default,
        enum_class=base_type,
    )

  if issubclass(base_type, (int, float, str)):
    return ff.Sequence(default=default)

  return None

def get_leaf_flag(field_type: type, default: tp.Any) -> tp.Optional[Item]:
  field_type = maybe_undo_optional(field_type)
  item_constructor = TYPE_TO_ITEM.get(field_type)
  if item_constructor is not None:
    return item_constructor(default)
  elif issubclass(field_type, enum.Enum):
    return ff.EnumClass(
        default=default,
        enum_class=field_type,
    )

  item = handle_list(field_type, default)
  if item is not None:
    return item

  # TODO: also log path to unsupported field
  logging.warn(f'Unsupported field of type {field_type}')
  return None

def is_leaf(type_: type) -> bool:
  type_ = maybe_undo_optional(type_)
  if issubclass(type_, dict) or dataclasses.is_dataclass(type_):
    return False
  return True

def get_flags_from_default(default) -> tp.Optional[tree.Structure[ff.Item]]:
  if isinstance(default, dict):
    result = {}
    for k, v in default.items():
      flag = get_flags_from_default(v)
      if flag is not None:
        result[k] = flag
    return result

  if dataclasses.is_dataclass(default):
    result = {}
    for field in dataclasses.fields(default):
      if is_leaf(field.type):
        flag = get_leaf_flag(field.type, getattr(default, field.name))
        if flag is not None:
          result[field.name] = flag
      else:
        result[field.name] = get_flags_from_default(getattr(default, field.name))
    return result

  field_type = type(default)
  return get_leaf_flag(field_type, default)


def _get_default(field: dataclasses.Field):
  if field.default_factory is not dataclasses.MISSING:
    return field.default_factory()
  if field.default is not dataclasses.MISSING:
    return field.default
  return dataclasses.MISSING


def get_flags_from_dataclass(cls: type) -> tree.Structure[ff.Item]:
  if not dataclasses.is_dataclass(cls):
    raise TypeError(f'{cls} is not a dataclass')

  result = {}

  for field in dataclasses.fields(cls):
    field_default = _get_default(field)
    if dataclasses.is_dataclass(field.type):
      if field_default is dataclasses.MISSING:
        result[field.name] = get_flags_from_dataclass(field.type)
      else:
        result[field.name] = get_flags_from_default(field_default)
    elif field.type is dict:
      if field_default is dataclasses.MISSING:
        raise ValueError(f'Field {cls.__name__}.{field.name} is a dict but has no default value')
      result[field.name] = get_flags_from_default(field_default)
    else:
      if field_default is dataclasses.MISSING:
        raise ValueError(f'Field {cls.__name__}.{field.name} has no default value')
      item = get_leaf_flag(field.type, field_default)
      if item is not None:
        result[field.name] = item

  return result


def define_dict_from_dataclass(name: str, cls: type) -> flags.FlagHolder:
  # Note: calling this has the unfortunate property that the flags are
  # associated with the flag_utils.py module, not the module that calls this,
  # which makes them not show up in help messages unless you pass --helpfull.
  return ff.DEFINE_dict(name, **get_flags_from_dataclass(cls))


def dataclass_from_dict(cls: tp.Type[T], nest: dict) -> T:
  """Recursively construct a dataclass from a nested dict."""
  recursed = {}

  for field in dataclasses.fields(cls):
    if field.name not in nest:
      if field.default is not dataclasses.MISSING:
        value = field.default
      elif field.default_factory is not dataclasses.MISSING:
        value = field.default_factory()
      else:
        raise ValueError(f'No value specified for {cls.__name__}.{field.name}')
    else:
      value = nest[field.name]
      if dataclasses.is_dataclass(field.type):
        value = dataclass_from_dict(field.type, value)

    recursed[field.name] = value

  return cls(**recursed)

def override_dict(
    base: dict,
    overrides: flags.FlagHolder[dict],
    prefix: tp.Sequence[str],
) -> dict:
  """Override a base config value from another dictionary."""

  def maybe_update(path: tp.Sequence[str], base_value):
    key = '.'.join([overrides.name, *prefix, *path])
    flag = overrides._flagvalues[key]
    if flag.using_default_value:
      return base_value

    logging.info(f'Overriding from --{key}={flag.value}')
    return flag.value

  return tree.map_structure_with_path(maybe_update, base)
