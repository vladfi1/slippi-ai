import typing as tp

import jax.numpy as jnp

from slippi_ai import utils

T = tp.TypeVar('T')

def bm_to_tm(nest: T) -> T:
  """Converts [B, 2, T] to [T, B, 2]."""
  return utils.map_single_structure(
      lambda x: jnp.moveaxis(x, 2, 0), nest)

def tm_to_bm(nest: T) -> T:
  """Converts [T, B, 2] to [B, 2, T]."""
  return utils.map_single_structure(
      lambda x: jnp.moveaxis(x, 0, 2), nest)
