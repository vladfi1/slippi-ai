import typing as tp

import jax
import jax.numpy as jnp

from slippi_ai import utils
from slippi_ai.types import Action

T = tp.TypeVar('T')

def bm_to_tm(nest: T) -> T:
  """Converts [B, 2, T] to [T, B, 2]."""
  return utils.map_single_structure(
      lambda x: jnp.moveaxis(x, 2, 0), nest)

def tm_to_bm(nest: T) -> T:
  """Converts [T, B, 2] to [B, 2, T]."""
  return utils.map_single_structure(
      lambda x: jnp.moveaxis(x, 0, 2), nest)


def compute_unique_fraction(actions: list[Action]) -> jax.Array:
  # Compute fraction of actions that are unique

  # We assume that the action components are scalars
  stacked_actions = utils.map_nt(  # [S, T, FS, B, 2]
      lambda *xs: jnp.stack(xs, axis=2), *actions)
  combined_actions = jnp.stack(
      jax.tree.leaves(stacked_actions), axis=-1)  # [S, T, FS, B, 2, C]
  num_samples = combined_actions.shape[0]

  actions_eq = combined_actions == jnp.expand_dims(combined_actions, axis=1)  # [S, S, T, FS, B, 2, C]
  actions_eq = jnp.all(actions_eq, axis=[2, -1])  # [S, S, T, B, 2]

  ns = jnp.arange(num_samples)
  # is_first[i, j] = i < j for i, j in [0, S)
  is_first = jnp.expand_dims(ns, 1) < ns  # [S, S]
  is_first = jnp.expand_dims(is_first, axis=[2, 3, 4])  # [S, S, 1, 1, 1]
  # i is disqualified by j if i < j and action[i] == action[j]
  disqualified = jnp.logical_and(actions_eq, is_first)  # [S, S, T, B, 2]
  is_unique = ~jnp.any(disqualified, axis=1)  # [S, T, B, 2]
  unique_fraction = jnp.mean(is_unique, axis=0)  # [T, B, 2]

  return unique_fraction

def information_fraction(
  payoff_matrices: jax.Array,  # [..., S, S]
  eps: float = 1e-8,
) -> jax.Array:
  """Fraction of a payoff matrix that can't be explained by additive interactions."""
  P = payoff_matrices
  m = P.mean(axis=[-2, -1], keepdims=True)
  P_m = P - m
  r = P_m.mean(axis=-1, keepdims=True)
  c = P_m.mean(axis=-2, keepdims=True)
  I = P_m - r - c

  return I.var(axis=[-2, -1]) / (P_m.var(axis=[-2, -1]) + eps)
