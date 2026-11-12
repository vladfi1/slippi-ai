"""The enhanced embed's item MLP, run only on item slots that hold an item.

EnhancedEmbedModule embeds each of the 15 item slots of every frame, runs the
item MLP on all of them, zeroes the empty slots and sums over slots. In
training data most slots are empty (under one item per frame on average), so
`mlp_sum` gathers the occupied (frame, slot) pairs into a buffer of
CAPACITY_PER_FRAME rows per frame, runs the MLP on those rows only and sums
them back into their frames. If a batch holds more items than the buffer, it
runs the full computation instead, so the result is always the same function,
up to summation order.

The backward pass recomputes the MLP from the (small) item inputs instead of
saving activations, so the two branches don't both keep [frames x slots]
residuals. Used for unrolls only (training); `step` is untouched.
SLIPPI_ITEM_COMPACT=0 disables it.
"""
# Shapes use the usual math names (M, J, C).
# pylint: disable=invalid-name

import functools
import math
import os
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx

from slippi_ai.jax import embed as embed_lib
from slippi_ai.jax import pallas_utils

Array = jax.Array

# Buffer rows per frame. Training batches average 0.7-0.8 items per frame.
CAPACITY_PER_FRAME = 2


def enabled() -> bool:
  """Whether `mlp_sum` replaces the full item MLP. Read at trace time."""
  return os.environ.get('SLIPPI_ITEM_COMPACT', '1') != '0'


def _full(mlp: nnx.Module, embedding: embed_lib.Embedding, items: tp.Any,
          exists: Array, dtype) -> Array:
  """The original computation: every slot through the MLP, empty ones zeroed."""
  h = mlp(embedding(items).astype(dtype))
  h = jnp.where(exists[..., None], h, 0)
  return jnp.sum(h, axis=-2)


def _compact(mlp: nnx.Module, embedding: embed_lib.Embedding, items: tp.Any,
             exists: Array, dtype, C: int) -> Array:
  """The same sum over at most C occupied slots."""
  lead, J = exists.shape[:-1], exists.shape[-1]
  M = math.prod(lead)
  flat_exists = exists.reshape(M * J)
  # Flat indices (frame * J + slot) of occupied slots in ascending order,
  # padded with M * J.
  (idx,) = jnp.nonzero(flat_exists, size=C, fill_value=M * J)
  valid = idx < M * J
  safe = jnp.where(valid, idx, 0)
  selected = jax.tree.map(lambda x: x.reshape(M * J)[safe], items)
  h = mlp(embedding(selected).astype(dtype))
  h = jnp.where(valid[:, None], h, 0)
  # Padding goes to segment M, which segment_sum drops.
  frame = jnp.where(valid, idx // J, M)
  out = jax.ops.segment_sum(h.astype(jnp.float32), frame, num_segments=M,
                            indices_are_sorted=True)
  return out.astype(h.dtype).reshape(*lead, -1)


def _apply(graphdef, embedding, dtype, C, state, items, exists):
  mlp = nnx.merge(graphdef, state)
  count = jnp.sum(exists, dtype=jnp.int32)
  return jax.lax.cond(
      count <= C,
      lambda: _compact(mlp, embedding, items, exists, dtype, C),
      lambda: _full(mlp, embedding, items, exists, dtype))


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _mlp_sum(graphdef, embedding, dtype, C, state, items, exists):
  return _apply(graphdef, embedding, dtype, C, state, items, exists)


def _mlp_sum_fwd(graphdef, embedding, dtype, C, state, items, exists):
  out = _apply(graphdef, embedding, dtype, C, state, items, exists)
  return out, (state, items, exists)


def _mlp_sum_bwd(graphdef, embedding, dtype, C, res, g):
  state, items, exists = res
  # Under shard_map the parameters are replicated and the items vary over the
  # data axes. Differentiate a varying copy so the branches below hold no
  # collectives (their choice can differ between devices), then psum.
  varying = tuple(sorted(jax.typeof(g).mat.varying))
  state_v = jax.tree.map(lambda p: pallas_utils.match_vma(p, g), state)
  count = jnp.sum(exists, dtype=jnp.int32)

  def grads(fn):
    def branch():
      mlp_out, vjp = jax.vjp(lambda s: fn(nnx.merge(graphdef, s)), state_v)
      del mlp_out
      return vjp(g)[0]
    return branch

  dstate = jax.lax.cond(
      count <= C,
      grads(lambda mlp: _compact(mlp, embedding, items, exists, dtype, C)),
      grads(lambda mlp: _full(mlp, embedding, items, exists, dtype)))
  if varying:
    dstate = jax.tree.map(lambda d: jax.lax.psum(d, varying), dstate)
  # The items are data, not parameters.
  return dstate, None, None


_mlp_sum.defvjp(_mlp_sum_fwd, _mlp_sum_bwd)


def mlp_sum(mlp: nnx.Module, embedding: embed_lib.Embedding, items: tp.Any,
            exists: Array, dtype, capacity_per_frame: int = CAPACITY_PER_FRAME) -> Array:
  """sum over slots of where(exists, mlp(embedding(items)), 0).

  items: the stacked item fields, each [..., J]; exists: [..., J] bool.
  Returns [..., F] in `dtype` (the MLP's output dtype).
  """
  lead, J = exists.shape[:-1], exists.shape[-1]
  M = math.prod(lead)
  C = min(M * J, capacity_per_frame * M)
  graphdef, state = nnx.split(mlp)
  return _mlp_sum(graphdef, embedding, dtype, C, state, items, exists)
