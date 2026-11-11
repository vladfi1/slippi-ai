"""The simple embed and the network's first Linear, with one-hot fields looked up.

The simple embed concatenates every leaf embedding of the state-action into one
wide vector (6509 columns at the default config, 5617 of them from 44 one-hot
fields), and tx_like's first layer multiplies it by a [6509, H] kernel.
Multiplying a one-hot by the kernel selects one kernel row, so the forward pass
here is a Pallas kernel that sums the selected rows and multiplies only the few
dense columns (floats and bools) on the tensor cores. The weight gradient still
needs the full one-hot input, which the backward pass rebuilds from the saved
indices and multiplies in XLA as before. Same parameters and the same function
as concatenate-then-Linear, up to summation order.

Used when `supported` holds (GPU or Pallas interpret mode, bf16, the hidden
size a multiple of 128); `SLIPPI_EMBED_LOOKUP=0` disables it.
"""
# Shapes use the usual math names (M, F, H).
# pylint: disable=invalid-name

import dataclasses
import functools
import os
import typing as tp

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from slippi_ai.jax import embed as embed_lib
from slippi_ai.jax import pallas_utils

Array = jax.Array

# Output tiles: BM rows x BN columns, all one-hot leaves summed in f32 registers.
BM = 64
BN = 128


def supported(H: int, dtype) -> bool:
  """Whether `linear` runs for a [*, H] output in `dtype`. Read at trace time."""
  if os.environ.get('SLIPPI_EMBED_LOOKUP', '1') == '0':
    return False
  if jax.default_backend() != 'gpu' and not pallas_utils.interpret():
    return False
  return dtype == jnp.bfloat16 and H % BN == 0


@dataclasses.dataclass(frozen=True)
class Layout:
  """Static description of the embedded columns, in concatenation order.

  segments: (kind, part, offset, size) per leaf: kind 'onehot' with part the
    index into the stacked indices, or 'dense' with part the first column in
    the stacked dense values; offset is the leaf's first kernel row.
  """
  size: int
  segments: tuple[tuple[str, int, int, int], ...]

  @functools.cached_property
  def onehot_offsets(self) -> tuple[int, ...]:
    return tuple(s[2] for s in self.segments if s[0] == 'onehot')

  @functools.cached_property
  def dense_rows(self) -> tuple[int, ...]:
    rows = []
    for kind, _, offset, size in self.segments:
      if kind == 'dense':
        rows.extend(range(offset, offset + size))
    return tuple(rows)


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=['dense', 'idx'], meta_fields=['layout'])
@dataclasses.dataclass
class SparseEmbedded:
  """An embedded state-action before concatenation.

  dense: [..., D] values of the non-one-hot leaves, in column order.
  idx: [..., F] int32 class of each one-hot leaf, -1 where the one-hot is all
    zeros (out-of-range input).
  """
  dense: Array
  idx: Array
  layout: Layout

  def astype(self, dtype) -> 'SparseEmbedded':
    return SparseEmbedded(self.dense.astype(dtype), self.idx, self.layout)

  def to_dense(self) -> Array:
    """The concatenated embedding, as the simple embed would produce it."""
    parts = []
    for kind, part, _, size in self.layout.segments:
      if kind == 'onehot':
        parts.append(jax.nn.one_hot(self.idx[..., part], size, dtype=self.dense.dtype))
      else:
        parts.append(self.dense[..., part:part + size])
    return jnp.concatenate(parts, axis=-1)


def _is_plain_one_hot(e: embed_lib.Embedding) -> bool:
  return (isinstance(e, embed_lib.OneHotEmbedding)
          and type(e).__call__ is embed_lib.OneHotEmbedding.__call__)


def embed(embedding: embed_lib.Embedding, struct: tp.Any) -> SparseEmbedded:
  """Like `embedding(struct)`, but keeps one-hot leaves as indices.

  Mirrors StructEmbedding/ListEmbedding/CompoundEmbedding concatenation; any
  other leaf is evaluated as usual and kept as dense columns.
  """
  dense: list[Array] = []
  idx: list[Array] = []
  segments: list[tuple[str, int, int, int]] = []
  dense_cols = 0
  offset = 0

  def visit(e: embed_lib.Embedding, x: tp.Any):
    nonlocal dense_cols, offset
    if isinstance(e, embed_lib.StructEmbedding):
      for field, op in e.embedding:
        if op.size == 0:
          continue
        visit(op, e.getter(x, field))
    elif isinstance(e, embed_lib.ListEmbedding):
      if len(x) != e.num_items:
        raise ValueError(f'Expected {e.num_items} items, got {len(x)}')
      for item in x:
        visit(e.item_embedding, item)
    elif isinstance(e, embed_lib.CompoundEmbedding):
      visit(e._embed_mid, x)  # pylint: disable=protected-access
    elif _is_plain_one_hot(e):
      t = jnp.asarray(x)
      # jax.nn.one_hot gives an all-zero row for classes outside [0, size).
      valid = (t >= 0) & (t < e.size)
      idx.append(jnp.where(valid, t.astype(jnp.int32), -1))
      segments.append(('onehot', len(idx) - 1, offset, e.size))
      offset += e.size
    else:
      value = e(x)
      if value.shape[-1] != e.size:
        raise ValueError(f'{type(e).__name__} gave width {value.shape[-1]} != {e.size}')
      dense.append(value.astype(jnp.float32))
      segments.append(('dense', dense_cols, offset, e.size))
      dense_cols += e.size
      offset += e.size

  visit(embedding, struct)
  if offset != embedding.size:
    raise ValueError(f'embedded width {offset} != embedding size {embedding.size}')
  if not idx:
    raise ValueError('no one-hot leaves to look up')
  idx_arr = jnp.stack(idx, axis=-1)
  lead = idx_arr.shape[:-1]
  dense_arr = (jnp.concatenate(dense, axis=-1) if dense
               else jnp.zeros(lead + (0,), jnp.float32))
  return SparseEmbedded(dense_arr, idx_arr, Layout(offset, tuple(segments)))


def _lookup_kernel(rows_ref, w_ref, dense_ref, wd_ref, b_ref, o_ref, *, F, R):
  """o = sum_f w[rows[f]] + dense @ wd + bias, for one [BM, BN] tile."""
  m0 = pl.program_id(0) * BM
  cols = pl.ds(pl.program_id(1) * BN, BN)

  def body(f, acc):
    r = rows_ref[f, pl.ds(m0, BM)]
    valid = r < R  # row R: an all-zero one-hot or padding
    w = w_ref[jnp.where(valid, r, 0), cols]
    return acc + jnp.where(valid[:, None], w.astype(jnp.float32), 0.)

  acc = pl.dot(dense_ref[pl.ds(m0, BM), :], wd_ref[:, cols])
  acc = jax.lax.fori_loop(0, F, body, acc)
  # Round, then add the bias in the output dtype, as nnx.Linear does.
  y = acc.astype(o_ref.dtype).astype(jnp.float32) + b_ref[cols].astype(jnp.float32)
  o_ref[pl.ds(m0, BM), cols] = y.astype(o_ref.dtype)


def _forward(layout: Layout, kernel: Array, bias: Array, dense: Array, idx: Array) -> Array:
  lead = idx.shape[:-1]
  F = idx.shape[-1]
  R, H = kernel.shape
  dtype = kernel.dtype
  M = 1
  for d in lead:
    M *= d
  Mp = -(-M // BM) * BM
  # Kernel row of each one-hot leaf, R where the one-hot is all zeros, laid
  # out [F, M] so each leaf is one contiguous load per tile.
  offsets = jnp.asarray(layout.onehot_offsets, jnp.int32)
  rows = jnp.where(idx >= 0, idx + offsets, R).reshape(M, F).T
  rows = jnp.pad(rows, ((0, 0), (0, Mp - M)), constant_values=R)
  # The dense columns, zero-padded to a power of two (at least 16) for pl.dot.
  D = len(layout.dense_rows)
  Dp = max(16, 1 << max(D - 1, 0).bit_length())
  dense_p = jnp.pad(dense.reshape(M, D).astype(dtype), ((0, Mp - M), (0, Dp - D)))
  w_dense = kernel[jnp.asarray(layout.dense_rows, jnp.int32)] if D else kernel[:0]
  w_dense = jnp.pad(w_dense, ((0, Dp - D), (0, 0)))
  match = lambda x: pallas_utils.match_vma(x, rows)
  out = pl.pallas_call(
      functools.partial(_lookup_kernel, F=F, R=R),
      out_shape=jax.ShapeDtypeStruct((Mp, H), dtype, manual_axis_type=jax.typeof(rows).mat),
      grid=(Mp // BM, H // BN),
      compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
      interpret=pallas_utils.interpret(),
      name='embed_lookup',
  )(rows, match(kernel), dense_p, match(w_dense), match(bias.astype(dtype)))
  return out[:M].reshape(*lead, H)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _linear(layout, kernel, bias, dense, idx):
  return _forward(layout, kernel, bias, dense, idx)


def _linear_fwd(layout, kernel, bias, dense, idx):
  return _forward(layout, kernel, bias, dense, idx), (dense.astype(kernel.dtype), idx)


def _linear_bwd(layout, res, dy):
  dense, idx = res
  x = SparseEmbedded(dense, idx, layout).to_dense()
  lead = tuple(range(dy.ndim - 1))
  dkernel = jax.lax.dot_general(
      x, dy, ((lead, lead), ((), ())),
      preferred_element_type=jnp.float32).astype(dense.dtype)
  dbias = jnp.sum(dy.astype(jnp.float32), axis=lead).astype(dy.dtype)
  # Under shard_map the weights are replicated while the activations vary
  # over the data axis: the cotangent of that implicit broadcast is a psum.
  varying = tuple(sorted(jax.typeof(dkernel).mat.varying))
  if varying:
    dkernel = jax.lax.psum(dkernel, varying)
    dbias = jax.lax.psum(dbias, varying)
  # The embedded inputs are data (indices and features), not parameters.
  return dkernel, dbias, None, None


_linear.defvjp(_linear_fwd, _linear_bwd)


def linear(x: SparseEmbedded, kernel: Array, bias: tp.Optional[Array]) -> Array:
  """`x.to_dense() @ kernel + bias`, gathering kernel rows for one-hot leaves.

  For bf16 kernels with `supported(H, dtype)`; x's dense values are cast to the
  kernel's dtype.
  """
  if kernel.shape[0] != x.layout.size:
    raise ValueError(f'kernel has {kernel.shape[0]} rows, embedding {x.layout.size}')
  if bias is None:
    bias = jnp.zeros(kernel.shape[1], kernel.dtype)
  return _linear(x.layout, kernel, bias, x.dense, x.idx)
