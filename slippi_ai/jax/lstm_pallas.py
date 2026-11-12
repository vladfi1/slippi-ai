"""Persistent cooperative LSTM kernels for training unrolls (Pallas / Triton).

One kernel launch per layer and direction covers the whole sequence. The
grid's programs stay resident for all T frames and synchronise once per frame
with a grid-wide barrier (an atomic counter in global memory), which replaces
the per-frame chain of small kernels that makes XLA's scan latency-bound.

Forward: program (p, r) owns hidden units [pU, (p+1)U) for batch rows
[rBM, (r+1)BM). Every frame it computes its units' four gate columns,
  acc[BM, 4U] = h_t[rows, :] @ Whp[:, p*4U:(p+1)*4U],
with Wh's columns permuted so a program's four gate slices are contiguous,
adds the input projection and bias, applies the cell update, and writes its
tile of the output h_{t+1} (plus the layer's residual input, for
`lstm_recurrence_residual`), c_{t+1} and the gate activations. The state
history it keeps for the next frame and for dWh has reset rows already
zeroed. The K = H contraction is a fori_loop over BK-wide blocks so that
Triton software-pipelines the loads (num_stages > 1).

Backward: frames are walked in reverse with the same tiling. Each program
forms its tile's pre-activation gate gradients (and accumulates the bias
gradient), then after the barrier computes its tile of
dh_{t-1} = dgates_t[rows, :] @ Wh^T[:, units] (K = 4H, also pipelined).

The input projection x @ Wi is done once for the whole sequence outside the
kernels, as are the weight gradients (dWh = h^T dgates, dWi = x^T dgates)
and dx.

Requirements: every program must be resident at once (programs <= SMs; the
planner reads the SM count from the device), so don't run these under MPS
with an SM limit. Batch and hidden sizes must tile (see `supported`). Resets
zero (c, h) before the frame's update and block gradient flow, matching
`Network.step_with_reset`. Tile sizes were tuned on an RTX 4090 (128 SMs);
at B = 512, H = 1024 the forward uses 128 x (4 x 32) tiles and the backward
64 x 64 tiles, one per SM.
"""
# Shapes and tile sizes use the usual math names (T, B, H, BM, cT, ...).
# pylint: disable=invalid-name
from __future__ import annotations

import functools
import math
import os

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from slippi_ai.jax import pallas_utils
from slippi_ai.jax.pallas_utils import plgpu

F32 = jnp.float32


def _num_sms() -> int:
  env = os.environ.get('SLIPPI_LSTM_PALLAS_SMS')  # to test other tilings
  if env:
    return int(env)
  try:
    return int(getattr(jax.devices()[0], 'core_count', 0)) or 128
  except RuntimeError:
    return 128


def _sigmoid(x):
  return 1.0 / (1.0 + jnp.exp(-x))


def _barrier(counter_ref, target):
  """Grid-wide barrier: every program adds 1, then spins until `target`."""
  if pallas_utils.interpret():
    return
  plgpu.debug_barrier()
  plgpu.atomic_add(counter_ref, (0,), 1)
  lax.while_loop(lambda _: plgpu.atomic_add(counter_ref, (0,), 0) < target,
                 lambda x: x, 0)
  plgpu.debug_barrier()


def _dot(a, b):
  return pl.dot(a, b, allow_tf32=a.dtype != F32)


def _fwd_kernel(gx_ref, whp_ref, b_ref, h0_ref, c0_ref, reset_ref, res_ref, counter_ref,
                h_ref, c_ref, gates_ref, y_ref, hT_ref, counter_out_ref,
                *, T, H, U, BM, NRG, BK, has_res):
  del counter_out_ref  # aliased to counter_ref
  pid = pl.program_id(0)
  NP = pl.num_programs(0)
  p = pid // NRG
  rows = pl.ds((pid % NRG) * BM, BM)
  J = pl.ds(p * U, U)
  act = h_ref.dtype

  def gate_cols(q):
    return pl.ds(q * H + p * U, U)

  bias = [b_ref[gate_cols(q)].astype(F32)[None, :] for q in range(4)]
  # h_ref[t] holds frame t's *input* state with reset rows already zeroed, so
  # the recurrent matmul needs no mask and the backward pass can use it as is
  # for dWh. The final state goes to hT_ref, unmasked.
  h_ref[0, rows, J] = jnp.where((reset_ref[0, rows] != 0)[:, None], 0.0,
                                h0_ref[rows, J].astype(F32)).astype(act)
  c = c0_ref[rows, J].astype(F32)
  c_ref[0, rows, J] = c
  _barrier(counter_ref, NP)

  def body(t, c):
    rmask = (reset_ref[t, rows] != 0)[:, None]
    res = res_ref[t, rows, J] if has_res else None

    def kstep(kb, acc):
      hblk = h_ref[t, rows, pl.ds(kb * BK, BK)]
      return acc + _dot(hblk, whp_ref[pl.ds(kb * BK, BK), pl.ds(p * 4 * U, 4 * U)])

    acc = lax.fori_loop(0, H // BK, kstep, jnp.zeros((BM, 4 * U), F32))
    # Triton has no value slicing: pull gate q out of the [BM, 4U] tile with
    # a masked reduction over a [BM, 4, U] view.
    acc3 = acc.reshape(BM, 4, U)
    gate_idx = lax.broadcasted_iota(jnp.int32, (BM, 4, U), 1)
    z = [jnp.sum(jnp.where(gate_idx == q, acc3, 0.0), axis=1)
         + gx_ref[t, rows, gate_cols(q)].astype(F32) + bias[q] for q in range(4)]
    i, f, g, o = _sigmoid(z[0]), _sigmoid(z[1]), jnp.tanh(z[2]), _sigmoid(z[3])
    c_new = f * jnp.where(rmask, 0.0, c) + i * g
    h_new = o * jnp.tanh(c_new)
    y = h_new + res.astype(F32) if has_res else h_new
    y_ref[t, rows, J] = y.astype(y_ref.dtype)
    @pl.when(t + 1 < T)
    def _():
      next_reset = (reset_ref[t + 1, rows] != 0)[:, None]
      h_ref[t + 1, rows, J] = jnp.where(next_reset, 0.0, h_new).astype(act)

    @pl.when(t + 1 == T)
    def _():
      hT_ref[rows, J] = h_new.astype(act)

    c_ref[t + 1, rows, J] = c_new
    for q, a in enumerate((i, f, g, o)):
      gates_ref[t, rows, gate_cols(q)] = a.astype(act)
    _barrier(counter_ref, (t + 2) * NP)
    return c_new

  lax.fori_loop(0, T, body, c)


def _bwd_kernel(gates_ref, c_ref, wht_ref, reset_ref, dy_ref, dhT_ref,
                dcT_ref, counter_ref,
                dgates_ref, dh0_ref, dc0_ref, db_ref, counter_out_ref,
                *, T, H, U, BM, NRG, BK):
  del counter_out_ref
  pid = pl.program_id(0)
  NP = pl.num_programs(0)
  p = pid // NRG
  rg = pid % NRG
  rows = pl.ds(rg * BM, BM)
  J = pl.ds(p * U, U)
  act = dgates_ref.dtype

  def gate_cols(q):
    return pl.ds(q * H + p * U, U)

  def body(it, carry):
    dh, dc, db = carry
    t = T - 1 - it
    rmask = (reset_ref[t, rows] != 0)[:, None]
    dh = dh + dy_ref[t, rows, J].astype(F32)
    i, f, g, o = (gates_ref[t, rows, gate_cols(q)].astype(F32) for q in range(4))
    c_new = c_ref[t + 1, rows, J]
    c_prev = jnp.where(rmask, 0.0, c_ref[t, rows, J])
    tc = jnp.tanh(c_new)
    do = dh * tc
    dc = dc + dh * o * (1.0 - tc * tc)
    dz = (dc * g * i * (1.0 - i),
          dc * c_prev * f * (1.0 - f),
          dc * i * (1.0 - g * g),
          do * o * (1.0 - o))
    for q in range(4):
      dgates_ref[t, rows, gate_cols(q)] = dz[q].astype(act)
    # The bias gradient is the sum of dgates over time and rows: accumulate this
    # tile's column sums here instead of re-reading dgates from memory after.
    db = tuple(db[q] + jnp.sum(dz[q], axis=0, keepdims=True) for q in range(4))
    new_dc = jnp.where(rmask, 0.0, dc * f)
    _barrier(counter_ref, (it + 1) * NP)

    def kstep(kb, acc):
      dg = dgates_ref[t, rows, pl.ds(kb * BK, BK)]
      return acc + _dot(dg, wht_ref[pl.ds(kb * BK, BK), J])  # Wh^T block [BK, U]

    acc = lax.fori_loop(0, (4 * H) // BK, kstep, jnp.zeros((BM, U), F32))
    return jnp.where(rmask, 0.0, acc), new_dc, db

  carry = (dhT_ref[rows, J].astype(F32), dcT_ref[rows, J].astype(F32),
           tuple(jnp.zeros((1, U), F32) for _ in range(4)))
  dh, dc, db = lax.fori_loop(0, T, body, carry)
  dh0_ref[rows, J] = dh
  dc0_ref[rows, J] = dc
  for q in range(4):  # one partial row per row group, summed outside
    db_ref[pl.ds(rg, 1), gate_cols(q)] = db[q]


def _choose_tile(B, H, us, bms, target_area):
  """(U, BM) using the most programs that fit on the SMs, then the tile
  closest to target_area; earlier entries of us/bms win ties."""
  sms = _num_sms()
  best, best_key = None, None
  for u in us:
    for bm in bms:
      if H % u or B % bm:
        continue
      n = (H // u) * (B // bm)
      if n > sms:
        continue
      key = (n, -abs(math.log2(bm * u / target_area)))
      if best_key is None or key > best_key:
        best, best_key = (u, bm), key
  return best


def _fwd_plan(B, H, itemsize):
  """(U, BM, BK, num_warps, num_stages) or None if the shapes don't tile."""
  if pallas_utils.interpret():  # sequential programs: the barrier needs a single one
    return H, B, H, 4, 1
  if H % 32:
    return None
  tile = _choose_tile(B, H, us=(32, 64, 16), bms=(128, 64, 256, 32, 16), target_area=128 * 32)
  if tile is None:
    return None
  U, BM = tile
  big = BM * 4 * U >= 128 * 64
  # fp32 tiles are twice the shared memory per stage.
  return U, BM, 32, 8 if big else 4, 3 if itemsize <= 2 else 2


def _bwd_plan(B, H, itemsize):
  if pallas_utils.interpret():
    return H, B, 4 * H, 4, 1
  if H % 16:
    return None
  tile = _choose_tile(B, H, us=(64, 32, 128, 16), bms=(64, 128, 32, 16), target_area=64 * 64)
  if tile is None:
    return None
  U, BM = tile
  big = BM * U >= 64 * 64
  return U, BM, 64 if itemsize <= 2 else 32, 8 if big else 4, 3 if itemsize <= 2 else 2


def supported(B, H):
  return _fwd_plan(B, H, 2) is not None and _bwd_plan(B, H, 2) is not None


def _call(kernel, args, out_shapes, NP, num_warps, num_stages, **kw):
  mat = jax.typeof(args[0]).mat
  out_shapes = tuple(jax.ShapeDtypeStruct(o.shape, o.dtype, manual_axis_type=mat)
                     for o in out_shapes)
  n_in = len(args)
  counter = jnp.zeros((1,), jnp.int32)
  return pl.pallas_call(
      functools.partial(kernel, **kw),
      out_shape=out_shapes + (jax.ShapeDtypeStruct((1,), jnp.int32, manual_axis_type=mat),),
      grid=(NP,),
      input_output_aliases={n_in: len(out_shapes)},
      interpret=pallas_utils.interpret(),
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
  )(*args, counter)[:-1]


@jax.custom_vjp
def lstm_recurrence(gx, wh, b, h0, c0, reset_i):
  """gx: [T, B, 4H] input projection without bias (gate order i, f, g, o),
  wh: [H, 4H], b: [4H], h0/c0: [B, H], reset_i: [T, B] int32.
  Returns (h [T, B, H], (c_T, h_T))."""
  out, _ = _fwd(gx, wh, b, h0, c0, reset_i)
  return out


@jax.custom_vjp
def lstm_recurrence_residual(gx, wh, b, h0, c0, reset_i, residual):
  """As lstm_recurrence, but returns (residual + h, (c_T, h_T)) with the sum
  written by the kernel. residual: [T, B, H] in gx's dtype."""
  out, _ = _fwd_res(gx, wh, b, h0, c0, reset_i, residual)
  return out


def _fwd_impl(gx, wh, b, h0, c0, reset_i, residual):
  T, B, G = gx.shape
  H = G // 4
  act = gx.dtype
  U, BM, BK, nw, ns = _fwd_plan(B, H, jnp.dtype(act).itemsize)
  P, NRG = H // U, B // BM
  sd = jax.ShapeDtypeStruct
  # Permute Wh columns so each program's 4 gate slices are one contiguous block.
  whp = wh.astype(act).reshape(H, 4, P, U).transpose(0, 2, 1, 3).reshape(H, 4 * H)
  whp = pallas_utils.match_vma(whp, gx)
  has_res = residual is not None
  res_arg = residual if has_res else jnp.zeros((1,), act)
  h_all, c_all, gates, y, hT = _call(
      _fwd_kernel, (gx, whp, pallas_utils.match_vma(b, gx), h0, c0, reset_i, pallas_utils.match_vma(res_arg, gx)),
      (sd((T, B, H), act), sd((T + 1, B, H), F32), sd((T, B, G), act), sd((T, B, H), act),
       sd((B, H), act)),
      P * NRG, nw, ns, T=T, H=H, U=U, BM=BM, NRG=NRG, BK=BK, has_res=has_res)
  cT = c_all[T].astype(c0.dtype)
  hT = hT.astype(h0.dtype)
  return (y, (cT, hT)), (gates, h_all, c_all, wh, b, reset_i, h0, c0)


def _fwd(gx, wh, b, h0, c0, reset_i):
  return _fwd_impl(gx, wh, b, h0, c0, reset_i, None)


def _fwd_res(gx, wh, b, h0, c0, reset_i, residual):
  return _fwd_impl(gx, wh, b, h0, c0, reset_i, residual)


def _bwd(res, cts):
  gates, h_all, c_all, wh, b, reset_i, h0, c0 = res
  dy, (dcT, dhT) = cts
  T, B, G = gates.shape
  H = G // 4
  act = gates.dtype
  U, BM, BK, nw, ns = _bwd_plan(B, H, jnp.dtype(act).itemsize)
  P, NRG = H // U, B // BM
  sd = jax.ShapeDtypeStruct
  wht = pallas_utils.match_vma(wh.astype(act).T, gates)  # [4H, H]
  dgates, dh0, dc0, db_parts = _call(
      _bwd_kernel,
      (gates, c_all, wht, reset_i, dy.astype(act),
       dhT.astype(F32), dcT.astype(F32)),
      (sd((T, B, G), act), sd((B, H), F32), sd((B, H), F32), sd((NRG, G), F32)),
      P * NRG, nw, ns, T=T, H=H, U=U, BM=BM, NRG=NRG, BK=BK)
  precision = lax.Precision.HIGHEST if act == F32 else None
  # h_all[t] is frame t's input state with reset rows already zeroed.
  dwh = jnp.einsum('tbh,tbg->hg', h_all, dgates, precision=precision,
                   preferred_element_type=F32).astype(wh.dtype)
  # Under shard_map the weight is replicated while h/dgates vary across the
  # data axis: the cotangent of the implicit broadcast is a psum (what JAX's
  # own transpose rule would insert), and it also gives dwh the weight's type.
  db = jnp.sum(db_parts, axis=0).astype(b.dtype)
  varying = tuple(sorted(jax.typeof(dwh).mat.varying))
  if varying:
    dwh = jax.lax.psum(dwh, varying)
    db = jax.lax.psum(db, varying)
  return dgates, dwh, db, dh0.astype(h0.dtype), dc0.astype(c0.dtype), None


def _bwd_res(res, cts):
  dy = cts[0]
  return (*_bwd(res, cts), dy.astype(res[0].dtype))


lstm_recurrence.defvjp(_fwd, _bwd)
lstm_recurrence_residual.defvjp(_fwd_res, _bwd_res)
