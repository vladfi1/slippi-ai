"""Residual feed-forward block, out = r + Linear2(gelu(Linear1(LayerNorm(r)))),
as Pallas Triton kernels with the elementwise work fused into the matmuls.

XLA runs each matmul here as a bare dot, so the LayerNorm, bias adds, GELU and
residual add (and their gradients, carried in f32) are separate memory-bound
passes over [M, H] and [M, 4H] tensors. Here they ride along with the matmuls
or with one light pass over the rows:

  forward   x = LayerNorm(r)                        row kernel (also mean, rstd)
            h = x @ W1 + b1, a = gelu(h)             matmul kernel, writes h and a
            out = a @ W2 + b2 + r                    matmul kernel
  backward  da = dout @ W2^T, dh = da * gelu'(h)     matmul kernel, + db1 partials
            dW1, dW2, dx = dh @ W1^T                 XLA
            dr = dout + LayerNorm'(dx)               row kernel, + dscale, dbias,
                                                     db2 partials

The activation is jax.nn.gelu(approximate=True) and the LayerNorm is
nnx.LayerNorm's (f32 statistics, fast variance clipped at zero).
"""
# Shapes and tile sizes use the usual math names (M, D, F, BM, ...).
# pylint: disable=invalid-name
from __future__ import annotations

import functools
import math
import os

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from slippi_ai.jax import pallas_utils

F32 = jnp.float32
_C = math.sqrt(2 / math.pi)
_A = 0.044715

# (BM, BN, BK, num_warps, num_stages, GROUP_M) per matmul kernel, tuned on an
# RTX 4090 at the tx_like trainer's shape (M = 84 x 512, D = 1024, F = 4096).
FWD_CONFIG = (128, 64, 32, 4, 3, 8)
BWD_CONFIG = (64, 128, 32, 4, 3, 8)
DOWN_CONFIG = (128, 128, 32, 4, 4, 8)
# Row kernels: (rows per program, rows per inner step, num_warps).
LN_FWD_CONFIG = (8, 1, 2)
LN_BWD_CONFIG = (128, 8, 8)


def _gelu(h):
  return 0.5 * h * (1.0 + jnp.tanh(_C * (h + _A * h * h * h)))


def _gelu_grad(h):
  t = jnp.tanh(_C * (h + _A * h * h * h))
  return 0.5 * (1.0 + t) + 0.5 * h * (1.0 - t * t) * _C * (1.0 + 3 * _A * h * h)


def _tile(M, N, BM, BN, GM):
  """Grouped (L2-friendly) program order: GM row blocks share each column sweep."""
  pid = pl.program_id(0)
  num_m, num_n = M // BM, N // BN
  per_group = GM * num_n
  first_m = (pid // per_group) * GM
  gsize = jnp.minimum(num_m - first_m, GM)
  local = pid % per_group
  return first_m + local % gsize, local // gsize


def _fwd_kernel(x_ref, w_ref, b_ref, h_ref, a_ref, *, M, N, K, BM, BN, BK, GM):
  pm, pn = _tile(M, N, BM, BN, GM)
  rows, cols = pl.ds(pm * BM, BM), pl.ds(pn * BN, BN)

  def kstep(k, acc):
    ks = pl.ds(k * BK, BK)
    return acc + pl.dot(x_ref[rows, ks], w_ref[ks, cols])

  acc = lax.fori_loop(0, K // BK, kstep, jnp.zeros((BM, BN), F32))
  h = acc + b_ref[cols].astype(F32)[None, :]
  h_ref[rows, cols] = h.astype(h_ref.dtype)
  a_ref[rows, cols] = _gelu(h).astype(a_ref.dtype)


def _bwd_kernel(dy_ref, w_ref, h_ref, dh_ref, dbp_ref, *, M, N, K, BM, BN, BK, GM):
  """da = dy @ W2^T with W2 stored [N, K]; dh = da * gelu'(h)."""
  pm, pn = _tile(M, N, BM, BN, GM)
  rows, cols = pl.ds(pm * BM, BM), pl.ds(pn * BN, BN)

  def kstep(k, acc):
    ks = pl.ds(k * BK, BK)
    return acc + pl.dot(dy_ref[rows, ks], w_ref[cols, ks], trans_b=True)

  da = lax.fori_loop(0, K // BK, kstep, jnp.zeros((BM, BN), F32))
  dh = da * _gelu_grad(h_ref[rows, cols].astype(F32))
  dh_ref[rows, cols] = dh.astype(dh_ref.dtype)
  dbp_ref[pl.ds(pm, 1), cols] = jnp.sum(dh, axis=0, keepdims=True)


def _down_kernel(a_ref, w_ref, b_ref, r_ref, out_ref, *, M, N, K, BM, BN, BK, GM):
  """out = a @ W2 + b2 + r."""
  pm, pn = _tile(M, N, BM, BN, GM)
  rows, cols = pl.ds(pm * BM, BM), pl.ds(pn * BN, BN)

  def kstep(k, acc):
    ks = pl.ds(k * BK, BK)
    return acc + pl.dot(a_ref[rows, ks], w_ref[ks, cols])

  acc = lax.fori_loop(0, K // BK, kstep, jnp.zeros((BM, BN), F32))
  out = acc + b_ref[cols].astype(F32)[None, :] + r_ref[rows, cols].astype(F32)
  out_ref[rows, cols] = out.astype(out_ref.dtype)


def _ln_fwd_kernel(r_ref, g_ref, b_ref, x_ref, mean_ref, rstd_ref, *, D, R, CH, eps):
  """nnx.LayerNorm over the last axis, R rows per program in CH-row steps."""
  p = pl.program_id(0)
  gamma = g_ref[:].astype(F32)[None, :]
  beta = b_ref[:].astype(F32)[None, :]

  def body(i, carry):
    rows = pl.ds(p * R + i * CH, CH)
    r = r_ref[rows, :].astype(F32)
    mean = jnp.sum(r, axis=1) / D
    var = jnp.maximum(jnp.sum(r * r, axis=1) / D - mean * mean, 0.0)
    rstd = lax.rsqrt(var + eps)
    x = (r - mean[:, None]) * (rstd[:, None] * gamma) + beta
    x_ref[rows, :] = x.astype(x_ref.dtype)
    mean_ref[rows] = mean
    rstd_ref[rows] = rstd
    return carry

  lax.fori_loop(0, R // CH, body, 0)


def _ln_bwd_kernel(r_ref, dx_ref, dout_ref, g_ref, mean_ref, rstd_ref,
                   dr_ref, dg_ref, dbeta_ref, db2_ref, *, D, R, CH):
  """dr = dout + LayerNorm backward of dx; per-program column sums of
  dx * xhat (dscale), dx (dbias) and dout (the output bias b2)."""
  p = pl.program_id(0)
  gamma = g_ref[:].astype(F32)[None, :]

  def body(i, accs):
    rows = pl.ds(p * R + i * CH, CH)
    rstd = rstd_ref[rows][:, None]
    xhat = (r_ref[rows, :].astype(F32) - mean_ref[rows][:, None]) * rstd
    dx = dx_ref[rows, :].astype(F32)
    dout = dout_ref[rows, :].astype(F32)
    dxhat = dx * gamma
    m1 = jnp.sum(dxhat, axis=1, keepdims=True) / D
    m2 = jnp.sum(dxhat * xhat, axis=1, keepdims=True) / D
    dr_ref[rows, :] = (dout + rstd * (dxhat - m1 - xhat * m2)).astype(dr_ref.dtype)
    ag, ab, ad = accs
    return (ag + jnp.sum(dx * xhat, axis=0, keepdims=True),
            ab + jnp.sum(dx, axis=0, keepdims=True),
            ad + jnp.sum(dout, axis=0, keepdims=True))

  z = jnp.zeros((1, D), F32)
  ag, ab, ad = lax.fori_loop(0, R // CH, body, (z, z, z))
  dg_ref[pl.ds(p, 1), :] = ag
  dbeta_ref[pl.ds(p, 1), :] = ab
  db2_ref[pl.ds(p, 1), :] = ad


def _rows_call(kernel, args, out_shapes, M, config, **kw):
  R, CH, nw = config
  mat = jax.typeof(args[0]).mat
  out_shapes = tuple(jax.ShapeDtypeStruct(s, d, manual_axis_type=mat) for s, d in out_shapes)
  return pl.pallas_call(
      functools.partial(kernel, R=R, CH=CH, **kw),
      out_shape=out_shapes,
      grid=(M // R,),
      interpret=pallas_utils.interpret(),
      compiler_params=plgpu.CompilerParams(num_warps=nw, num_stages=1),
  )(*args)


def _call(kernel, args, out_shapes, M, N, K, config):
  BM, BN, BK, nw, ns, GM = config
  mat = jax.typeof(args[0]).mat
  out_shapes = tuple(jax.ShapeDtypeStruct(s, d, manual_axis_type=mat) for s, d in out_shapes)
  return pl.pallas_call(
      functools.partial(kernel, M=M, N=N, K=K, BM=BM, BN=BN, BK=BK, GM=GM),
      out_shape=out_shapes,
      grid=((M // BM) * (N // BN),),
      interpret=pallas_utils.interpret(),
      compiler_params=plgpu.CompilerParams(num_warps=nw, num_stages=ns),
  )(*args)


def supported(M, D, F, dtype) -> bool:
  """Whether resblock runs for r [M, D] and a hidden width F: on GPU (or in
  Pallas interpret mode), bf16, and sizes that tile. SLIPPI_FFW_PALLAS=0
  disables."""
  if os.environ.get('SLIPPI_FFW_PALLAS', '1') == '0':
    return False
  if jax.default_backend() != 'gpu' and not pallas_utils.interpret():
    return False
  if dtype != jnp.bfloat16:
    return False
  for (BM, BN, BK, *_), (m, n, k) in ((FWD_CONFIG, (M, F, D)), (BWD_CONFIG, (M, F, D)),
                                       (DOWN_CONFIG, (M, D, F))):
    if m % BM or n % BN or k % BK:
      return False
  for rows, step, _ in (LN_FWD_CONFIG, LN_BWD_CONFIG):
    if M % rows or rows % step:
      return False
  return D <= 2048


@functools.partial(jax.custom_vjp, nondiff_argnums=(7,))
def resblock(r, gamma, beta, w1, b1, w2, b2, eps):
  """r + gelu(LayerNorm(r) @ w1 + b1) @ w2 + b2 for r [M, D] (all bf16)."""
  return _rb_fwd(r, gamma, beta, w1, b1, w2, b2, eps)[0]


def _rb_fwd(r, gamma, beta, w1, b1, w2, b2, eps):
  M, D = r.shape
  F = w1.shape[1]
  act = r.dtype
  x, mean, rstd = _rows_call(
      _ln_fwd_kernel, (r, pallas_utils.match_vma(gamma, r), pallas_utils.match_vma(beta, r)),
      (((M, D), act), ((M,), F32), ((M,), F32)), M, LN_FWD_CONFIG, D=D, eps=eps)
  h, a = _call(_fwd_kernel, (x, pallas_utils.match_vma(w1, x), pallas_utils.match_vma(b1, x)),
               (((M, F), act), ((M, F), act)), M, F, D, FWD_CONFIG)
  out, = _call(_down_kernel, (a, pallas_utils.match_vma(w2, a), pallas_utils.match_vma(b2, a), r),
               (((M, D), act),), M, D, F, DOWN_CONFIG)
  return out, (r, x, mean, rstd, h, a, gamma, beta, w1, b1, w2, b2)


def _rb_bwd(eps, res, dout):
  del eps
  r, x, mean, rstd, h, a, gamma, beta, w1, b1, w2, b2 = res
  M, D = r.shape
  F = w1.shape[1]
  act = r.dtype
  dout = dout.astype(act)
  dh, dbp = _call(_bwd_kernel, (dout, pallas_utils.match_vma(w2, dout), h),
                  (((M, F), act), ((M // BWD_CONFIG[0], F), F32)), M, F, D, BWD_CONFIG)
  dx = jnp.dot(dh, w1.T)
  nprog = M // LN_BWD_CONFIG[0]
  dr, dgp, dbetap, db2p = _rows_call(
      _ln_bwd_kernel, (r, dx, dout, pallas_utils.match_vma(gamma, r), mean, rstd),
      (((M, D), act), ((nprog, D), F32), ((nprog, D), F32), ((nprog, D), F32)),
      M, LN_BWD_CONFIG, D=D)
  grads = [jnp.sum(dgp, axis=0), jnp.sum(dbetap, axis=0),
           jnp.einsum('md,mf->df', x, dh, preferred_element_type=F32),
           jnp.sum(dbp, axis=0),
           jnp.einsum('mf,md->fd', a, dout, preferred_element_type=F32),
           jnp.sum(db2p, axis=0)]
  varying = tuple(sorted(jax.typeof(grads[2]).mat.varying))
  if varying:
    grads = [jax.lax.psum(g, varying) for g in grads]
  return (dr, *(g.astype(p.dtype) for g, p in zip(grads, (gamma, beta, w1, b1, w2, b2))))


resblock.defvjp(_rb_fwd, _rb_bwd)
