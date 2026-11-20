"""EXPERIMENT: LSTM recurrence in plain JAX with a hand-written VJP.

Same interface as lstm_pallas.lstm_recurrence. The forward scan saves the gate
activations; the backward scan only forms dgates and dh_prev = dgates @ Wh^T per
step, and the weight/bias gradients are one matmul/reduction over the whole
sequence.
"""
# pylint: disable=invalid-name
import jax
import jax.numpy as jnp

F32 = jnp.float32


@jax.custom_vjp
def lstm_recurrence(gx, wh, b, h0, c0, reset_i):
  """gx: [T, B, 4H] without bias, wh: [H, 4H], b: [4H], h0/c0: [B, H],
  reset_i: [T, B]. Returns (h [T, B, H], (c_T, h_T))."""
  return _fwd(gx, wh, b, h0, c0, reset_i)[0]


def _fwd(gx, wh, b, h0, c0, reset_i):
  dt = gx.dtype
  reset = reset_i.astype(bool)

  def step(carry, xs):
    c, h = carry
    g, r = xs
    c = jnp.where(r[:, None], 0, c)
    h = jnp.where(r[:, None], 0, h)
    pre = (g + b + jnp.dot(h, wh)).astype(F32)
    i, f, gg, o = jnp.split(pre, 4, axis=-1)
    si, sf, tg, so = jax.nn.sigmoid(i), jax.nn.sigmoid(f), jnp.tanh(gg), jax.nn.sigmoid(o)
    c_new = sf * c.astype(F32) + si * tg
    tc = jnp.tanh(c_new)
    h_new = (so * tc).astype(dt)
    acts = jnp.concatenate([si, sf, tg, so], axis=-1).astype(dt)
    return (c_new.astype(dt), h_new), (h_new, h, c, acts, tc.astype(dt))

  (cT, hT), (ys, hprev, cprev, acts, tcs) = jax.lax.scan(step, (c0, h0), (gx, reset))
  return (ys, (cT, hT)), (wh, b, hprev, cprev, acts, tcs, reset)


def _bwd(res, cot):
  wh, b, hprev, cprev, acts, tcs, reset = res
  dys, (dcT, dhT) = cot
  dt = hprev.dtype
  wht = wh.T

  def step(carry, xs):
    dh_next, dc_next = carry
    a, tc, cp, r, dy = xs
    si, sf, tg, so = jnp.split(a.astype(F32), 4, axis=-1)
    tc = tc.astype(F32)
    dh = dh_next + dy.astype(F32)
    dc = dc_next + dh * so * (1 - tc * tc)
    di = dc * tg * si * (1 - si)
    df = dc * cp.astype(F32) * sf * (1 - sf)
    dg = dc * si * (1 - tg * tg)
    do = dh * tc * so * (1 - so)
    dgates = jnp.concatenate([di, df, dg, do], axis=-1).astype(dt)
    dh_prev = jnp.dot(dgates, wht, preferred_element_type=F32)
    keep = ~r[:, None]
    return (jnp.where(keep, dh_prev, 0), jnp.where(keep, dc * sf, 0)), dgates

  (dh0, dc0), dgates = jax.lax.scan(
      step, (dhT.astype(F32), dcT.astype(F32)), (acts, tcs, cprev, reset, dys),
      reverse=True)
  dwh = jnp.einsum('tbh,tbg->hg', hprev, dgates, preferred_element_type=F32)
  db = jnp.sum(dgates.astype(F32), axis=(0, 1))
  # Under shard_map the weights are replicated while activations vary.
  varying = tuple(sorted(jax.typeof(dwh).mat.varying - jax.typeof(wh).mat.varying))
  if varying:
    dwh = jax.lax.psum(dwh, varying)
    db = jax.lax.psum(db, varying)
  return (dgates, dwh.astype(wh.dtype), db.astype(b.dtype),
          dh0.astype(dt), dc0.astype(dt), None)


lstm_recurrence.defvjp(_fwd, _bwd)
