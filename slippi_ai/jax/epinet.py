"""Epinets, from "Epistemic Neural Networks" (arXiv:2107.08924).

An epinet supplements a base network's output head with a term
sigma(x, z) that depends on an epistemic index z ~ N(0, I). Variation of
the output with z expresses epistemic (resolvable) uncertainty. The epinet
is the sum of a learnable network and a frozen prior network:

  f(x, z) = base(x) + sigma_L(sg[x], z) + prior_scale * sigma_P(sg[x], z)

where both sigma_L and sigma_P have the form mlp([x, z])^T z, so the output
at z = 0 is exactly the base network's output.
"""

import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx

from slippi_ai.jax import jax_utils


@dataclasses.dataclass
class EpinetConfig:
  # Unused; kept so that configs from old checkpoints still parse.
  enabled: bool = False
  index_dim: int = 8
  num_layers: int = 2
  hidden_size: int = 16
  prior_scale: float = 1.0


def epistemic_metrics(vs: jax.Array) -> dict:
  """Epistemic uncertainty statistics for per-index predictions [N, ...].

  Requires at least 2 indices; with fewer the statistics are NaN.

  The per-state std is a noisy estimate of the true sigma, with a sampling std
  of its own of ~sigma/sqrt(2(N-1)). Summaries of how it varies across states
  are therefore dominated by that noise floor unless N is large, which makes
  them incomparable between runs (or between train and eval) that use different
  numbers of indices. The summaries below instead work on the variance scale,
  where the chi-squared sampling moments are exact, and subtract the noise off.

  `epistemic_fraction` is then flat in N. `dispersion` still reads low at small
  N (~7% at N=4, exact by N=16) because the sqrt of a noisy unbiased estimate
  is biased down; it is comparable across N only for N >~ 16.
  """
  n = vs.shape[0]

  # Unbiased per-state epistemic variance: E[var | sigma] = sigma^2.
  var = jnp.var(vs, axis=0, ddof=1)  # [...]
  mean_var = jnp.mean(var)  # unbiased for E_s[sigma^2]

  # Since E[var^2 | sigma] = sigma^4 (n+1)/(n-1), this is unbiased for the
  # across-state variance of the true epistemic variance, Var_s(sigma^2).
  across_var = (
      (n - 1) / (n + 1) * jnp.mean(jnp.square(var)) - jnp.square(mean_var))
  # Normalized by the mean so as to be dimensionless: the q-value scale drifts
  # over training, but this does not.
  dispersion = jnp.sqrt(jnp.maximum(across_var, 0)) / (mean_var + 1e-8)

  # Law of total variance: Var_{s,z}[q] = E_s[Var_z q] + Var_s[E_z q], i.e.
  # epistemic uncertainty plus genuine discrimination between states. The
  # empirical variance of the ensemble mean overstates the latter by exactly
  # E_s[sigma^2]/n, since the mean over n indices is itself noisy.
  between_var = jnp.maximum(jnp.var(jnp.mean(vs, axis=0)) - mean_var / n, 0)
  epistemic_fraction = mean_var / (mean_var + between_var + 1e-8)

  broadcast = lambda x: jnp.broadcast_to(x, var.shape)
  return dict(
      epistemic_std=jnp.sqrt(var),
      dispersion=broadcast(dispersion),
      epistemic_fraction=broadcast(epistemic_fraction),
  )


class PriorParam(nnx.Variable):
  """Parameter of a prior network; not an nnx.Param, so never trained."""


class FrozenLinear(nnx.Module):
  """Like nnx.Linear, but with untrainable PriorParam weights."""

  def __init__(self, rngs: nnx.Rngs, in_size: int, out_size: int):
    initializer = nnx.initializers.lecun_normal()
    self.kernel = PriorParam(initializer(rngs.params(), (in_size, out_size)))
    self.bias = PriorParam(jnp.zeros((out_size,)))

  def __call__(self, x: jax.Array) -> jax.Array:
    return x @ self.kernel.value + self.bias.value


class FrozenMLP(nnx.Module):
  """Like jax_utils.MLP, but with untrainable PriorParam weights."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      input_size: int,
      features: list[int],
      activation=jax.nn.relu,
  ):
    self.activation = activation

    layers = []
    in_size = input_size
    for i, out_size in enumerate(features):
      if i > 0:
        layers.append(activation)
      layers.append(FrozenLinear(rngs, in_size, out_size))
      in_size = out_size

    self.layers = nnx.List(layers)
    self.output_size = in_size

  def __call__(self, x: jax.Array) -> jax.Array:
    for layer in self.layers:
      x = layer(x)
    return x


class Epinet(nnx.Module):
  """Additive epinet with a learnable component and a frozen prior."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      input_size: int,
      output_size: int,
      config: EpinetConfig,
  ):
    self.config = config
    self.output_size = output_size

    features = [config.hidden_size] * config.num_layers
    features.append(config.index_dim * output_size)
    in_size = input_size + config.index_dim

    self.learnable = jax_utils.MLP(rngs, in_size, features)
    self.prior = FrozenMLP(rngs, in_size, features)

    # Zero-init the final learnable layer so that the epinet's initial output
    # is just the prior.
    final_layer: nnx.Linear = self.learnable.layers[-1]
    final_layer.kernel[...] = jnp.zeros_like(final_layer.kernel[...])

  def __call__(
      self,
      x: jax.Array,  # [..., input_size]
      z: jax.Array,  # [..., index_dim], broadcastable against x
  ) -> jax.Array:  # [..., output_size]
    x = jax.lax.stop_gradient(x)
    z = jnp.broadcast_to(z, x.shape[:-1] + z.shape[-1:]).astype(x.dtype)
    xz = jnp.concatenate([x, z], axis=-1)

    def project(out: jax.Array) -> jax.Array:
      # [..., index_dim * output_size] -> [..., output_size]
      out = out.reshape(
          out.shape[:-1] + (self.config.index_dim, self.output_size))
      return jnp.einsum('...dc,...d->...c', out, z)

    prior = jax.lax.stop_gradient(project(self.prior(xz)))
    return project(self.learnable(xz)) + self.config.prior_scale * prior
