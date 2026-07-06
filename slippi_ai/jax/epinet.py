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
  enabled: bool = False
  index_dim: int = 8
  num_layers: int = 2
  hidden_size: int = 16
  prior_scale: float = 1.0


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
