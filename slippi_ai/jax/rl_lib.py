import typing as tp

import numpy as np
import jax
import jax.numpy as jnp

Array = jax.Array | np.ndarray

class Gaussian(tp.NamedTuple):
  """A diagonal Gaussian, parameterized by mean and variance."""
  mean: jax.Array  # [...]
  variance: jax.Array  # [...]

def discount_from_halflife(
    halflife_seconds: float,
    frame_skip: int = 1,
    fps: float = 60,
) -> float:
  """Computes the per-frame discount factor corresponding to a given halflife."""
  return 0.5 ** (1 / (halflife_seconds * fps / frame_skip))

def discounted_returns(
    rewards: Array,
    discounts: Array,
    bootstrap: Array,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
  """Computes discounted returns.

  Args:
    rewards: Reward tensor of shape [T, B].
    discounts: The discount factors at each step. Shape [T, B].
    bootstrap: Predicted returns on the last step. Shape [B].

  Returns:
    The discounted returns, of shape [T, B].
  """
  rewards = rewards.astype(dtype)
  discounts = discounts.astype(dtype)
  bootstrap = bootstrap.astype(dtype)

  def scan_fn(acc, inputs):
    reward, discount = inputs
    value = reward + discount * acc
    return value, value

  _, returns = jax.lax.scan(
      scan_fn, bootstrap, (rewards, discounts), reverse=True)
  return returns

def generalized_returns(
    rewards: Array,
    discounts: Array,
    values: Array,
    bootstrap: Array,
    lambdas: Array,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
  values = values.astype(dtype)
  bootstrap = bootstrap.astype(dtype)
  rewards = rewards.astype(dtype)
  discounts = discounts.astype(dtype)
  lambdas = lambdas.astype(dtype)

  def scan_fn(future_value, inputs):
    reward, discount, current_value, lambda_ = inputs
    value = reward + discount * future_value
    smoothed_value = lambda_ * value + (1 - lambda_) * current_value
    return smoothed_value, value

  _, returns = jax.lax.scan(
      scan_fn, bootstrap, (rewards, discounts, values, lambdas), reverse=True)
  return returns

def generalized_returns_with_resetting(
    rewards: Array,
    values: Array,  # For t=[0, T-1]
    is_resetting: Array,  # For t=[1, T]
    bootstrap: Array,  # For t=T
    discount: float,
    lambda_: float = 1.0,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
  discounts = jnp.where(is_resetting, 0.0, discount)
  lambdas = jnp.full_like(discounts, lambda_)

  return generalized_returns(
      rewards=rewards,
      discounts=discounts,
      values=values,
      bootstrap=bootstrap,
      lambdas=lambdas,
      dtype=dtype,
  )

def generalized_returns_gaussian(
    rewards: Array,
    values: Gaussian,  # [T, B]
    bootstrap: Gaussian,  # [B]
    discounts: Array,
    lambdas: Array,
    dtype: jnp.dtype = jnp.float32,
) -> Gaussian:
  """Distributional version of generalized_returns for Gaussian value estimates.

  Propagates both the mean and variance through the (generalized lambda) Bellman
  backup. Rewards are treated as deterministic, so the one-step backup scales the
  future variance by discount**2. The lambda-interpolation treats the smoothed
  value as the linear combination lambda * backup + (1 - lambda) * value of two
  independent Gaussians, which is itself Gaussian; we fit its mean and variance
  (so the variance picks up squared coefficients).

  Returns a Gaussian whose fields have shape [T, B].
  """
  rewards = rewards.astype(dtype)
  values = Gaussian(values.mean.astype(dtype), values.variance.astype(dtype))
  bootstrap = Gaussian(bootstrap.mean.astype(dtype), bootstrap.variance.astype(dtype))
  discounts = discounts.astype(dtype)
  lambdas = lambdas.astype(dtype)

  def scan_fn(future: Gaussian, inputs):
    reward, discount, current_value, lambda_ = inputs

    # One-step backup; reward is deterministic.
    backup = Gaussian(
        mean=reward + discount * future.mean,
        variance=jnp.square(discount) * future.variance,
    )

    # The carry is the smoothed return: mean and variance of the linear
    # combination of two independent Gaussians lambda * backup + (1 - lambda) *
    # value (variance gets squared coefficients). As in the scalar version, the
    # emitted target is the one-step backup, not the smoothed carry.
    smoothed = Gaussian(
        mean=lambda_ * backup.mean + (1 - lambda_) * current_value.mean,
        variance=(
            jnp.square(lambda_) * backup.variance
            + jnp.square(1 - lambda_) * current_value.variance),
    )
    return smoothed, backup

  _, returns = jax.lax.scan(
      scan_fn, bootstrap,
      (rewards, discounts, values, lambdas),
      reverse=True)
  return returns

def generalized_returns_gaussian_with_resetting(
    rewards: Array,
    values: Gaussian,  # For t=[0, T-1]
    is_resetting: Array,  # For t=[1, T]
    bootstrap: Gaussian,  # For t=T
    discount: float,
    lambda_: float = 1.0,
    dtype: jnp.dtype = jnp.float32,
) -> Gaussian:
  discounts = jnp.where(is_resetting, 0.0, discount)
  lambdas = jnp.full_like(discounts, lambda_)

  return generalized_returns_gaussian(
      rewards=rewards,
      values=values,
      bootstrap=bootstrap,
      discounts=discounts,
      lambdas=lambdas,
      dtype=dtype,
  )
