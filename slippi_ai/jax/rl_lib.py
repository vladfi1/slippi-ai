import numpy as np
import jax
import jax.numpy as jnp

Array = jax.Array | np.ndarray

def discount_from_halflife(halflife_seconds: float, fps: float = 60.0) -> float:
  """Computes the per-frame discount factor corresponding to a given halflife."""
  return 0.5 ** (1 / (halflife_seconds * fps))

def discounted_returns(
    rewards: Array,
    discounts: Array,
    bootstrap: Array,
) -> jax.Array:
  """Computes discounted returns.

  Args:
    rewards: Reward tensor of shape [T, B].
    discounts: The discount factors at each step. Shape [T, B].
    bootstrap: Predicted returns on the last step. Shape [B].

  Returns:
    The discounted returns, of shape [T, B].
  """
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
) -> jax.Array:
  def scan_fn(future_value, inputs):
    reward, discount, current_value, lambda_ = inputs
    value = reward + discount * future_value
    smoothed_value = lambda_ * value + (1 - lambda_) * current_value
    return smoothed_value, smoothed_value

  _, smoothed_returns = jax.lax.scan(
      scan_fn, bootstrap, (rewards, discounts, values, lambdas), reverse=True)
  return smoothed_returns

def generalized_returns_with_resetting(
    rewards: Array,
    values: Array,  # For t=[0, T-1]
    is_resetting: Array,  # For t=[1, T]
    bootstrap: Array,  # For t=T
    discount: float,
    lambda_: float = 1.0,
) -> jax.Array:
  discounts = jnp.where(is_resetting, 0.0, discount)
  lambdas = jnp.where(is_resetting, 0.0, lambda_)
  return generalized_returns(
      rewards=rewards,
      discounts=discounts,
      values=values,
      bootstrap=bootstrap,
      lambdas=lambdas,
  )
