import numpy as np
import jax

Array = jax.Array | np.ndarray


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
