import tensorflow as tf

def _bellman(value, reward_and_discount):
  reward, discount = reward_and_discount
  return reward + discount * value

def discounted_returns(
    rewards: tf.Tensor,
    discounts: tf.Tensor,
    bootstrap: tf.Tensor,
) -> tf.Tensor:
  """Computes discounted returns.
  
  Args:
    rewards: Reward tensor of shape [T, B].
    discounts: The discount factors at each step. Shape [T, B].
    bootstrap: Predicted returns on the last step. Shape [B].

  Returns:
    The discounted returns, of shape [T, B].
  """
  return tf.scan(_bellman, (rewards, discounts), bootstrap, reverse=True)
