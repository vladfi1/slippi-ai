import typing as tp

import tensorflow as tf
import sonnet as snt

from slippi_ai import data, embed, networks, tf_utils
from slippi_ai.rl_lib import discounted_returns
from slippi_ai.networks import RecurrentState

class ValueOutputs(tp.NamedTuple):
  returns: tf.Tensor  # [T, B]
  advantages: tf.Tensor  # [T, B]
  loss: tf.Tensor
  metrics: dict

class ValueFunction(snt.Module):

  def __init__(
      self,
      network_config: dict,
      embed_state_action: embed.StructEmbedding[embed.StateAction],
  ):
    super().__init__(name='ValueFunction')
    self.network = networks.construct_network(**network_config)
    self.embed_state_action = embed_state_action
    self.value_head = snt.Linear(1, name='value_head')
    self.initial_state = self.network.initial_state

  def loss(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
      discount: float = 0.99,
  ) -> tp.Tuple[ValueOutputs, RecurrentState]:
    """Computes prediction loss on a batch of frames.

    Args:
      frames: Time-major batch of states, actions, and rewards.
        Assumed to have one frame of overlap.
      initial_state: Batch of initial recurrent states.
      discount: Per-frame discount factor for returns.
    """
    rewards = frames.reward

    all_inputs = self.embed_state_action(frames.state_action)
    inputs, last_input = all_inputs[:-1], all_inputs[-1]
    outputs, final_state = self.network.unroll(inputs, initial_state)

    # Includes "overlap" frame.
    # unroll_length = state_action.state.stage.shape[0] - delay

    values = tf.squeeze(self.value_head(outputs), -1)
    last_output, _ = self.network.step(last_input, final_state)
    last_value = tf.squeeze(self.value_head(last_output), -1)
    discounts = tf.fill(tf.shape(rewards), tf.cast(discount, tf.float32))
    value_targets = discounted_returns(
        rewards=rewards,
        discounts=discounts,
        bootstrap=last_value)
    value_targets = tf.stop_gradient(value_targets)
    advantages = value_targets - values
    value_loss = tf.square(advantages)

    _, value_variance = tf_utils.mean_and_variance(value_targets)
    uev = value_loss / (value_variance + 1e-8)

    reward_mean, reward_variance = tf_utils.mean_and_variance(rewards)

    metrics = {
        'reward': dict(
            mean=reward_mean,
            variance=reward_variance,
            max=tf.reduce_max(rewards),
            min=tf.reduce_min(rewards),
        ),
        'loss': value_loss,
        'variance': value_variance,
        'uev': uev,  # unexplained variance
    }

    outputs = ValueOutputs(
        returns=value_targets,
        advantages=advantages,
        loss=value_loss,
        metrics=metrics,
    )

    return outputs, final_state


class FakeValueFunction(snt.Module):

  def initial_state(self, batch_size: int) -> RecurrentState:
    del batch_size
    return ()

  def loss(self, frames, initial_state, discount):
    del frames, initial_state, discount
    return tf.constant(0.), (), {}
