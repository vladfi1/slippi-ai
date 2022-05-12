from typing import Any, Tuple
import sonnet as snt
import tensorflow as tf

from slippi_ai.controller_heads import ControllerHead
from slippi_ai import networks, types
from slippi_ai.rl_lib import discounted_returns
from slippi_ai import networks, embed

RecurrentState = networks.RecurrentState

class Policy(snt.Module):

  def __init__(
      self,
      network: networks.Network,
      controller_head: ControllerHead,
      embed_state_action: embed.StructEmbedding[embed.StateActionReward],
  ):
    super().__init__(name='Policy')
    self.network = network
    self.controller_head = controller_head
    self.embed_state_action = embed_state_action
    self.initial_state = self.network.initial_state
    self.value_head = snt.Linear(1, name='value_head')

  def loss(
      self,
      state_action: embed.StateActionReward,
      initial_state: RecurrentState,
      value_cost: float = 0.5,
      discount: float = 0.99,
  ) -> Tuple[tf.Tensor, RecurrentState, dict]:
    inputs = self.embed_state_action(state_action)
    outputs, final_state = self.network.unroll(inputs, initial_state)

    action = state_action.action
    prev_action = tf.nest.map_structure(lambda t: t[:-1], action)
    next_action = tf.nest.map_structure(lambda t: t[1:], action)

    distances = self.controller_head.distance(
        outputs[:-1], prev_action, next_action)
    policy_loss = tf.add_n(tf.nest.flatten(distances))

    # compute value loss
    values = tf.squeeze(self.value_head(outputs), -1)
    num_frames = tf.cast(state_action.action.repeat[1:] + 1, tf.float32)
    discounts = tf.pow(tf.cast(discount, tf.float32), num_frames)
    value_targets = discounted_returns(
        rewards=tf.cast(state_action.reward[1:], tf.float32),
        discounts=discounts,
        bootstrap=values[-1])
    value_targets = tf.stop_gradient(value_targets)
    value_loss = tf.square(value_targets - values[:-1])
    value_stddev = tf.sqrt(tf.reduce_mean(value_loss))

    # compute metrics
    total_loss = policy_loss + value_cost * value_loss
    weighted_loss = tf.reduce_sum(total_loss) / tf.reduce_sum(num_frames)

    metrics = dict(
        policy=dict(
            types.nt_to_nest(distances),
            loss=policy_loss,
        ),
        value=dict(
            stddev=value_stddev,
            loss=value_loss,
        ),
        total_loss=total_loss,
        weighted_loss=weighted_loss,
    )

    return total_loss, final_state, metrics

  def sample(
      self,
      state_action: embed.StateActionReward,
      initial_state: RecurrentState,
      **kwargs,
  ) -> Tuple[embed.Action, RecurrentState]:
    input = self.embed_state_action(state_action)
    output, final_state = self.network.unroll(input, initial_state)

    prev_action = state_action.action
    next_action = self.controller_head.sample(
        output, prev_action, **kwargs)
    return next_action, final_state
