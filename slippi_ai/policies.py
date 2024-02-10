from typing import Any, Tuple
import sonnet as snt
import tensorflow as tf

from slippi_ai.controller_heads import ControllerHead
from slippi_ai.rl_lib import discounted_returns
from slippi_ai import data, networks, embed, types, utils

RecurrentState = networks.RecurrentState

class Policy(snt.Module):

  def __init__(
      self,
      network: networks.Network,
      controller_head: ControllerHead,
      embed_state_action: embed.StructEmbedding[embed.StateAction],
      train_value_head: bool = True,
  ):
    super().__init__(name='Policy')
    self.network = network
    self.controller_head = controller_head
    self.embed_state_action = embed_state_action
    self.initial_state = self.network.initial_state
    self.train_value_head = train_value_head

    if train_value_head:
      self.value_head = snt.Linear(1, name='value_head')

  @property
  def controller_embedding(self) -> embed.Embedding[embed.Controller, embed.Action]:
    return self.controller_head.controller_embedding()

  def loss(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
      value_cost: float = 0.5,
      discount: float = 0.99,
  ) -> Tuple[tf.Tensor, RecurrentState, dict]:
    """Computes prediction loss on a batch of frames.

    Args:
      frames: Time-major batch of states, actions, and rewards.
      initial_state: Batch of initial recurrent states.
      value_cost: Weighting of value function loss.
      discount: Per-frame discount factor for returns.
    """
    state_action = frames.state_action
    inputs = self.embed_state_action(state_action)
    outputs, final_state = self.network.unroll(inputs, initial_state)

    action = state_action.action
    prev_action = tf.nest.map_structure(lambda t: t[:-1], action)
    next_action = tf.nest.map_structure(lambda t: t[1:], action)

    distances = self.controller_head.distance(
        outputs[:-1], prev_action, next_action)
    policy_loss = tf.add_n(tf.nest.flatten(distances))
    total_loss = policy_loss

    metrics = dict(
        policy=dict(
            types.nt_to_nest(distances),
            loss=policy_loss,
        )
    )

    # compute value loss
    if self.train_value_head:
      rewards = frames.reward
      values = tf.squeeze(self.value_head(outputs), -1)
      discounts = tf.fill(tf.shape(rewards), tf.cast(discount, tf.float32))
      value_targets = discounted_returns(
          rewards=rewards,
          discounts=discounts,
          bootstrap=values[-1])
      value_targets = tf.stop_gradient(value_targets)
      value_loss = tf.square(value_targets - values[:-1])

      _, value_variance = utils.mean_and_variance(value_targets)
      uev = value_loss / (value_variance + 1e-8)

      reward_mean, reward_variance = utils.mean_and_variance(rewards)

      metrics['value'] = {
          'reward': dict(
              mean=reward_mean,
              variance=reward_variance,
              max=tf.reduce_max(rewards),
              min=tf.reduce_min(rewards),
          ),
          'return': value_targets,
          'loss': value_loss,
          'variance': value_variance,
          'uev': uev,  # unexplained variance
      }

      total_loss += value_cost * value_loss

    metrics.update(
        total_loss=total_loss,
    )

    return total_loss, final_state, metrics

  def sample(
      self,
      state_action: embed.StateAction,
      initial_state: RecurrentState,
      **kwargs,
  ) -> Tuple[embed.Action, RecurrentState]:
    input = self.embed_state_action(state_action)
    output, final_state = self.network.step(input, initial_state)

    prev_action = state_action.action
    next_action = self.controller_head.sample(
        output, prev_action, **kwargs)
    return next_action, final_state

DEFAULT_CONFIG = dict(
    train_value_head=True,
)
