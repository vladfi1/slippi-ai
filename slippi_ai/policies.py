import typing as tp
import sonnet as snt
import tensorflow as tf

from slippi_ai.controller_heads import ControllerHead
from slippi_ai.rl_lib import discounted_returns
from slippi_ai import networks, embed, types, utils

RecurrentState = networks.RecurrentState

class SampleOutputs(tp.NamedTuple):
  action: embed.ActionWithRepeat
  log_prob: float
  next_state: RecurrentState

class UnrollOutputs(tp.NamedTuple):
  log_probs: tf.Tensor  # [T-1, B]
  values: tf.Tensor  # [T, B]
  final_state: RecurrentState  # [B]
  metrics: dict  # mixed

class Policy(snt.Module):

  def __init__(
      self,
      network: networks.Network,
      controller_head: ControllerHead,
      embed_state_action: embed.StructEmbedding[embed.StateActionReward],
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

  def unroll(
      self,
      state_action: embed.StateActionReward,
      initial_state: RecurrentState,
      discount: float = 0.99,
  ) -> UnrollOutputs:
    inputs = self.embed_state_action(state_action)
    outputs, final_state = self.network.unroll(inputs, initial_state)

    action = state_action.action
    prev_action = tf.nest.map_structure(lambda t: t[:-1], action)
    next_action = tf.nest.map_structure(lambda t: t[1:], action)

    distances = self.controller_head.distance(
        outputs[:-1], prev_action, next_action)
    policy_loss = tf.add_n(tf.nest.flatten(distances))
    log_probs = -policy_loss

    metrics = dict(
        policy=dict(
            types.nt_to_nest(distances),
            loss=policy_loss,
        )
    )

    num_frames = tf.cast(action.repeat[1:] + 1, tf.float32)

    # compute value loss
    values = tf.squeeze(self.value_head(outputs), -1)
    discounts = tf.pow(tf.cast(discount, tf.float32), num_frames)
    value_targets = discounted_returns(
        rewards=tf.cast(state_action.reward[1:], tf.float32),
        discounts=discounts,
        bootstrap=values[-1])
    value_targets = tf.stop_gradient(value_targets)
    value_loss = tf.square(value_targets - values[:-1])

    _, value_variance = utils.mean_and_variance(value_targets)
    uev = value_loss / (value_variance + 1e-8)

    reward_mean, reward_variance = utils.mean_and_variance(
        state_action.reward)

    metrics['value'] = {
        'reward': dict(
            mean=reward_mean,
            variance=reward_variance,
            max=tf.reduce_max(state_action.reward),
            min=tf.reduce_min(state_action.reward),
        ),
        'return': value_targets,  # [T-1, B]
        'loss': value_loss,  # [T-1, B]
        'variance': value_variance,
        'uev': uev,  # unexplained variance
    }

    return UnrollOutputs(
        log_probs=log_probs,
        values=values,
        final_state=final_state,
        metrics=metrics)

  def imitation_loss(
      self,
      state_action: embed.StateActionReward,
      initial_state: RecurrentState,
      discount: float = 0.99,
      value_cost: float = 0.5,
  ) -> tp.Tuple[tf.Tensor, RecurrentState, dict]:
    unroll_outputs = self.unroll(
        state_action, initial_state,
        discount=discount,
    )

    metrics = unroll_outputs.metrics

    total_loss = metrics['policy']['loss']
    if self.train_value_head:
      total_loss += value_cost * metrics['value']['loss']

    metrics.update(
        total_loss=total_loss,
    )

    return total_loss, unroll_outputs.final_state, metrics

  def sample(
      self,
      state_action: embed.StateActionReward,
      initial_state: RecurrentState,
      **kwargs,
  ) -> tp.Tuple[embed.ActionWithRepeat, RecurrentState]:
    input = self.embed_state_action(state_action)
    output, final_state = self.network.step(input, initial_state)

    prev_action = state_action.action
    next_action = self.controller_head.sample(
        output, prev_action, **kwargs)
    return next_action, final_state

DEFAULT_CONFIG = dict(
    train_value_head=True,
)
