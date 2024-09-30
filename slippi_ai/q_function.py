import typing as tp

import tensorflow as tf
import sonnet as snt

from slippi_ai import data, embed, networks, tf_utils
from slippi_ai.rl_lib import discounted_returns
from slippi_ai.networks import RecurrentState

class QOutputs(tp.NamedTuple):
  returns: tf.Tensor  # [T, B]
  advantages: tf.Tensor  # [T, B]
  loss: tf.Tensor
  hidden_states: RecurrentState  # [T, B]
  metrics: dict


class QFunction(snt.Module):

  def __init__(
      self,
      network_config: dict,
      embed_state_action: embed.StructEmbedding[embed.StateAction],
      embed_action: embed.StructEmbedding[embed.Action],
  ):
    super().__init__(name='QFunction')
    self.core_net = networks.construct_network2(network_config, 'core')
    self.action_net = networks.construct_network2(network_config, 'action')
    self.value_head = snt.Linear(1, name='value_head')
    self.q_head = snt.Linear(1, name='q_head')
    self.embed_state_action = embed_state_action
    self.embed_action = embed_action

  def initial_state(self, batch_size: int) -> networks.RecurrentState:
    return self.core_net.initial_state(batch_size)

  def initialize_variables(self):
    dummy_state_action = tf.nest.map_structure(
        tf.convert_to_tensor, self.embed_state_action.dummy([2, 1]))
    dummy_reward = tf.zeros([1, 1], tf.float32)
    dummy_frames = data.Frames(dummy_state_action, dummy_reward)
    initial_state = self.initial_state(1)
    self.loss(dummy_frames, initial_state, discount=0)

  def loss(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
      discount: float,
      discount_on_death: tp.Optional[float] = None,
  ) -> tp.Tuple[QOutputs, RecurrentState]:
    """Computes prediction loss on a batch of frames.

    Args:
      frames: Time-major batch of states, actions, and rewards.
        Assumed to have one frame of overlap.
      initial_state: Batch of initial recurrent states.
      discount: Per-frame discount factor for returns.
      discount_on_death: Discount factor to use when a player dies.
    """
    rewards = frames.reward

    all_inputs = self.embed_state_action(frames.state_action)
    inputs, last_input = all_inputs[:-1], all_inputs[-1]
    outputs, hidden_states = self.core_net.scan(inputs, initial_state)
    final_state = tf.nest.map_structure(lambda t: t[-1], hidden_states)

    # Includes "overlap" frame.
    # unroll_length = state_action.state.stage.shape[0] - delay

    # Compute value loss
    values = tf.squeeze(self.value_head(outputs), -1)
    last_output, _ = self.core_net.step(last_input, final_state)
    last_value = tf.squeeze(self.value_head(last_output), -1)
    discounts = tf.fill(tf.shape(rewards), tf.cast(discount, tf.float32))

    if discount_on_death is not None:
      death_happened = tf.abs(rewards) > 0.9  # this is a hack
      discounts = tf.where(
          death_happened, tf.cast(discount_on_death, tf.float32), discounts)

    value_targets = discounted_returns(
        rewards=rewards,
        discounts=discounts,
        bootstrap=last_value)
    value_targets = tf.stop_gradient(value_targets)
    advantages = value_targets - values
    value_loss = tf.square(advantages)

    _, value_variance = tf_utils.mean_and_variance(value_targets)
    uev = value_loss / (value_variance + 1e-8)

    # Compute Q loss
    next_actions = tf.nest.map_structure(
        lambda t: t[1:], frames.state_action.action)
    # Here we are batching over time (and batch)
    q_values = snt.BatchApply(self.q_values_from_hidden_states)(
        hidden_states, next_actions)

    q_loss = tf.square(value_targets - q_values)
    quev = q_loss / (value_variance + 1e-8)
    uev_delta = uev - quev

    metrics = {
        'reward': tf_utils.get_stats(rewards),
        'return': tf_utils.get_stats(value_targets),
        'v': {
            'loss': value_loss,
            'uev': uev,
        },
        'q': {
            'loss': q_loss,
            'uev': quev,
            'uev_delta': uev_delta,
            'rel_v_loss': q_loss / (value_loss + 1e-8),
        },
    }

    outputs = QOutputs(
        returns=value_targets,
        advantages=advantages,
        loss=value_loss + q_loss,
        hidden_states=hidden_states,
        metrics=metrics,
    )

    return outputs, final_state

  def q_values_from_hidden_states(
      self,
      hidden_states: RecurrentState,
      actions: embed.Action,
  ) -> tf.Tensor:
    action_inputs = self.embed_action(actions)
    action_outputs, _ = self.action_net.step(action_inputs, hidden_states)
    return tf.squeeze(self.q_head(action_outputs), -1)

@snt.allow_empty_variables
class FakeQFunction(snt.Module):

  def initial_state(self, batch_size: int) -> RecurrentState:
    del batch_size
    return ()

  def loss(self, frames: data.Frames, initial_state, discount):
    del discount

    outputs = QOutputs(
        returns=tf.zeros_like(frames.reward),
        loss=tf.constant(0, dtype=tf.float32),
        advantages=tf.zeros_like(frames.reward),
        metrics={},
    )

    return outputs, initial_state
