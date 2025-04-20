import dataclasses
import typing as tp

import tensorflow as tf
import sonnet as snt

from slippi_ai import data, embed, networks, tf_utils
from slippi_ai.utils import field
from slippi_ai.rl_lib import discounted_returns
from slippi_ai.networks import RecurrentState
from slippi_ai.nash.data import split, merge
from slippi_ai.tf_utils import expand_tile

class QOutputs(tp.NamedTuple):
  returns: tf.Tensor  # [T, B]
  advantages: tf.Tensor  # [T, B]
  values: tf.Tensor  # [T, B]
  q_values: tf.Tensor
  loss: tf.Tensor
  hidden_states: RecurrentState  # [T, B]
  metrics: dict

@dataclasses.dataclass
class HeadConfig:
  num_layers: int = 1
  hidden_size: int = 128

@dataclasses.dataclass
class QFunctionConfig:
  head: HeadConfig = field(HeadConfig)
  network: dict = field(lambda: networks.DEFAULT_CONFIG)

def make_value_head(
    num_layers: int,
    hidden_size: int,
    name: str = 'value_head',
) -> snt.Module:
  with tf.name_scope(name):
    return snt.Sequential([
        snt.nets.MLP([hidden_size] * num_layers + [1]),
        lambda x: tf.squeeze(x, -1),
    ])


def to_merged_value_outputs(outputs: tf.Tensor, batch_dim=1) -> tf.Tensor:
  """Goes from [T, 2B, O] to [T, 2B, 2O]."""
  split_outputs = split(outputs, axis=batch_dim)
  p0_outputs = tf.concat(split_outputs, -1)
  p1_outputs = tf.concat(list(reversed(split_outputs)), -1)
  return tf.concat([p0_outputs, p1_outputs], axis=batch_dim)

class QFunction(snt.Module):

  def __init__(
      self,
      config: QFunctionConfig,
      embed_state_action: embed.StructEmbedding[embed.StateAction],
      embed_action: embed.StructEmbedding[embed.Action],
  ):
    super().__init__(name='QFunction')
    self.core_net = networks.construct_network2(config.network, 'core')
    self.action_net = networks.construct_network2(config.network, 'action')

    self.value_head = make_value_head(
        config.head.num_layers, config.head.hidden_size, name='value_head')
    self.q_head = make_value_head(
        config.head.num_layers, config.head.hidden_size, name='q_head')
    self.embed_state_action = embed_state_action
    self.embed_action = embed_action

  def initial_state(self, batch_size: int) -> networks.RecurrentState:
    return [self.core_net.initial_state(batch_size)] * 2

  def initialize_variables(self):
    T = 2
    B = 1
    dummy_state_action = tf.nest.map_structure(
        tf.convert_to_tensor, self.embed_state_action.dummy([T, 2 * B]))
    dummy_reward = tf.zeros([T - 1, 2 * B], tf.float32)
    dummy_resetting = tf.zeros([T, 2 * B], tf.bool)
    dummy_frames = data.Frames(
        dummy_state_action, dummy_resetting, dummy_reward)

    initial_state = tf.nest.map_structure(
        lambda *xs: tf.concat(xs, axis=0),
        *self.initial_state(B))
    self.loss(dummy_frames, initial_state, discount=0)

  def loss(
      self,
      frames: data.Frames,  # [T, 2B]
      initial_state: RecurrentState,  # [2B]
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
    del discount_on_death

    # Below tensors are merged by default.

    # TODO: The core_net only sees one player's actions; should it see both?
    all_inputs = self.embed_state_action(frames.state_action)
    inputs, last_input = all_inputs[:-1], all_inputs[-1]
    outputs, hidden_states = self.core_net.scan(
        inputs=inputs,
        reset=frames.is_resetting[:-1],
        initial_state=initial_state,
    )
    final_state = tf.nest.map_structure(lambda t: t[-1], hidden_states)

    # Includes "overlap" frame.
    # unroll_length = state_action.state.stage.shape[0] - delay

    # Compute value loss
    value_outputs = to_merged_value_outputs(outputs, batch_dim=1)
    values = self.value_head(value_outputs)
    last_output, _ = self.core_net.step(last_input, final_state)
    last_value_output = to_merged_value_outputs(last_output, batch_dim=0)
    last_value = self.value_head(last_value_output)

    discounts = tf.fill(tf.shape(frames.reward), tf.cast(discount, tf.float32))

    value_targets = discounted_returns(
        rewards=frames.reward,
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
    q_values = self.q_values_from_hidden_states(hidden_states, next_actions)

    q_loss = tf.square(value_targets - q_values)
    quev = q_loss / (value_variance + 1e-8)
    uev_delta = uev - quev

    metrics = {
        'reward': tf_utils.get_stats(frames.reward),
        'return': tf_utils.get_stats(value_targets),
        'v': {
            'loss': value_loss,
            'uev': uev,
        },
        'q': {
            'loss': q_loss,
            'uev': quev,
            'uev_delta': uev_delta,
            # Take log to result in a geometric mean.
            'rel_v_loss': tf.math.log((value_loss + 1e-8) / (q_loss + 1e-8)),
        },
    }

    outputs = QOutputs(
        returns=value_targets,
        advantages=advantages,
        values=values,
        q_values=q_values,
        loss=value_loss + q_loss,
        hidden_states=hidden_states,
        metrics=metrics,
    )

    return outputs, final_state

  def q_values_from_hidden_states(
      self,
      hidden_states: RecurrentState,  # [T, 2B, ...]
      actions: embed.Action,  # [T, 2B, ...]
  ) -> tf.Tensor:  # [T, 2B]
    action_inputs = self.embed_action(actions)
    outputs, _ = snt.BatchApply(self.action_net.step)(
        action_inputs, hidden_states)  # [T, 2B, O]
    value_outputs = to_merged_value_outputs(outputs)  # [T, 2B, 2O]
    q_values = self.q_head(value_outputs)  # [T, 2B]
    return q_values

  def multi_q_values_from_hidden_states(
      self,
      hidden_states: RecurrentState,  # [T, 2B, ...]
      actions: embed.Action,  # [S, T, 2B, ...]
      sample_dim: int = 0,
      batch_dim: int = 2,
  ) -> tf.Tensor:  # [2, S, S, T, 2B]
    action_inputs = self.embed_action(actions)
    s = action_inputs.shape[0]

    hidden_states = tf.nest.map_structure(
        lambda x: expand_tile(x, axis=sample_dim, multiple=s),
        hidden_states)  # [S, T, 2B, ...]

    outputs, _ = snt.BatchApply(self.action_net.step, num_dims=3)(
        action_inputs, hidden_states)  # [S, T, 2B, O]
    p0_outputs, p1_outputs = split(outputs, axis=batch_dim)  # [S, T, B, O]

    p0_outputs = expand_tile(p0_outputs, axis=sample_dim+1, multiple=s)  # [S, S, T, B, O]
    p1_outputs = expand_tile(p1_outputs, axis=sample_dim, multiple=s)  # [S, S, T, B, O]

    p01_outputs = tf.concat([p0_outputs, p1_outputs], axis=-1)  # [S, S, T, B, 2O]
    p10_outputs = tf.concat([p1_outputs, p0_outputs], axis=-1)  # [S, S, T, B, 2O]

    value_outputs = tf.stack([p01_outputs, p10_outputs], axis=0)  # [2, S, S, T, B, 2O]
    q_values = self.q_head(value_outputs)   # [2, S, S, T, B]

    return q_values

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
