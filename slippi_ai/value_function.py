import typing as tp

import tensorflow as tf
import sonnet as snt

from melee.enums import Action

from slippi_ai import data, embed, networks, tf_utils, types, utils
from slippi_ai.rl_lib import discounted_returns
from slippi_ai.networks import RecurrentState

class ValueOutputs(tp.NamedTuple):
  returns: tf.Tensor  # [T, B]
  advantages: tf.Tensor  # [T, B]
  loss: tf.Tensor
  metrics: dict

def player_respawn(player: types.Player) -> tf.Tensor:
  """Returns a boolean mask indicating when a player respawns."""
  actions = player.action
  # Note: players seem to be able to skip ON_HALO_WAIT
  respawn = Action.ON_HALO_DESCENT.value
  return tf.logical_and(actions[:-1] != respawn, actions[1:] == respawn)

class ValueFunction(snt.Module):

  def __init__(
      self,
      network_config: dict,
      num_names: int,
      embed_config: embed.EmbedConfig,
  ):
    super().__init__(name='ValueFunction')
    self.network = networks.build_embed_network(
          embed_config=embed_config,
          num_names=num_names,
          network_config=network_config,
    )
    self.value_head = snt.Linear(1, name='value_head')
    self.initial_state = self.network.initial_state

  def loss(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
      discount: float,
      discount_on_death: tp.Optional[float] = None,
  ) -> tp.Tuple[ValueOutputs, RecurrentState]:
    """Computes prediction loss on a batch of frames.

    Args:
      frames: Time-major batch of states, actions, and rewards.
        Assumed to have one frame of overlap.
      initial_state: Batch of initial recurrent states.
      discount: Per-frame discount factor for returns.
      discount_on_death: Discount factor to use when either player *respawns*.
        The reward for KOs comes on the frame of death, which precedes respawn.
    """
    rewards = frames.reward

    inputs = utils.map_nt(lambda t: t[:-1], frames.state_action)
    last_input = utils.map_nt(lambda t: t[-1], frames.state_action)
    outputs, final_state = self.network.unroll(
        inputs, frames.is_resetting[:-1], initial_state)

    # Includes "overlap" frame.
    # unroll_length = state_action.state.stage.shape[0] - delay

    values = tf.squeeze(self.value_head(outputs), -1)
    last_output, _ = self.network.step_with_reset(
        last_input, frames.is_resetting[-1], final_state)
    last_value = tf.squeeze(self.value_head(last_output), -1)
    discounts = tf.fill(tf.shape(rewards), tf.cast(discount, tf.float32))

    if discount_on_death is not None:
      respawn_happened = tf.logical_or(
          player_respawn(frames.state_action.state.p0),
          player_respawn(frames.state_action.state.p1))

      discounts_on_death = tf.fill(
          discounts.shape, tf.cast(discount_on_death, tf.float32))

      discounts = tf.where(respawn_happened, discounts_on_death, discounts)
      assert discounts.shape == rewards.shape

    value_targets = discounted_returns(
        rewards=rewards,
        discounts=discounts,
        bootstrap=last_value)
    value_targets = tf.stop_gradient(value_targets)
    advantages = value_targets - values
    value_loss = tf.square(advantages)

    _, value_variance = tf_utils.mean_and_variance(value_targets)
    uev = value_loss / (value_variance + 1e-8)

    metrics = {
        'reward': tf_utils.get_stats(rewards),
        'return': tf_utils.get_stats(value_targets),
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


@snt.allow_empty_variables
class FakeValueFunction(snt.Module):

  def initial_state(self, batch_size: int) -> RecurrentState:
    del batch_size
    return ()

  def loss(self, frames: data.Frames, initial_state, discount, **_):
    del discount

    outputs = ValueOutputs(
        returns=tf.zeros_like(frames.reward),
        loss=tf.constant(0, dtype=tf.float32),
        advantages=tf.zeros_like(frames.reward),
        metrics={},
    )

    return outputs, initial_state
