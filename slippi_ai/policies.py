import dataclasses
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
      delay: int = 0,
  ):
    super().__init__(name='Policy')
    self.network = network
    self.controller_head = controller_head
    self.embed_state_action = embed_state_action
    self.initial_state = self.network.initial_state
    self.train_value_head = train_value_head
    self.delay = delay

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
    # Let's say that delay is D and total unroll-length is U + D + 1 (overlap
    # is D + 1). Then the first trajectory has game states [0, U + D] and the
    # second trajectory has game states [U, 2U + D]. That means that we want to
    # use states [0, U-1] to predict actions [D + 1, U + D] (with previous
    # actions being [D, U + D - 1]). The final hidden state should be the one
    # preceding timestep U, meaning we compute it from game states [0, U-1]. We
    # will use game state U to bootstrap the value function.

    delay = self.delay
    state_action = frames.state_action

    # Includes "overlap" frame.
    unroll_length = state_action.state.stage.shape[0] - delay

    # Match state t with action t + delay.
    delayed_state_action = embed.StateAction(
        state=tf.nest.map_structure(
            lambda t: t[:unroll_length], state_action.state),
        action=tf.nest.map_structure(
            lambda t: t[delay:], state_action.action),
        name=state_action.name[delay:],
    )
    del state_action

    all_inputs = self.embed_state_action(delayed_state_action)
    inputs, last_input = all_inputs[:-1], all_inputs[-1]
    outputs, final_state = self.network.unroll(inputs, initial_state)

    # Predict next action.
    action = delayed_state_action.action
    prev_action = tf.nest.map_structure(lambda t: t[:-1], action)
    next_action = tf.nest.map_structure(lambda t: t[1:], action)

    distances = self.controller_head.distance(
        outputs, prev_action, next_action)
    policy_loss = tf.add_n(tf.nest.flatten(distances))
    total_loss = policy_loss

    metrics = dict(
        policy=dict(
            types.nt_to_nest(distances),
            loss=policy_loss,
        )
    )

    # Compute value loss.
    if self.train_value_head:
      # Only use rewards that follow actions.
      rewards = frames.reward[delay:]

      values = tf.squeeze(self.value_head(outputs), -1)
      last_output, _ = self.network.step(last_input, final_state)
      last_value = tf.squeeze(self.value_head(last_output), -1)
      discounts = tf.fill(tf.shape(rewards), tf.cast(discount, tf.float32))
      value_targets = discounted_returns(
          rewards=rewards,
          discounts=discounts,
          bootstrap=last_value)
      value_targets = tf.stop_gradient(value_targets)
      value_loss = tf.square(value_targets - values)

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

@dataclasses.dataclass
class PolicyConfig:
  train_value_head: bool = True
  delay: int = 0
