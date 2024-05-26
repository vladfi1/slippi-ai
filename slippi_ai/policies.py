import dataclasses
from typing import Any, Tuple
import typing as tp

import sonnet as snt
import tensorflow as tf

from slippi_ai.controller_heads import (
    ControllerHead,
    DistanceOutputs,
    SampleOutputs,
)
from slippi_ai.rl_lib import discounted_returns
from slippi_ai import data, networks, embed, types, tf_utils
from slippi_ai.value_function import ValueOutputs

RecurrentState = networks.RecurrentState

class UnrollOutputs(tp.NamedTuple):
  log_probs: tf.Tensor  # [T, B]
  distances: DistanceOutputs  # Struct of [T, B]
  value_outputs: ValueOutputs
  final_state: RecurrentState  # [B]
  metrics: dict  # mixed


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

    self.value_head = snt.Linear(1, name='value_head')
    if not train_value_head:
      self.value_head = snt.Sequential([tf.stop_gradient, self.value_head])

  @property
  def controller_embedding(self) -> embed.Embedding[embed.Controller, embed.Action]:
    return self.controller_head.controller_embedding()

  def unroll(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
      discount: float = 0.99,
  ) -> UnrollOutputs:
    """Computes prediction loss on a batch of frames.

    Assumes that actions and rewards are delayed, and that one extra
    "overlap" frame is tacked on at the end.

    Args:
      frames: Time-major batch of states, actions, and rewards.
      initial_state: Batch of initial recurrent states.
      value_cost: Weighting of value function loss.
      discount: Per-frame discount factor for returns.
    """
    all_inputs = self.embed_state_action(frames.state_action)
    inputs, last_input = all_inputs[:-1], all_inputs[-1]
    outputs, final_state = self.network.unroll(inputs, initial_state)

    # Predict next action.
    action = frames.state_action.action
    prev_action = tf.nest.map_structure(lambda t: t[:-1], action)
    next_action = tf.nest.map_structure(lambda t: t[1:], action)

    distance_outputs = self.controller_head.distance(
        outputs, prev_action, next_action)
    distances = distance_outputs.distance
    policy_loss = tf.add_n(tf.nest.flatten(distances))
    log_probs = -policy_loss

    metrics = dict(
        loss=policy_loss,
        controller=dict(
            types.nt_to_nest(distances),
        )
    )

    rewards = frames.reward

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

    metrics['value'] = {
        'reward': tf_utils.get_stats(rewards),
        'loss': value_loss,
        'return': value_targets,
        'uev': uev,  # unexplained variance
    }

    value_outputs = ValueOutputs(
        returns=value_targets,
        advantages=advantages,
        loss=value_loss,
        metrics=metrics['value'],
    )

    return UnrollOutputs(
        log_probs=log_probs,
        distances=distance_outputs,
        value_outputs=value_outputs,
        final_state=final_state,
        metrics=metrics)

  def imitation_loss(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
      discount: float = 0.99,
      value_cost: float = 0.5,
  ) -> tp.Tuple[tf.Tensor, RecurrentState, dict]:
    # Let's say that delay is D and total unroll-length is U + D + 1 (overlap
    # is D + 1). Then the first trajectory has game states [0, U + D] and the
    # second trajectory has game states [U, 2U + D]. That means that we want to
    # use states [0, U-1] to predict actions [D + 1, U + D] (with previous
    # actions being [D, U + D - 1]). The final hidden state should be the one
    # preceding timestep U, meaning we compute it from game states [0, U-1]. We
    # will use game state U to bootstrap the value function.

    state_action = frames.state_action
    # Includes "overlap" frame.
    unroll_length = state_action.state.stage.shape[0] - self.delay

    frames = data.Frames(
        state_action=embed.StateAction(
            state=tf.nest.map_structure(
                lambda t: t[:unroll_length], state_action.state),
            action=tf.nest.map_structure(
                lambda t: t[self.delay:], state_action.action),
            name=state_action.name[self.delay:],
        ),
        # Only use rewards that follow actions.
        reward=frames.reward[self.delay:],
    )

    unroll_outputs = self.unroll(
        frames, initial_state,
        discount=discount,
    )

    metrics = unroll_outputs.metrics

    total_loss = -tf.reduce_mean(unroll_outputs.log_probs)
    if self.train_value_head:
      value_loss = tf.reduce_mean(unroll_outputs.value_outputs.loss)
      total_loss += value_cost * value_loss

    metrics.update(
        total_loss=total_loss,
    )

    return total_loss, unroll_outputs.final_state, metrics

  def sample(
      self,
      state_action: embed.StateAction,
      initial_state: RecurrentState,
      **kwargs,
  ) -> tp.Tuple[SampleOutputs, RecurrentState]:
    input = self.embed_state_action(state_action)
    output, final_state = self.network.step(input, initial_state)

    prev_action = state_action.action
    next_action = self.controller_head.sample(
        output, prev_action, **kwargs)
    return next_action, final_state

  def multi_sample(
      self,
      states: list[embed.Game],  # time-indexed
      prev_action: embed.Action,  # only for first step
      name_code: int,
      initial_state: RecurrentState,
      **kwargs,
  ) -> Tuple[list[SampleOutputs], RecurrentState]:
    actions = []
    hidden_state = initial_state
    for game in range(states):
      state_action = embed.StateAction(
          state=game,
          action=prev_action,
          name=name_code,
      )
      next_action, hidden_state = self.sample(
          state_action, hidden_state, **kwargs)
      actions.append(next_action)
      prev_action = next_action

    return actions, hidden_state

@dataclasses.dataclass
class PolicyConfig:
  train_value_head: bool = True
  delay: int = 0
