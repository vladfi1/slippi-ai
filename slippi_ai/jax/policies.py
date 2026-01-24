import dataclasses
from typing import Optional, Tuple
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx

from slippi_ai.jax.controller_heads import (
    ControllerHead,
    DistanceOutputs,
    SampleOutputs,
)
from slippi_ai.jax.value_function import ValueOutputs
from slippi_ai.jax import embed, networks, jax_utils, rl_lib
from slippi_ai import data, types, utils

Array = jax.Array
RecurrentState = networks.RecurrentState


class UnrollOutputs(tp.NamedTuple):
  log_probs: Array  # [T, B]
  distances: DistanceOutputs  # Struct of [T, B]
  value_outputs: ValueOutputs
  final_state: RecurrentState  # [B]
  metrics: dict  # mixed


class UnrollWithOutputs(tp.NamedTuple):
  imitation_loss: Array  # [T, B]
  distances: DistanceOutputs  # Struct of [T, B]
  outputs: Array  # [T, B]
  final_state: RecurrentState  # [B]
  metrics: dict  # mixed


class Policy(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      network: networks.StateActionNetwork,
      controller_head: ControllerHead,
      hidden_size: int,
      train_value_head: bool = True,
      delay: int = 0,
  ):
    self.network = network
    self.controller_head = controller_head

    self.train_value_head = train_value_head
    self.delay = delay

    self.value_head = nnx.Linear(hidden_size, 1, rngs=rngs)

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return self.network.initial_state(batch_size, rngs)

  @property
  def controller_embedding(self) -> embed.Embedding[embed.Controller, data.Action]:
    return self.controller_head.controller_embedding()

  def _value_outputs(
      self,
      outputs: Array,  # t = [0, T-1]
      last_input: data.StateAction,  # t = T
      is_resetting: Array,  # t = [0, T]
      final_state: RecurrentState,  # t = T - 1
      rewards: Array,  # t = [0, T-1]
      discount: float,
  ) -> ValueOutputs:
    values = jnp.squeeze(self.value_head(outputs), -1)
    last_output, _ = self.network.step_with_reset(
        last_input, is_resetting[-1], final_state)
    last_value = jnp.squeeze(self.value_head(last_output), -1)

    discounts = jnp.where(is_resetting[1:], 0.0, discount)
    value_targets = rl_lib.discounted_returns(
        rewards=rewards,
        discounts=discounts,
        bootstrap=last_value)
    value_targets = jax.lax.stop_gradient(value_targets)
    advantages = value_targets - values
    value_loss = jnp.square(advantages)

    _, value_variance = jax_utils.mean_and_variance(value_targets)
    uev = value_loss / (value_variance + 1e-8)

    metrics = {
        'reward': jax_utils.get_stats(rewards),
        'loss': value_loss,
        'return': value_targets,
        'uev': uev,  # unexplained variance
    }

    return ValueOutputs(
        returns=value_targets,
        advantages=advantages,
        loss=value_loss,
        metrics=metrics,
    )

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
      discount: Per-frame discount factor for returns.
    """
    inputs = utils.map_nt(lambda t: t[:-1], frames.state_action)
    last_input = utils.map_nt(lambda t: t[-1], frames.state_action)
    outputs, final_state = self.network.unroll(
        inputs, frames.is_resetting[:-1], initial_state)

    # Predict next action.
    action = frames.state_action.action
    prev_action = jax.tree.map(lambda t: t[:-1], action)
    next_action = jax.tree.map(lambda t: t[1:], action)

    distance_outputs = self.controller_head.distance(
        outputs, prev_action, next_action)
    distances = distance_outputs.distance
    policy_loss = sum(jax.tree.leaves(distances))
    log_probs = -policy_loss

    metrics = dict(
        loss=policy_loss,
        controller=dict(
            types.nt_to_nest(distances),
        )
    )

    value_outputs = self._value_outputs(
        outputs, last_input, frames.is_resetting, final_state,
        frames.reward, discount)
    metrics['value'] = value_outputs.metrics

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
  ) -> tp.Tuple[Array, RecurrentState, dict]:
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
        state_action=data.StateAction(
            state=jax.tree.map(
                lambda t: t[:unroll_length], state_action.state),
            action=jax.tree.map(
                lambda t: t[self.delay:], state_action.action),
            name=state_action.name[self.delay:],
        ),
        is_resetting=frames.is_resetting[:unroll_length],
        # Only use rewards that follow actions.
        reward=frames.reward[self.delay:],
    )

    unroll_outputs = self.unroll(
        frames, initial_state,
        discount=discount,
    )

    metrics = unroll_outputs.metrics

    total_loss = -jnp.mean(unroll_outputs.log_probs)
    if self.train_value_head:
      value_loss = jnp.mean(unroll_outputs.value_outputs.loss)
      total_loss = total_loss + value_cost * value_loss

    metrics.update(
        total_loss=total_loss,
    )

    return total_loss, unroll_outputs.final_state, metrics

  def unroll_with_outputs(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
      discount: float = 0.99,
  ) -> UnrollWithOutputs:
    inputs = utils.map_nt(lambda t: t[:-1], frames.state_action)
    last_input = utils.map_nt(lambda t: t[-1], frames.state_action)
    outputs, final_state = self.network.unroll(
        inputs, frames.is_resetting[:-1], initial_state)

    # Predict next action.
    action = frames.state_action.action
    prev_action = jax.tree.map(lambda t: t[:-1], action)
    next_action = jax.tree.map(lambda t: t[1:], action)

    distance_outputs = self.controller_head.distance(
        outputs, prev_action, next_action)
    distances = distance_outputs.distance
    policy_loss = sum(jax.tree.leaves(distances))

    metrics = dict(
        loss=policy_loss,
        controller=dict(
            types.nt_to_nest(distances),
        )
    )

    value_outputs = self._value_outputs(
        outputs, last_input, frames.is_resetting, final_state,
        frames.reward, discount)
    metrics['value'] = value_outputs.metrics

    return UnrollWithOutputs(
        imitation_loss=policy_loss,
        distances=distances,
        outputs=outputs,
        final_state=final_state,
        metrics=metrics,
    )

  def sample(
      self,
      rngs: nnx.Rngs,
      state_action: data.StateAction,
      initial_state: RecurrentState,
      is_resetting: Optional[Array] = None,
      **kwargs,
  ) -> tp.Tuple[SampleOutputs, RecurrentState]:
    if is_resetting is None:
      batch_size = state_action.state.stage.shape[0]
      is_resetting = jnp.zeros([batch_size], dtype=jnp.bool_)

    output, final_state = self.network.step_with_reset(
        state_action, is_resetting, initial_state)

    prev_action = state_action.action
    next_action = self.controller_head.sample(
        rngs, output, prev_action, **kwargs)
    return next_action, final_state

  def multi_sample(
      self,
      rngs: nnx.Rngs,
      states: list[embed.Game],  # time-indexed
      prev_action: data.Action,  # only for first step
      name_code: int,
      initial_state: RecurrentState,
      **kwargs,
  ) -> Tuple[list[SampleOutputs], RecurrentState]:
    # TODO: use scan?
    actions = []
    hidden_state = initial_state
    for game in states:
      state_action = data.StateAction(
          state=game,
          action=prev_action,
          name=name_code,
      )
      next_action, hidden_state = self.sample(
          rngs, state_action, hidden_state, **kwargs)
      actions.append(next_action)
      prev_action = next_action.controller_state

    return actions, hidden_state


@dataclasses.dataclass
class PolicyConfig:
  train_value_head: bool = True
  delay: int = 0
