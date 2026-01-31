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
from slippi_ai.jax import embed, networks
from slippi_ai import data, types, utils

Array = jax.Array
RecurrentState = networks.RecurrentState


class UnrollOutputs(tp.NamedTuple):
  log_probs: Array  # [T, B]
  distances: DistanceOutputs  # Struct of [T, B]
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
      network: networks.StateActionNetwork,
      controller_head: ControllerHead,
      delay: int = 0,
  ):
    self.network = network
    self.controller_head = controller_head

    self.delay = delay

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return self.network.initial_state(batch_size, rngs)

  @property
  def controller_embedding(self) -> embed.Embedding[embed.Controller, data.Action]:
    return self.controller_head.controller_embedding()

  def unroll(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
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
        controller=types.nt_to_nest(distances),
    )

    return UnrollOutputs(
        log_probs=log_probs,
        distances=distance_outputs,
        final_state=final_state,
        metrics=metrics)

  def imitation_loss(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
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

    unroll_outputs = self.unroll(frames, initial_state)
    metrics = unroll_outputs.metrics

    # Take time-mean
    # metrics = jax.tree.map(lambda t: jnp.mean(t, axis=0), metrics)

    total_loss = -jnp.mean(unroll_outputs.log_probs)

    # All metrics should have shape [T, B]
    return total_loss, metrics, unroll_outputs.final_state

  def unroll_with_outputs(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
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
        controller=types.nt_to_nest(distances),
    )

    return UnrollWithOutputs(
        imitation_loss=policy_loss,
        distances=distance_outputs,
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
  delay: int = 0
