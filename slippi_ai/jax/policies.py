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
    ControllerType,
)
from slippi_ai.jax import embed, networks, jax_utils
from slippi_ai import data, types, utils, policies
from slippi_ai.types import S

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

Rank1 = tuple[int]
Rank2 = tuple[int, int]

class Policy(nnx.Module, policies.Policy[ControllerType, RecurrentState]):

  @property
  def platform(self) -> policies.Platform:
    return policies.Platform.JAX

  def __init__(
      self,
      network: networks.StateActionNetwork,
      controller_head: ControllerHead[ControllerType],
      delay: int = 0,
  ):
    self.network = network
    self._controller_head = controller_head

    self._delay = delay

  @property
  def delay(self) -> int:
    return self._delay

  @property
  def controller_head(self) -> ControllerHead[ControllerType]:
    return self._controller_head

  def encode_game(self, game: data.Game) -> data.Game:
    return self.network.encode_game(game)

  def initial_state(self, batch_size: int, rngs: tp.Optional[nnx.Rngs] = None) -> RecurrentState:
    if rngs is None:
      rngs = nnx.Rngs(0)
    return self.network.initial_state(batch_size, rngs)

  def unroll(
      self,
      frames: data.Frames[Rank2, ControllerType],
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

    distance_outputs = self._controller_head.distance(
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
      frames: data.Frames[Rank2, ControllerType],
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
    unroll_length = state_action.state.stage.shape[0] - self._delay

    frames = data.Frames(
        state_action=data.StateAction(
            state=jax.tree.map(
                lambda t: t[:unroll_length], state_action.state),
            action=jax.tree.map(
                lambda t: t[self._delay:], state_action.action),
            name=state_action.name[self._delay:],
        ),
        is_resetting=frames.is_resetting[:unroll_length],
        # Only use rewards that follow actions.
        reward=frames.reward[self._delay:],
    )

    unroll_outputs = self.unroll(frames, initial_state)
    metrics = unroll_outputs.metrics

    # Take time-mean
    # metrics = jax.tree.map(lambda t: jnp.mean(t, axis=0), metrics)

    loss = -unroll_outputs.log_probs

    # All metrics and loss should have shape [T, B]
    return loss, metrics, unroll_outputs.final_state

  def unroll_with_outputs(
      self,
      frames: data.Frames[Rank2, ControllerType],
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

    distance_outputs = self._controller_head.distance(
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
      state_action: data.StateAction[S, ControllerType],
      initial_state: RecurrentState,
      is_resetting: Optional[Array] = None,
      **kwargs,
  ) -> tp.Tuple[SampleOutputs[ControllerType], RecurrentState]:
    if is_resetting is None:
      batch_size = state_action.state.stage.shape[0]
      is_resetting = jnp.zeros([batch_size], dtype=jnp.bool_)

    output, final_state = self.network.step_with_reset(
        state_action, is_resetting, initial_state)

    prev_action = state_action.action
    next_action = self._controller_head.sample(
        rngs, output, prev_action, **kwargs)
    return next_action, final_state

  def multi_sample(
      self,
      rngs: nnx.Rngs,
      states: list[embed.Game],  # time-indexed
      prev_action: ControllerType,  # only for first step
      name_code: int,
      initial_state: RecurrentState,
      **kwargs,
  ) -> Tuple[list[SampleOutputs[ControllerType]], RecurrentState]:
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

  def build_agent(self, batch_size: int, **kwargs):
    from slippi_ai.jax import agents  # avoid circular import
    return agents.BasicAgent(self, batch_size, **kwargs)

  def get_state(self):
    return jax_utils.get_module_state(self)

  def set_state(self, state):
    jax_utils.set_module_state(self, state)

@dataclasses.dataclass
class PolicyConfig:
  delay: int = 0
