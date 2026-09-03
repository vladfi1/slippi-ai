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
    AutoRegressive,
    ResidualAutoRegressive,
)
from slippi_ai.jax import embed, networks, jax_utils
from slippi_ai import data, types, utils, policies
from slippi_ai.types import S, Game, Frames

Array = jax.Array
RecurrentState = networks.RecurrentState


class UnrollOutputs(tp.NamedTuple, tp.Generic[S, ControllerType]):
  log_probs: list[Array]  # [T, B]
  distances: list[DistanceOutputs[ControllerType]]  # Struct of [T, B]
  final_state: RecurrentState  # [B]
  metrics: dict  # mixed


class UnrollWithOutputs(tp.NamedTuple, tp.Generic[S, ControllerType]):
  imitation_loss: Array  # [T, B]
  distances: list[DistanceOutputs[ControllerType]]  # Struct of [T, B]
  outputs: Array  # [T, B]
  final_state: RecurrentState  # [B]
  metrics: dict  # mixed

Rank2 = tuple[int, int]

class Policy(nnx.Module, policies.Policy[ControllerType, RecurrentState]):

  @property
  def platform(self) -> policies.Platform:
    return policies.Platform.JAX

  def __init__(
      self,
      network: networks.StateActionNetwork[ControllerType],
      controller_head: ControllerHead[ControllerType],
      delay: int = 0,
      frame_skip: int = 1,
  ):
    self.network = network
    self._controller_head = controller_head

    self._delay = delay
    self._frame_skip = frame_skip

    if delay % frame_skip != 0:
      raise ValueError(f"Delay {delay} must be divisible by frame_skip {frame_skip}.")
    self._skip_delay = delay // frame_skip

  @property
  def delay(self) -> int:
    return self._delay

  @property
  def frame_skip(self) -> int:
    return self._frame_skip

  @property
  def controller_head(self) -> ControllerHead[ControllerType]:
    return self._controller_head

  def enable_remat(self):
    """Save memory at the cost of extra compute."""

    # TODO: do this in a less hacky way

    if isinstance(self.controller_head, (AutoRegressive, ResidualAutoRegressive)):
      self.controller_head.remat = True

    self.network.enable_remat()

  def encode_game(self, game: Game) -> Game:
    return self.network.encode_game(game)

  def initial_state(self, batch_size: networks.Shape, rngs: tp.Optional[nnx.Rngs] = None) -> RecurrentState:
    if rngs is None:
      rngs = nnx.Rngs(0)
    return self.network.initial_state(batch_size, rngs)

  def unroll(
      self,
      frames: Frames[S, ControllerType],
      initial_state: RecurrentState,
  ) -> UnrollOutputs[S, ControllerType]:
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
    prev_action = utils.map_nt(lambda t: t[:-1], action)
    next_action = utils.map_nt(lambda t: t[1:], action)

    distance_outputs = self._controller_head.distance_outputs(
        outputs, prev_action, next_action)
    losses = [
        jax_utils.add_n(jax.tree.leaves(do.distance))
        for do in distance_outputs]
    policy_loss = jax_utils.add_n(losses) / len(losses)
    log_probs = [-loss for loss in losses]

    metrics = dict(
        loss=policy_loss,
        controller={
            i: types.nt_to_nest(do.distance)
            for i, do in enumerate(distance_outputs)
        },
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
    unroll_length = frames.is_resetting.shape[0] - self._skip_delay

    frames = data.Frames(
        state_action=data.StateAction(
            state=jax.tree.map(
                lambda t: t[:unroll_length], state_action.state),
            action=jax.tree.map(
                lambda t: t[self._skip_delay:], state_action.action),
            name=state_action.name[self._skip_delay:],
            rating=state_action.rating[self._skip_delay:],
        ),
        is_resetting=frames.is_resetting[:unroll_length],
        # Only use rewards that follow actions.
        reward=frames.reward[self._skip_delay:],
    )

    unroll_outputs = self.unroll(frames, initial_state)
    metrics = unroll_outputs.metrics

    # Take time-mean
    # metrics = jax.tree.map(lambda t: jnp.mean(t, axis=0), metrics)

    loss = metrics['loss']

    # All metrics and loss should have shape [T, B]
    return loss, metrics, unroll_outputs.final_state

  def unroll_with_outputs(
      self,
      frames: data.Frames[S, ControllerType],
      initial_state: RecurrentState,
  ) -> UnrollWithOutputs[S, ControllerType]:
    inputs = utils.map_nt(lambda t: t[:-1], frames.state_action)
    outputs, final_state = self.network.unroll(
        inputs, frames.is_resetting[:-1], initial_state)

    # Predict next action.
    action = frames.state_action.action
    prev_action = utils.map_nt(lambda t: t[:-1], action)
    next_action = utils.map_nt(lambda t: t[1:], action)

    distance_outputs = self._controller_head.distance_outputs(
        outputs, prev_action, next_action)
    losses = [
        jax_utils.add_n(jax.tree.leaves(do.distance))
        for do in distance_outputs]
    policy_loss = jax_utils.add_n(losses) / len(losses)

    metrics = dict(
        loss=policy_loss,
        controller={
            i: types.nt_to_nest(do.distance)
            for i, do in enumerate(distance_outputs)
        },
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
  ) -> tp.Tuple[list[SampleOutputs[ControllerType]], RecurrentState]:
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
      states: list[embed.Game[S]],  # time-indexed
      prev_action: ControllerType,  # only for first step
      name_code: int,
      initial_state: RecurrentState,
      **kwargs,
  ) -> Tuple[list[SampleOutputs[ControllerType]], RecurrentState]:
    # TODO: use scan?
    actions: list[SampleOutputs[ControllerType]] = []
    hidden_state = initial_state
    for game in states:
      state_action = data.StateAction[S, ControllerType](
          state=game,
          action=prev_action,
          name=name_code,
      )
      next_action, hidden_state = self.sample(
          rngs, state_action, hidden_state, **kwargs)
      actions.extend(next_action)
      prev_action = next_action[-1].controller_state

    return actions, hidden_state

  def build_agent(self, batch_size: int, **kwargs):
    from slippi_ai.jax import agents  # avoid circular import
    return agents.BasicAgent(self, batch_size, **kwargs)

  def get_state(self, to_numpy: bool = True):
    return jax_utils.get_module_state(self, to_numpy=to_numpy)

  def set_state(self, state):
    jax_utils.set_module_state(self, state)

@dataclasses.dataclass
class PolicyConfig:
  delay: int = 0
  frame_skip: int = 1  # 1 means no frame skip
