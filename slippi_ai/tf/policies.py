import dataclasses
import typing as tp

import sonnet as snt
import tensorflow as tf

from slippi_ai.agents import BoolArray
from slippi_ai.tf.controller_heads import (
    ControllerHead,
    DistanceOutputs,
    SampleOutputs,
    ControllerType,
)
from slippi_ai.tf.rl_lib import discounted_returns
from slippi_ai import data, types, utils, policies
from slippi_ai.types import S
from slippi_ai.tf import embed, networks, tf_utils
from slippi_ai.tf.value_function import ValueOutputs

Outputs = tf_utils.Outputs
RecurrentState = networks.RecurrentState

Rank2 = tuple[int, int]

class UnrollOutputs(tp.NamedTuple, tp.Generic[ControllerType]):
  log_probs: tf.Tensor  # [T, B]
  distances: DistanceOutputs[ControllerType]  # Struct of [T, B]
  value_outputs: ValueOutputs
  final_state: RecurrentState  # [B]
  metrics: dict  # mixed

class UnrollWithOutputs(tp.NamedTuple, tp.Generic[ControllerType]):
  imitation_loss: tf.Tensor  # [T, B]
  distances: DistanceOutputs[ControllerType]  # Struct of [T, B]
  outputs: tf.Tensor  # [T, B]
  final_state: RecurrentState  # [B]
  metrics: dict  # mixed

class Policy(snt.Module, policies.Policy[ControllerType, RecurrentState]):

  @property
  def platform(self) -> policies.Platform:
    return policies.Platform.TF

  def __init__(
      self,
      network: networks.StateActionNetwork,
      controller_head: ControllerHead[ControllerType],
      train_value_head: bool = True,
      delay: int = 0,
  ):
    super().__init__(name='Policy')
    self.network = network
    self._controller_head = controller_head

    self.initial_state = self.network.initial_state
    self.train_value_head = train_value_head
    self._delay = delay

    self.value_head = snt.Linear(1, name='value_head')
    if not train_value_head:
      self.value_head = snt.Sequential([tf.stop_gradient, self.value_head])

  @property
  def delay(self) -> int:
    return self._delay

  @property
  def controller_head(self):
    return self._controller_head

  def encode_game(self, game: data.Game) -> data.Game:
    return self.network.encode_game(game)

  def initial_state(self, batch_size: int):
    return self.network.initial_state(batch_size)

  def initialize_variables(self):
    T = 2 + self.delay
    B = 1
    dummy_state_action = self.network.dummy((T, B))
    dummy_reward = tf.zeros([T-1, B], tf.float32)
    is_resetting = tf.fill([T, B], False)
    dummy_frames = data.Frames(dummy_state_action, is_resetting, dummy_reward)
    initial_state = self.initial_state(B)

    # imitation_loss also initializes value function
    self.imitation_loss(dummy_frames, initial_state)

  def _value_outputs(
      self,
      outputs,  # t = [0, T-1]
      last_input: data.StateAction,  # t = T
      is_resetting: tf.Tensor,  # t = [0, T]
      final_state: RecurrentState,  # t = T - 1
      rewards: tf.Tensor,  # t = [0, T-1]
      discount: float,
  ) -> ValueOutputs:
    values = tf.squeeze(self.value_head(outputs), -1)
    last_output, _ = self.network.step_with_reset(
        last_input, is_resetting[-1], final_state)
    last_value = tf.squeeze(self.value_head(last_output), -1)

    discounts = tf.where(is_resetting[1:], 0.0, tf.cast(discount, tf.float32))
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
      frames: data.Frames[Rank2, ControllerType],
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
    inputs = utils.map_nt(lambda t: t[:-1], frames.state_action)
    last_input = utils.map_nt(lambda t: t[-1], frames.state_action)
    outputs, final_state = self.network.unroll(
        inputs, frames.is_resetting[:-1], initial_state)

    # Predict next action.
    action = frames.state_action.action
    prev_action = tf.nest.map_structure(lambda t: t[:-1], action)
    next_action = tf.nest.map_structure(lambda t: t[1:], action)

    distance_outputs = self._controller_head.distance(
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
      frames: data.Frames[Rank2, ControllerType],
      initial_state: RecurrentState,
      discount: float = 0.99,
      value_cost: float = 0.5,
  ) -> tuple[tf.Tensor, RecurrentState, dict]:
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
        is_resetting=frames.is_resetting[:unroll_length],
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

  def unroll_with_outputs(
      self,
      frames: data.Frames[Rank2, ControllerType],
      initial_state: RecurrentState,
      discount: float = 0.99,
  ):
    inputs = utils.map_nt(lambda t: t[:-1], frames.state_action)
    last_input = utils.map_nt(lambda t: t[-1], frames.state_action)
    outputs, final_state = self.network.unroll(
        inputs, frames.is_resetting[:-1], initial_state)

    # Predict next action.
    action = frames.state_action.action
    prev_action = tf.nest.map_structure(lambda t: t[:-1], action)
    next_action = tf.nest.map_structure(lambda t: t[1:], action)

    distance_outputs = self._controller_head.distance(
        outputs, prev_action, next_action)
    distances = distance_outputs.distance
    policy_loss = tf.add_n(tf.nest.flatten(distances))

    metrics = dict(
        loss=policy_loss,
        controller=dict(
            types.nt_to_nest(distances),
        )
    )

    # We're only really doing this to initialize the value_head...
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
      state_action: data.StateAction[S, ControllerType],
      initial_state: RecurrentState,
      is_resetting: tp.Optional[BoolArray] = None,
      **kwargs,
  ) -> tuple[SampleOutputs[ControllerType], RecurrentState]:
    if is_resetting is None:
      batch_size = state_action.state.stage.shape[0]
      is_resetting = tf.fill([batch_size], False)

    output, final_state = self.network.step_with_reset(
        state_action, is_resetting, initial_state)

    prev_action = state_action.action
    next_action = self._controller_head.sample(
        output, prev_action, **kwargs)
    return next_action, final_state

  def multi_sample(
      self,
      states: list[embed.Game],  # time-indexed
      prev_action: ControllerType,  # only for first step
      name_code: data.NAME_DTYPE,
      initial_state: RecurrentState,
      **kwargs,
  ) -> tuple[list[SampleOutputs[ControllerType]], RecurrentState]:
    actions = []
    hidden_state = initial_state
    for game in states:
      state_action = embed.StateAction(
          state=game,
          action=prev_action,
          name=name_code,
      )
      next_action, hidden_state = self.sample(
          state_action, hidden_state, **kwargs)
      actions.append(next_action)
      prev_action = next_action.controller_state

    return actions, hidden_state

  def build_agent(self, batch_size: int, **kwargs):
    from slippi_ai.tf import agents  # avoid circular import
    return agents.BasicAgent(
        policy=self,
        batch_size=batch_size,
        **kwargs,
    )

  def get_state(self):
    return self.variables

  def set_state(self, state):
    for var, val in zip(self.variables, state):
      var.assign(val)


@dataclasses.dataclass
class PolicyConfig:
  train_value_head: bool = True
  delay: int = 0
