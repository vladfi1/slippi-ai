import functools
import typing as tp

import numpy as np
import tensorflow as tf

from slippi_ai import utils, data, agents
from slippi_ai.types import Game
from slippi_ai.data import Action, StateAction
from slippi_ai.controller_heads import SampleOutputs, ControllerType
from slippi_ai.tf import policies, tf_utils


class BasicAgent(agents.BasicAgent[ControllerType, policies.RecurrentState]):
  """Wraps a Policy to track hidden state."""

  def __init__(
      self,
      policy: policies.Policy[ControllerType],
      batch_size: int,
      name_code: tp.Union[int, tp.Sequence[int]],
      sample_kwargs: dict = {},
      compile: bool = True,
      jit_compile: bool = False,
      run_on_cpu: bool = False,
  ):
    self._policy = policy
    self._batch_size = batch_size
    self.set_name_code(name_code)

    # The controller_head may discretize certain components of the action.
    # Agents only work with the discretized action space; you will need
    # to call `decode` on the action before sending it to Dolphin.
    default_controller = self._policy.controller_head.dummy_controller([batch_size])
    self._prev_controller = default_controller

    def base_sample(
        state_action: StateAction,
        prev_state: policies.RecurrentState,
        needs_reset: tf.Tensor,
    ) -> tuple[SampleOutputs, policies.RecurrentState]:
      # Note: sample outputs are discretized by the controller_head.
      return policy.sample(
          state_action, prev_state, needs_reset, **sample_kwargs)

    # Note that (compiled) multi_sample doesn't work with set_name_code
    # because the name code is baked into the tensorflow graph.
    # TODO: optimize with packed_compile
    def base_multi_sample(
        states: list[tuple[Game, tf.Tensor]],  # time-indexed
        prev_action: ControllerType,  # only for first step
        initial_state: policies.RecurrentState,
    ) -> tuple[list[SampleOutputs], policies.RecurrentState]:
      actions: list[SampleOutputs] = []
      hidden_state = initial_state
      for game, needs_reset in states:
        state_action = StateAction(
            state=game,
            action=prev_action,
            name=self._name_code,
        )
        next_action, hidden_state = base_sample(
            state_action, hidden_state, needs_reset)
        actions.append(next_action)
        prev_action = next_action.controller_state

      return actions, hidden_state

    sample_1 = base_sample
    multi_sample_1 = base_multi_sample

    if run_on_cpu:
      if jit_compile and tf.config.list_physical_devices('GPU'):
        raise UserWarning("jit compilation may ignore run_on_cpu")
      sample_1 = tf_utils.run_on_cpu(base_sample)
      multi_sample_1 = tf_utils.run_on_cpu(base_multi_sample)

    if compile:
      # compile_fn = tf.function(jit_compile=jit_compile, autograph=False)
      # base_multi_sample = compile_fn(base_multi_sample)

      # Packing significantly speeds up single-step inference, particularly
      # when items are enabled.
      sample_2 = tf_utils.packed_compile(
          sample_1,
          self.sample_signature(),
          jit_compile=jit_compile,
          autograph=False,
      )

      @functools.cache
      def compile_multi_sample(num_steps: int):
        return tf_utils.packed_compile(
            multi_sample_1,
            self.multi_sample_signature(num_steps),
            jit_compile=jit_compile,
            autograph=False,
        )

      def multi_sample_2(
          states: list[tuple[Game, tf.Tensor]],  # time-indexed
          prev_action: ControllerType,  # only for first step
          initial_state: policies.RecurrentState,
      ) -> tuple[list[SampleOutputs], policies.RecurrentState]:
        compiled_fn = compile_multi_sample(len(states))
        return compiled_fn(states, prev_action, initial_state)

    else:
      sample_2 = sample_1
      multi_sample_2 = multi_sample_1

    self._sample = sample_2
    self._multi_sample = multi_sample_2

    self._hidden_state = self._policy.initial_state(batch_size)

  def hidden_state(self) -> policies.RecurrentState:
    """Returns the current hidden state."""
    return self._hidden_state

  def sample_signature(self) -> tf_utils.Signature:
    dummy_state_action = self._policy.network.dummy((self._batch_size,))
    state_action_spec = utils.map_nt(
        lambda x: tf_utils.ArraySpec(
            shape=x.shape,
            dtype=x.dtype,
        ), dummy_state_action)

    # Don't pack action and prev_state as they are already Tensors.
    state_action_spec = state_action_spec._replace(action=None)
    prev_state = None

    needs_reset = tf_utils.ArraySpec(
        shape=(self._batch_size,),
        dtype=np.dtype('bool'),
    )

    return (state_action_spec, prev_state, needs_reset)

  def multi_sample_signature(self, num_steps: int) -> tf_utils.Signature:
    dummy_state_action = self._policy.network.dummy((self._batch_size,))
    state_spec = utils.map_nt(
        lambda x: tf_utils.ArraySpec(
            shape=x.shape,
            dtype=x.dtype,
        ), dummy_state_action.state)

    needs_reset_spec = tf_utils.ArraySpec(
        shape=(self._batch_size,),
        dtype=np.dtype('bool'),
    )

    states_spec = [(state_spec, needs_reset_spec)] * num_steps

    return (states_spec, None, None)

  def set_name_code(self, name_code: tp.Union[int, tp.Sequence[int]]):
    if isinstance(name_code, int):
      name_code = [name_code] * self._batch_size
    elif len(name_code) != self._batch_size:
      raise ValueError(f'name_code list must have length batch_size={self._batch_size}')
    self._name_code = np.array(name_code, dtype=data.NAME_DTYPE)

  def warmup(self):
    """Warm up the agent by running a dummy step."""
    game = self._policy.network.dummy((self._batch_size,)).state
    needs_reset = np.full([self._batch_size], False)
    self.step(game, needs_reset)

  def step(
      self,
      game: Game,
      needs_reset: agents.BoolArray,
  ) -> SampleOutputs:
    """Doesn't take into account delay."""
    state_action = StateAction(
        state=self._policy.network.encode_game(game),
        action=self._prev_controller,
        name=self._name_code,
    )

    # Keep hidden state and prev_controller on device.
    sample_outputs: SampleOutputs
    sample_outputs, self._hidden_state = self._sample(
        state_action, self._hidden_state, needs_reset)
    self._prev_controller = sample_outputs.controller_state

    return utils.map_single_structure(lambda t: t.numpy(), sample_outputs)

  def multi_step(
      self,
      states: list[tuple[Game, agents.BoolArray]],
  ) -> list[SampleOutputs]:
    states = [
        (self._policy.network.encode_game(game), needs_reset)
        for game, needs_reset in states
    ]

    # Keep hidden state and _prev_controller on device.
    sample_outputs: list[SampleOutputs]
    sample_outputs, self._hidden_state = self._multi_sample(
        states, self._prev_controller, self._hidden_state)
    self._prev_controller = sample_outputs[-1].controller_state

    return utils.map_single_structure(lambda t: t.numpy(), sample_outputs)

  def step_unbatched(
      self,
      game: Game,
      needs_reset: bool
  ) -> SampleOutputs:
    assert self._batch_size == 1
    batched_game = utils.map_single_structure(
        lambda x: tf.expand_dims(x, 0), game)
    batched_needs_reset = np.array([needs_reset])
    batched_action = self.step(batched_game, batched_needs_reset)
    return utils.map_single_structure(lambda x: x.item(), batched_action)
