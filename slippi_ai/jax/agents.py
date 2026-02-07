import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from slippi_ai import utils, data, agents
from slippi_ai.types import Game
from slippi_ai.data import StateAction
from slippi_ai.controller_heads import SampleOutputs, ControllerType
from slippi_ai.jax import policies


class BasicAgent(agents.BasicAgent[ControllerType, policies.RecurrentState]):
  """Wraps a Policy to track hidden state."""

  def __init__(
      self,
      policy: policies.Policy[ControllerType],
      batch_size: int,
      name_code: tp.Union[int, tp.Sequence[int]],
      rngs: tp.Optional[nnx.Rngs] = None,
      seed: int = 0,
      sample_kwargs: dict = {},
      compile: bool = True,
      run_on_cpu: bool = False,
  ):
    self._policy = policy
    self._batch_size = batch_size
    self.set_name_code(name_code)
    self._compile = compile

    if run_on_cpu:
      raise NotImplementedError('run_on_cpu is not supported in JAX agents.')

    # The controller_head may discretize certain components of the action.
    # Agents only work with the discretized action space; you will need
    # to call `decode` on the action before sending it to Dolphin.
    default_controller = self._policy.controller_head.dummy_controller([batch_size])
    self._prev_controller = default_controller

    self._rngs = nnx.Rngs(seed) if rngs is None else rngs

    def sample(
        policy: policies.Policy[ControllerType],
        rngs: nnx.Rngs,
        state_action: StateAction[ControllerType],
        needs_reset: jax.Array,
        prev_state: policies.RecurrentState,
    ) -> tuple[SampleOutputs[ControllerType], policies.RecurrentState]:
      # Note: sample outputs are discretized by the controller_head.
      return policy.sample(
          rngs, state_action, prev_state, needs_reset, **sample_kwargs)

    self._sample = nnx.cached_partial(sample, policy, self._rngs)
    self._jitted_sample = nnx.cached_partial(
        nnx.jit(sample, donate_argnums=(1, 4)),
        policy, self._rngs,
    )

    def multi_sample(
        policy: policies.Policy[ControllerType],
        rngs: nnx.Rngs,
        states_and_resets: list[tuple[Game, jax.Array]],  # time-indexed
        name_code: jax.Array,
        prev_action: ControllerType,  # only for first step
        initial_state: policies.RecurrentState,
    ) -> tuple[list[SampleOutputs[ControllerType]], policies.RecurrentState]:

      stacked_states_and_resets = jax.tree.map(
          lambda *xs: jnp.stack(xs, axis=0), *states_and_resets)

      @nnx.scan(in_axes=(0, 0, nnx.Carry), out_axes=(0, nnx.Carry))
      def scan_fn(
          rngs: nnx.Rngs,
          state_and_reset: tuple[Game, jax.Array],
          prev_action_and_state: tuple[ControllerType, policies.RecurrentState],
      ) -> tuple[SampleOutputs[ControllerType], tuple[ControllerType, policies.RecurrentState]]:
        gamestate, needs_reset = state_and_reset
        prev_action, prev_state = prev_action_and_state
        state_action = StateAction(
            state=gamestate,
            action=prev_action,
            name=name_code,
        )
        sample_outputs, new_state = sample(
            policy, rngs, state_action, needs_reset, prev_state)
        return sample_outputs, (sample_outputs.controller_state, new_state)

      length = len(states_and_resets)

      stacked_sample_outputs, (_, final_state) = scan_fn(
          rngs.fork(split=length), stacked_states_and_resets, (prev_action, initial_state))

      sample_outputs = [
          jax.tree.map(lambda t, i=i: t[i], stacked_sample_outputs)
          for i in range(length)]

      return sample_outputs, final_state

    self._multi_sample = nnx.cached_partial(multi_sample, policy, self._rngs)
    self._jitted_multi_sample = nnx.cached_partial(
        nnx.jit(multi_sample, donate_argnums=(1, 5)),
        policy, self._rngs,
    )

    self._hidden_state = self._policy.initial_state(batch_size, self._rngs)

  def hidden_state(self) -> policies.RecurrentState:
    """Returns the current hidden state."""
    return self._hidden_state

  def set_name_code(self, name_code: tp.Union[int, tp.Sequence[int]]):
    if isinstance(name_code, int):
      name_code = [name_code] * self._batch_size
    elif len(name_code) != self._batch_size:
      raise ValueError(f'name_code list must have length batch_size={self._batch_size}')
    self._name_code = np.array(name_code, dtype=data.NAME_DTYPE)

  @property
  def name_code(self) -> np.ndarray[tuple[int], np.dtype[data.NAME_DTYPE]]:
    return self._name_code

  def warmup(self):
    """Warm up the agent by running a dummy step."""
    game = self._policy.network.dummy((self._batch_size,)).state
    needs_reset = np.full([self._batch_size], False)
    self.step(game, needs_reset)

  def step(
      self,
      game: Game,
      needs_reset: agents.BoolArray,
  ) -> SampleOutputs[ControllerType]:
    """Doesn't take into account delay."""
    state_action = StateAction(
        state=self._policy.network.encode_game(game),
        action=self._prev_controller,
        name=self._name_code,
    )

    # Keep hidden state and prev_controller on device.
    sample_fn = self._jitted_sample if self._compile else self._sample
    sample_outputs, self._hidden_state = sample_fn(
        state_action, needs_reset, self._hidden_state)

    # Use donate_argnums?
    self._prev_controller = sample_outputs.controller_state

    # Convert to numpy?
    return sample_outputs

  def multi_step(
      self,
      states: list[tuple[Game, agents.BoolArray]],
  ) -> list[SampleOutputs[ControllerType]]:
    states_and_resets = [
        (self._policy.network.encode_game(game), needs_reset)
        for game, needs_reset in states
    ]

    # Keep hidden state and _prev_controller on device.
    multi_sample_fn = self._jitted_multi_sample if self._compile else self._multi_sample
    sample_outputs, self._hidden_state = multi_sample_fn(
        states_and_resets, self._name_code, self._prev_controller, self._hidden_state)

    self._prev_controller = sample_outputs[-1].controller_state

    return sample_outputs

  def step_unbatched(
      self,
      game: Game,
      needs_reset: bool,
  ) -> SampleOutputs:
    assert self._batch_size == 1
    batched_game = utils.map_single_structure(
        lambda x: np.expand_dims(x, 0), game)
    batched_needs_reset = np.expand_dims(needs_reset, 0)
    batched_action = self.step(batched_game, batched_needs_reset)
    return utils.map_single_structure(lambda x: jnp.squeeze(x, 0), batched_action)
