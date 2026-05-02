import collections
import enum
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from slippi_ai import utils, data, agents
from slippi_ai.types import Game, S, BoolArray, Rank1
from slippi_ai.data import StateAction
from slippi_ai.policies import Platform
from slippi_ai.controller_heads import SampleOutputs, ControllerType
from slippi_ai.jax import policies, jax_utils

class DType(enum.Enum):
  FP32 = 'float32'
  FP16 = 'float16'
  BF16 = 'bfloat16'

  @property
  def dtype(self) -> jnp.dtype:
    return getattr(jnp, self.value)

  def is_default(self):
    return self is DType.FP32

class BasicAgent(agents.BasicAgent[ControllerType, policies.RecurrentState]):
  """Wraps a Policy to track hidden state."""

  @property
  def platform(self) -> Platform:
    return Platform.JAX

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
      pack_args: bool = False,
      functionalize: bool = False,
      dtype: str | DType = DType.FP32,
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
    self._prev_controller = [default_controller] * policy.frame_skip

    if rngs is None:
      rngs = nnx.Rngs(seed)

    self._sample_outputs = collections.deque[SampleOutputs[ControllerType]]()
    self._needs_reset: BoolArray[Rank1] = np.full([batch_size], False)

    def sample(
        policy: policies.Policy[ControllerType],
        /,
        rngs: nnx.Rngs,
        state_and_reset: tuple[Game, jax.Array],
        name_code: jax.Array,
        prev_actions: list[ControllerType],
        prev_state: policies.RecurrentState,
    ) -> tuple[list[SampleOutputs[ControllerType]], policies.RecurrentState]:
      # Note: sample outputs are discretized by the controller_head.
      game, needs_reset = state_and_reset
      state_action = StateAction(
          state=game,
          action=prev_actions,
          name=name_code,
      )
      return policy.sample(
          rngs, state_action, prev_state, needs_reset, **sample_kwargs)

    if isinstance(dtype, str):
      dtype = DType(dtype)
    self._dtype = dtype

    if dtype is not DType.FP32:
      sample = jax_utils.with_compute_dtype(sample, dtype.dtype)

    self._sample = jax_utils.cached_partial(sample, policy, rngs)

    if functionalize:
      self._jitted_sample = jax_utils.cached_functional_jit(
          sample, policy, rngs, donate_argnums=(5,))
    else:
      if pack_args:
        jitted_sample = jax_utils.packed_nnx_jit(
            sample, donate_argnums=(1, 5), pack_argnums=(2,))
      else:
        jitted_sample = jax_utils.nnx_jit(sample, donate_argnums=(1, 5))

      self._jitted_sample = jax_utils.cached_partial(
          jitted_sample, policy, rngs)

    def multi_sample(
        policy: policies.Policy[ControllerType],
        /,
        rngs: nnx.Rngs,
        states_and_resets: list[tuple[Game[S], jax.Array]],  # time-indexed
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
      ) -> tuple[list[SampleOutputs[ControllerType]], tuple[ControllerType, policies.RecurrentState]]:
        gamestate, needs_reset = state_and_reset
        prev_action, prev_state = prev_action_and_state
        sample_outputs, new_state = sample(
            policy, rngs, (gamestate, needs_reset), name_code, prev_action, prev_state)
        return sample_outputs, (sample_outputs[-1].controller_state, new_state)

      length = len(states_and_resets)

      stacked_sample_outputs, (_, final_state) = scan_fn(
          rngs.fork(split=length), stacked_states_and_resets, (prev_action, initial_state))

      sample_outputs: list[SampleOutputs[ControllerType]] = []
      for i in range(length):
        sample_outputs.extend(utils.map_nt(lambda t, i=i: t[i], stacked_sample_outputs))

      return sample_outputs, final_state

    if dtype is not DType.FP32:
      multi_sample = jax_utils.with_compute_dtype(multi_sample, dtype.dtype)

    self._multi_sample = jax_utils.cached_partial(multi_sample, policy, rngs)

    if functionalize:
      self._jitted_multi_sample = jax_utils.cached_functional_jit(
          multi_sample, policy, rngs, donate_argnums=(5,))
    else:
      if pack_args:
        jitted_multi_sample = jax_utils.packed_nnx_jit(
            multi_sample, donate_argnums=(1, 5), pack_argnums=(2,))
      else:
        jitted_multi_sample = jax_utils.nnx_jit(multi_sample, donate_argnums=(1, 5))

      self._jitted_multi_sample = jax_utils.cached_partial(
          jitted_multi_sample, policy, rngs)

    self._hidden_state = self._policy.initial_state(batch_size, rngs)
    if dtype is not DType.FP32:
      self._hidden_state = jax_utils.cast_floats_to_dtype(
          self._hidden_state, dtype.dtype)

  def hidden_state(self) -> policies.RecurrentState:
    """Returns the current hidden state."""
    # Return a copy for buffer donation.
    return jax.tree.map(jnp.copy, self._hidden_state)

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

  def dummy_sample_outputs(self, shape: tp.Sequence[int]) -> SampleOutputs[ControllerType]:
    outputs = self._policy.controller_head.dummy_sample_outputs(shape)
    return jax_utils.cast_floats_to_dtype(outputs, self._dtype.dtype)

  def step(
      self,
      game: Game,
      needs_reset: agents.BoolArray,
  ) -> SampleOutputs[ControllerType]:
    """Doesn't take into account delay."""
    sample_outputs = self.step_device(game, needs_reset)
    jax.copy_to_host_async(sample_outputs.controller_state)
    return sample_outputs

  def step_device(
      self,
      game: Game,
      needs_reset: agents.BoolArray,
  ) -> SampleOutputs[ControllerType]:
    """Sample an action and leave the output tree on device."""
    self._needs_reset |= needs_reset

    if not self._sample_outputs:
      game = self._policy.network.encode_game(game)
      # Keep hidden state and prev_controller on device.
      sample_fn = self._jitted_sample if self._compile else self._sample
      sample_outputs, self._hidden_state = sample_fn(
          (game, self._needs_reset), self._name_code,
          self._prev_controller, self._hidden_state)

      # Use donate_argnums?
      self._prev_controller = [so.controller_state for so in sample_outputs]

      sample_outputs = jax.copy_to_host_async(sample_outputs)
      self._sample_outputs.extend(sample_outputs)

      self._needs_reset[:] = False

    return self._sample_outputs.popleft()

  def multi_step(
      self,
      states: list[tuple[Game, agents.BoolArray]],
  ) -> list[SampleOutputs[ControllerType]]:
    raise NotImplementedError('multi_step + frame_skip is not implemented yet.')

    states_and_resets = [
        (self._policy.network.encode_game(game), needs_reset)
        for game, needs_reset in states
    ]
    # Keep hidden state and _prev_controller on device.
    multi_sample_fn = self._jitted_multi_sample if self._compile else self._multi_sample
    sample_outputs, self._hidden_state = multi_sample_fn(
        states_and_resets, self._name_code, self._prev_controller, self._hidden_state)

    for output in sample_outputs:
      jax.copy_to_host_async(output.controller_state)

    self._prev_controller = sample_outputs[-1].controller_state

    return sample_outputs


    sample_outputs = self.multi_step_stacked_device(states)
    sample_outputs = jax.tree.map(np.asarray, sample_outputs)
    sample_outputs = [
        jax.tree.map(lambda t, i=i: t[i], sample_outputs)
        for i in range(len(states))]
    return sample_outputs
