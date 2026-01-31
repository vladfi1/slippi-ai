
import dataclasses
import logging
from typing import Optional
import typing as tp
import types

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from slippi_ai.data import Frames
from slippi_ai.jax.policies import Policy
from slippi_ai.jax.networks import RecurrentState
from slippi_ai.jax import value_function as vf_lib
from slippi_ai.jax import jax_utils

PS = jax.sharding.PartitionSpec
Array = jax.Array


def swap_axes(t, axis1=0, axis2=1):
  """Swap two axes of a tensor."""
  permutation = list(range(len(t.shape)))
  permutation[axis2] = axis1
  permutation[axis1] = axis2
  return jnp.transpose(t, permutation)


@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  reward_halflife: float = 4
  use_shard_map: bool = True

# TODO: move to jax_utils
P = tp.ParamSpec('P')
T = tp.TypeVar('T')

def jit_method(
    method: tp.Callable[P, T],
    *,
    static_argnames: tp.Optional[tp.Iterable[str]] = None,
) -> tp.Callable[P, T]:
  if not isinstance(method, types.MethodType):
    raise TypeError('jit_method can only be applied to methods.')

  jitted = nnx.jit(
      donate_argnums=(0,),
      static_argnames=static_argnames,
  )(method.__func__)

  return nnx.cached_partial(jitted, method.__self__)

def _sharding_callback(sharding: tp.Optional[jax.sharding.Sharding]):
  print(sharding)
  import ipdb; ipdb.set_trace()

Metrics = dict[str, tp.Any]

def _policy_loss_fn(
    policy: Policy, data: tuple[Frames, RecurrentState],
# ) -> tp.Tuple[Metrics, RecurrentState]:
) -> tuple[jax_utils.Loss, tp.Tuple[Metrics, RecurrentState]]:
  frames, initial_states = data
  tm_frames: Frames = jax.tree.map(swap_axes, frames)
  loss, tm_metrics, final_states = policy.imitation_loss(tm_frames, initial_states)
  # metrics = jax.tree.map(lambda t: jnp.mean(t, axis=0), metrics)
  bm_metrics = jax.tree.map(swap_axes, tm_metrics)
  return loss, (bm_metrics, final_states)

def _value_loss_fn(
    value_function: vf_lib.ValueFunction,
    data: tuple[Frames, RecurrentState],
    discount: float,
# ) -> tp.Tuple[Metrics, RecurrentState]:
) -> tuple[jax_utils.Loss, tp.Tuple[Metrics, RecurrentState]]:
  frames, initial_states = data
  tm_frames: Frames = jax.tree.map(swap_axes, frames)
  value_outputs, final_states = value_function.loss(
      tm_frames, initial_states, discount)
  loss = jnp.mean(value_outputs.loss)
  bm_metrics = jax.tree.map(swap_axes, value_outputs.metrics)
  return loss, (bm_metrics, final_states)

class Learner(nnx.Module):

  def __init__(
      self,
      policy: Policy,
      learning_rate: float,
      reward_halflife: float,
      value_function: vf_lib.ValueFunction,
      mesh: jax.sharding.Mesh,
      compile: bool = True,
      use_shard_map: bool = True,
  ):
    self.policy = policy
    self.value_function = value_function
    self.discount = 0.5 ** (1 / (reward_halflife * 60))
    self.compile = compile
    self.use_shard_map = use_shard_map

    # Create optimizers using optax
    self.policy_optimizer = nnx.Optimizer(
        policy, optax.adam(learning_rate), wrt=nnx.Param)

    self.value_optimizer = nnx.Optimizer(
        self.value_function, optax.adam(learning_rate), wrt=nnx.Param)

    self.jit_step = jit_method(self._step, static_argnames=('train', 'compile'))
    self.jit_step_policy = jit_method(
        self._step_policy, static_argnames=('train',))
    self.jit_step_value_function = jit_method(
        self._step_value_function, static_argnames=('train',))

    jax_utils.replicate_module(self, mesh)

    # Train and run functions using shard_map. Empirically these have better
    # performance for the frame_tx network than the above "jit_step" methods,
    # which let XLA handle sharding automatically.
    self.sharded_train_policy = jax_utils.data_parallel_train(
        module=self.policy,
        optimizer=self.policy_optimizer,
        loss_fn=_policy_loss_fn,
        mesh=mesh,
    )

    self.sharded_run_policy = jax_utils.shard_map_loss_fn(
        module=self.policy,
        loss_fn=_policy_loss_fn,
        mesh=mesh,
    )

    self.sharded_train_value_function = jax_utils.data_parallel_train(
        module=self.value_function,
        optimizer=self.value_optimizer,
        loss_fn=_value_loss_fn,
        mesh=mesh,
    )

    self.sharded_run_value_function = jax_utils.shard_map_loss_fn(
        module=self.value_function,
        loss_fn=_value_loss_fn,
        mesh=mesh,
    )

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return (
        self.policy.initial_state(batch_size, rngs),
        self.value_function.initial_state(batch_size, rngs),
    )

  def _step_policy(
      self,
      bm_frames: Frames,
      initial_states: RecurrentState,
      train: bool = True,
  ) -> tuple[Array, Metrics, RecurrentState]:
    """Single training/eval step for policy only."""
    # Switch axes to time-major
    tm_frames: Frames = jax.tree.map(swap_axes, bm_frames)

    # Define loss function that takes policy as argument for gradient computation
    def loss_fn(policy: Policy):
      loss, metrics, final_states = policy.imitation_loss(
          tm_frames, initial_states)
      return loss, (metrics, final_states)

    if train:
      # Compute gradients for policy using value_and_grad to also get loss
      (loss, (metrics, final_states)), grads = nnx.value_and_grad(
          loss_fn, has_aux=True)(self.policy)

      # jax.debug.inspect_array_sharding(final_states, callback=_sharding_callback)
      # jax.debug.inspect_array_sharding(grads, callback=_sharding_callback)

      # Apply gradients
      self.policy_optimizer.update(self.policy, grads)
    else:
      # Just compute forward pass without gradients
      loss, (metrics, final_states) = loss_fn(self.policy)


    return loss, metrics, final_states

  def _step_value_function(
      self,
      bm_frames: Frames,
      initial_states: RecurrentState,
      train: bool = True,
  ) -> tuple[Array, Metrics, RecurrentState]:
    """Single training/eval step for value function only."""
    # Switch axes to time-major
    tm_frames: Frames = jax.tree.map(swap_axes, bm_frames)

    # Define loss function for value function
    def value_loss_fn(value_function: vf_lib.ValueFunction):
      delay = self.policy.delay
      # Value function sees non-delayed actions
      value_frames = jax.tree.map(
          lambda t: t[:t.shape[0] - delay] if delay > 0 else t, tm_frames)
      value_outputs, value_final_states = value_function.loss(
          value_frames, initial_states, self.discount)
      return jnp.mean(value_outputs.loss), (value_outputs.metrics, value_final_states)

    if train:
      # Compute gradients for value function
      (loss, (metrics, final_states)), value_grads = nnx.value_and_grad(
          value_loss_fn, has_aux=True)(self.value_function)

      # Apply gradients
      self.value_optimizer.update(self.value_function, value_grads)
    else:
      # Just compute forward pass without gradients
      loss, (metrics, final_states) = value_loss_fn(self.value_function)

    return loss, metrics, final_states

  def _step(
      self,
      bm_frames: Frames,
      initial_states: RecurrentState,
      train: bool = True,
      compile: Optional[bool] = None,
  ) -> tuple[dict, RecurrentState]:
    """Single training/eval step."""
    compile = self.compile if compile is None else compile

    policy_initial_states, value_initial_states = initial_states

    if compile and self.use_shard_map:
      if train:
        policy_metrics, policy_final_states = self.sharded_train_policy(
            (bm_frames, policy_initial_states))
        value_metrics, value_final_states = self.sharded_train_value_function(
            (bm_frames, value_initial_states), discount=self.discount)
      else:
        policy_metrics, policy_final_states = self.sharded_run_policy(
            (bm_frames, policy_initial_states))
        value_metrics, value_final_states = self.sharded_run_value_function(
            (bm_frames, value_initial_states), discount=self.discount)
    else:
      if compile:
        step_policy_fn = self.jit_step_policy
        step_value_fn = self.jit_step_value_function
      else:
        step_policy_fn = self._step_policy
        step_value_fn = self._step_value_function

      _, policy_metrics, policy_final_states = step_policy_fn(
          bm_frames, policy_initial_states, train=train)
      _, value_metrics, value_final_states = step_value_fn(
          bm_frames, value_initial_states, train=train)

    metrics = dict(
        policy=policy_metrics,
        value=value_metrics,
    )

    final_states = (policy_final_states, value_final_states)
    return metrics, final_states

  def step(
      self,
      frames: Frames,
      initial_states: RecurrentState,
      train: bool = True,
      compile: Optional[bool] = None,
      combined: bool = False,
  ) -> tuple[dict, RecurrentState]:
    """Training/eval step.

    Args:
      frames: Batch of frames to train on.
      initial_states: Initial recurrent states.
      train: Whether to compute gradients and update parameters.
      compile: Whether to use JIT compilation.

    Returns:
      Tuple of (metrics dict, final recurrent states).
    """
    compile = self.compile if compile is None else compile

    if combined:
      if not compile:
        logging.warning('Learner.combine only matters when compiling.')

      step_fn = self.jit_step if compile else self._step
      # If using jit_step, we don't need to compile the inner
      # step_{policy,value_function}
      return step_fn(frames, initial_states, train=train, compile=False)

    return self._step(
        frames,
        initial_states,
        train=train,
        compile=compile,
    )

  def get_state(self):
    _, state = nnx.split(self)
    return state.to_pure_dict()

  def set_state(self, state_dict):
    _, state = nnx.split(self)
    nnx.replace_by_pure_dict(state, state_dict)
    nnx.update(self, state)
