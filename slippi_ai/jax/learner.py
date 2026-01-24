import dataclasses
import functools
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from slippi_ai.data import Frames
from slippi_ai.jax.policies import Policy
from slippi_ai.jax.networks import RecurrentState
from slippi_ai.jax import value_function as vf_lib
from slippi_ai import utils

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
  decay_rate: float = 0.
  value_cost: float = 0.5
  reward_halflife: float = 4


class Learner(nnx.Module):

  def __init__(
      self,
      policy: Policy,
      learning_rate: float,
      value_cost: float,
      reward_halflife: float,
      value_function: Optional[vf_lib.ValueFunction] = None,
      decay_rate: Optional[float] = None,
  ):
    self.policy = policy
    self.value_function = value_function or vf_lib.FakeValueFunction()
    self.value_cost = value_cost
    self.discount = 0.5 ** (1 / (reward_halflife * 60))
    self.decay_rate = decay_rate

    # Create optimizers using optax
    self.policy_optimizer = nnx.Optimizer(
        policy, optax.adam(learning_rate), wrt=nnx.Param)

    self.value_optimizer = nnx.Optimizer(
        self.value_function, optax.adam(learning_rate), wrt=nnx.Param)

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return (
        self.policy.initial_state(batch_size, rngs),
        self.value_function.initial_state(batch_size, rngs),
    )

  def _step(
      self,
      bm_frames: Frames,
      initial_states: RecurrentState,
      train: bool = True,
  ) -> tuple[dict, RecurrentState]:
    """Single training/eval step."""
    policy_initial_states, value_initial_states = initial_states

    # Switch axes to time-major
    tm_frames: Frames = jax.tree.map(swap_axes, bm_frames)

    # Define loss function that takes policy as argument for gradient computation
    def policy_loss_fn(policy: Policy):
      policy_loss, policy_final_states, policy_metrics = policy.imitation_loss(
          tm_frames, policy_initial_states,
          self.value_cost, self.discount)
      return policy_loss, (policy_metrics, policy_final_states)

    # Define loss function for value function
    def value_loss_fn(value_function: vf_lib.ValueFunction):
      delay = self.policy.delay
      value_frames = jax.tree.map(
          lambda t: t[:t.shape[0] - delay] if delay > 0 else t, tm_frames)
      value_outputs, value_final_states = value_function.loss(
          value_frames, value_initial_states, self.discount)
      return jnp.mean(value_outputs.loss), (value_outputs.metrics, value_final_states)

    if train:
      # Compute gradients for policy using value_and_grad to also get loss
      (policy_loss, (policy_metrics, policy_final_states)), policy_grads = nnx.value_and_grad(
          policy_loss_fn, has_aux=True)(self.policy)

      # Compute gradients for value function
      (value_loss, (value_metrics, value_final_states)), value_grads = nnx.value_and_grad(
          value_loss_fn, has_aux=True)(self.value_function)

      # Apply gradients
      self.policy_optimizer.update(self.policy, policy_grads)
      self.value_optimizer.update(self.value_function, value_grads)

    else:
      # Just compute forward pass without gradients
      policy_loss, (policy_metrics, policy_final_states) = policy_loss_fn(self.policy)
      value_loss, (value_metrics, value_final_states) = value_loss_fn(self.value_function)

    total_loss = policy_loss + value_loss

    metrics = dict(
        total_loss=total_loss,
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
  ) -> tuple[dict, RecurrentState]:
    """Training step."""
    return self._step(frames, initial_states, train=train)
