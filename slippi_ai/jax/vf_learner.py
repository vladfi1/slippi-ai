
import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from slippi_ai.data import Batch, Frames, StateAction
from slippi_ai.jax.networks import RecurrentState
from slippi_ai.jax import value_function as vf_lib
from slippi_ai.jax import jax_utils, rl_lib
from slippi_ai.jax.jax_utils import swap_axes

Array = jax.Array


@dataclasses.dataclass
class VFLearnerConfig:
  learning_rate: float = 1e-4
  reward_halflife: float = 4
  gae_lambda: float = 0
  explicit_pmean: bool = False
  smap_optimizer: bool = True
  bf16: bool = False

  unroll_batch_size: tp.Optional[int] = None


def value_loss_fn(
    value_function: vf_lib.ValueFunction,
    frames: Frames,
    initial_states: RecurrentState,
    discount: float,
    unroll_batch_size: tp.Optional[int] = None,
    lambda_: float = 1.0,
) -> tuple[jax_utils.Loss, dict, RecurrentState]:
  tm_frames: Frames = jax.tree.map(swap_axes, frames)
  if unroll_batch_size is None:
    value_outputs, final_states = value_function.loss(
        tm_frames, initial_states, discount, lambda_=lambda_)
  else:
    value_outputs, final_states = value_function.loss_batched(
        tm_frames, initial_states, discount, unroll_batch_size,
        lambda_=lambda_)
  loss = jnp.mean(value_outputs.loss, axis=0)
  bm_metrics = jax.tree.map(swap_axes, value_outputs.metrics)
  return loss, bm_metrics, final_states


class VFLearner(nnx.Module):

  def __init__(
      self,
      value_function: vf_lib.ValueFunction,
      config: VFLearnerConfig,
      mesh: jax.sharding.Mesh,
      data_sharding: jax.sharding.NamedSharding,
  ):
    self.value_function = value_function
    self.discount = rl_lib.discount_from_halflife(config.reward_halflife)
    self.config = config
    self.data_sharding = data_sharding

    self.value_optimizer = nnx.Optimizer(
        value_function, optax.adamw(config.learning_rate), wrt=nnx.Param)

    jax_utils.replicate_module(self, mesh)

    sharding_kwargs = dict(
        mesh=mesh,
        explicit_pmean=config.explicit_pmean,
        smap_optimizer=config.smap_optimizer,
    )

    unroll_vf = self._unroll_value_function
    if config.bf16:
      unroll_vf = jax_utils.with_bf16_compute(unroll_vf)

    self.train_vf = jax_utils.data_parallel_train(
        module=self.value_function,
        optimizer=self.value_optimizer,
        loss_fn=unroll_vf,
        static_argnames=['lambda_'],
        **sharding_kwargs,
    )

    self.run_vf = jax_utils.shard_map_loss_fn(
        module=self.value_function,
        loss_fn=unroll_vf,
        mesh=mesh,
        static_argnames=['unroll_batch_size', 'lambda_'],
    )

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    state = self.value_function.initial_state(batch_size, rngs)
    if self.config.bf16:
      state = jax_utils.cast_floats_to_dtype(state, jnp.bfloat16)
    return state

  def _unroll_value_function(
      self,
      value_function: vf_lib.ValueFunction,
      bm_frames: Frames,
      initial_state: RecurrentState,
      *,
      unroll_batch_size: tp.Optional[int] = None,
      lambda_: float = 1.0,
  ) -> tuple[Array, dict, RecurrentState]:
    return value_loss_fn(
        value_function, bm_frames, initial_state, self.discount,
        unroll_batch_size=unroll_batch_size, lambda_=lambda_)

  def step(
      self,
      batch: Batch,
      initial_state: RecurrentState,
      train: bool = True,
  ) -> tuple[dict, RecurrentState]:
    state_action = StateAction(
        batch.game, batch.game.p0.controller, batch.name, batch.rating)
    frames = Frames(
        state_action=self.value_function.network.encode(state_action),
        is_resetting=batch.is_resetting,
        reward=batch.reward,
    )
    frames = jax_utils.device_put(frames, self.data_sharding)

    if train:
      metrics, final_state = self.train_vf(
          frames, initial_state, lambda_=self.config.gae_lambda)
    else:
      # Evaluate with lambda=1 for unbiased value targets.
      metrics, final_state = self.run_vf(
          frames, initial_state,
          unroll_batch_size=self.config.unroll_batch_size)

    return {'value': metrics}, final_state
