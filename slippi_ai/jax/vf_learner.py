
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
  explicit_pmean: bool = False
  smap_optimizer: bool = True


def value_loss_fn(
    value_function: vf_lib.ValueFunction,
    frames: Frames,
    initial_states: RecurrentState,
    discount: float,
) -> tuple[jax_utils.Loss, dict, RecurrentState]:
  tm_frames: Frames = jax.tree.map(swap_axes, frames)
  value_outputs, final_states = value_function.loss(tm_frames, initial_states, discount)
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

    self.train_vf = jax_utils.data_parallel_train(
        module=self.value_function,
        optimizer=self.value_optimizer,
        loss_fn=self._unroll_value_function,
        **sharding_kwargs,
    )

    self.run_vf = jax_utils.shard_map_loss_fn(
        module=self.value_function,
        loss_fn=self._unroll_value_function,
        mesh=mesh,
    )

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return self.value_function.initial_state(batch_size, rngs)

  def _unroll_value_function(
      self,
      value_function: vf_lib.ValueFunction,
      bm_frames: Frames,
      initial_state: RecurrentState,
  ) -> tuple[Array, dict, RecurrentState]:
    return value_loss_fn(value_function, bm_frames, initial_state, self.discount)

  def step(
      self,
      batch: Batch,
      initial_state: RecurrentState,
      train: bool = True,
  ) -> tuple[dict, RecurrentState]:
    state_action = StateAction(
        batch.game, batch.game.p0.controller, batch.name)
    frames = Frames(
        state_action=self.value_function.network.encode(state_action),
        is_resetting=batch.is_resetting,
        reward=batch.reward,
    )
    frames = jax_utils.device_put(frames, self.data_sharding)

    if train:
      metrics, final_state = self.train_vf(frames, initial_state)
    else:
      metrics, final_state = self.run_vf(frames, initial_state)

    return {'value': metrics}, final_state
