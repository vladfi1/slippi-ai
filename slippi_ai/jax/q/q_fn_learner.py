import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from slippi_ai import utils
from slippi_ai.data import Batch, Frames, StateAction
from slippi_ai.jax.policies import RecurrentState
from slippi_ai.jax.q import q_function as q_lib
from slippi_ai.jax import embed, rl_lib, jax_utils

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  reward_halflife: float = 4

Loss = jax.Array
Rank2 = tuple[int, int]

Q_FUNCTION = 'q_function'

class Learner(nnx.Module, tp.Generic[embed.Action]):

  def __init__(
      self,
      config: LearnerConfig,
      q_function: q_lib.QFunction[embed.Action],
      delay: int,
      mesh: jax.sharding.Mesh,
      data_sharding: jax.sharding.NamedSharding,
      explicit_pmean: bool = False,
      smap_optimizer: bool = True,
  ):
    self.config = config
    self.q_function = q_function
    self.delay = delay

    learning_rate = config.learning_rate
    self.q_function_optimizer = nnx.Optimizer(
        q_function, optax.adam(learning_rate), wrt=nnx.Param)

    self.discount = rl_lib.discount_from_halflife(config.reward_halflife)

    jax_utils.replicate_module(self, mesh)

    self.data_sharding = data_sharding
    sharding_kwargs = dict(
        mesh=mesh,
        explicit_pmean=explicit_pmean,
        smap_optimizer=smap_optimizer,
    )

    self.train_q_function = jax_utils.data_parallel_train(
        module=self.q_function,
        optimizer=self.q_function_optimizer,
        loss_fn=self._unroll_q_function,
        **sharding_kwargs,
    )

    self.run_q_function = jax_utils.shard_map_loss_fn(
        module=self.q_function,
        loss_fn=self._unroll_q_function,
        mesh=mesh,
    )

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return self.q_function.initial_state(batch_size, rngs)

  def _get_delayed_frames(self, frames: Frames[Rank2, embed.Action]) -> Frames[Rank2, embed.Action]:
    state_action = frames.state_action
    unroll_length = frames.is_resetting.shape[0] - self.delay

    return Frames(
        state_action=embed.StateAction(
            state=jax.tree.map(
                lambda t: t[:unroll_length], state_action.state),
            action=jax.tree.map(
                lambda t: t[self.delay:], state_action.action),
            name=state_action.name[:unroll_length],
        ),
        is_resetting=frames.is_resetting[:unroll_length],
        reward=frames.reward[self.delay:],
    )

  def _shard_frames(self, frames: Frames[Rank2, embed.Action]) -> Frames[Rank2, embed.Action]:
    return utils.map_single_structure(lambda x: jax.device_put(x, self.data_sharding), frames)

  def prepare_frames(self, batch: Batch[Rank2]) -> Frames[Rank2, embed.Action]:
    state_action = StateAction(
        batch.game, batch.game.p0.controller, batch.name)
    frames = Frames(
        state_action=self.q_function.core_net.encode(state_action),
        is_resetting=batch.is_resetting,
        reward=batch.reward,
    )
    return self._shard_frames(frames)

  def _unroll_q_function(
      self,
      q_function: q_lib.QFunction[embed.Action],
      bm_frames: Frames[Rank2, embed.Action],
      initial_state: RecurrentState,
  ) -> tuple[Loss, dict, RecurrentState]:
    frames = jax.tree.map(jax_utils.swap_axes, bm_frames)
    frames = self._get_delayed_frames(frames)

    q_outputs, final_state = q_function.loss(frames, initial_state, self.discount)

    bm_loss = jnp.mean(q_outputs.loss, axis=0)
    bm_metrics = jax.tree.map(jax_utils.swap_axes, q_outputs.metrics)

    return bm_loss, bm_metrics, final_state

  def step(
      self,
      batch: Batch,
      initial_state: RecurrentState,
      train: bool = True,
  ) -> tuple[dict, RecurrentState]:
    state_action = StateAction(
        batch.game, batch.game.p0.controller, batch.name)
    frames = Frames(
        state_action=self.q_function.core_net.encode(state_action),
        is_resetting=batch.is_resetting,
        reward=batch.reward,
    )
    frames = self._shard_frames(frames)

    if train:
      metrics, final_state = self.train_q_function(frames, initial_state)
    else:
      metrics, final_state = self.run_q_function(frames, initial_state)

    return {Q_FUNCTION: metrics}, final_state
