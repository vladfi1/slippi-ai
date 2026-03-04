import dataclasses
import typing as tp

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from slippi_ai import utils
from slippi_ai.nash.data import TwoPlayerBatch, batch_to_frames
from slippi_ai.types import Frames, Controller, S, Action
from slippi_ai.jax.policies import RecurrentState
from slippi_ai.jax.nash import q_function as q_lib
from slippi_ai.jax import embed, rl_lib, jax_utils
from slippi_ai.jax.nash.q_function import Rank3

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  reward_halflife: float = 4

  unroll_batch_size: tp.Optional[int] = None

Loss = jax.Array
Rank2 = tuple[int, int]

Q_FUNCTION = 'q_function'

class Learner(nnx.Module, tp.Generic[Action]):

  def __init__(
      self,
      config: LearnerConfig,
      q_function: q_lib.QFunction[Action],
      delay: int,
      mesh: jax.sharding.Mesh,
      data_sharding: jax.sharding.NamedSharding,
      explicit_pmean: bool = False,
      smap_optimizer: bool = True,
  ):
    self.config = config
    self.q_function = q_function
    self.delay = delay
    assert delay == 0

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
        static_argnames=['unroll_batch_size'],
    )

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return self.q_function.initial_state(batch_size, rngs)

  def _get_delayed_frames(self, frames: Frames[S, Action]) -> Frames[S, Action]:
    state_action = frames.state_action
    unroll_length = frames.is_resetting.shape[0] - self.delay

    return Frames[S, Action](
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

  def _encode_frames(
      self, frames: Frames[S, Controller],
  ) -> Frames[S, embed.Action]:
    return Frames(
        state_action=self.q_function.core_net.encode(frames.state_action),
        is_resetting=frames.is_resetting,
        reward=frames.reward,
    )

  def prepare_frames(self, batch: TwoPlayerBatch[Rank2]) -> Frames[Rank3, embed.Action]:
    # Note: inputs and outputs are batch-major
    zipped_frames = batch_to_frames(batch)
    encoded_frames = self._encode_frames(zipped_frames)
    return utils.map_single_structure(
      lambda x: jax.device_put(x, self.data_sharding), encoded_frames)

  def _unroll_q_function(
      self,
      q_function: q_lib.QFunction[embed.Action],
      combined_frames: Frames[Rank3, embed.Action],  # [B, 2, T]
      initial_state: RecurrentState,  # [B, 2]
      *,
      unroll_batch_size: tp.Optional[int] = None,
  ) -> tuple[Loss, dict, RecurrentState]:
    # Move time dimension to the front.
    frames = utils.map_nt(
      lambda x: jnp.moveaxis(x, 2, 0), combined_frames)
    frames = self._get_delayed_frames(frames)

    if unroll_batch_size is None:
      q_outputs, final_state = q_function.loss(
          frames, initial_state, self.discount)
    else:
      q_outputs, final_state = q_function.loss_batched(
          frames, initial_state, self.discount, unroll_batch_size)

    bm_loss = jnp.mean(q_outputs.loss, axis=[0, 2])
    bm_metrics = jax.tree.map(lambda x: jnp.moveaxis(x, 0, 2), q_outputs.metrics)

    return bm_loss, bm_metrics, final_state

  def step(
      self,
      batch: TwoPlayerBatch[Rank2],
      initial_state: RecurrentState,
      train: bool = True,
  ) -> tuple[dict, RecurrentState]:
    frames = self.prepare_frames(batch)

    if train:
      metrics, final_state = self.train_q_function(frames, initial_state)
    else:
      metrics, final_state = self.run_q_function(
        frames, initial_state, unroll_batch_size=self.config.unroll_batch_size)

    return {Q_FUNCTION: metrics}, final_state
