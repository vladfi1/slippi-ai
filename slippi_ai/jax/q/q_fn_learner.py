import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from slippi_ai import utils
from slippi_ai.data import Batch, Frames
from slippi_ai.types import Controller, S
from slippi_ai.jax.policies import RecurrentState
from slippi_ai.jax.q import q_function as q_lib
from slippi_ai.jax import embed, rl_lib, jax_utils

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  reward_halflife: float = 4
  gae_lambda: float = 0

  unroll_batch_size: tp.Optional[int] = None

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
    assert delay == 0

    learning_rate = config.learning_rate
    self.q_function_optimizer = nnx.Optimizer(
        q_function, optax.adam(learning_rate), wrt=nnx.Param)

    self.discount = rl_lib.discount_from_halflife(
      config.reward_halflife, frame_skip=self.q_function.frame_skip)

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
        static_argnames=['unroll_batch_size'],
    )

    self.run_q_function = jax_utils.shard_map_loss_fn(
        module=self.q_function,
        loss_fn=self._unroll_q_function,
        mesh=mesh,
        static_argnames=['unroll_batch_size'],
    )

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return self.q_function.initial_state(batch_size, rngs)

  def _get_delayed_frames(self, frames: Frames[S, embed.Action]) -> Frames[S, embed.Action]:
    # delay == 0, so this is a no-op; kept for parity with nash.
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

  def _encode_frames(
      self, frames: Frames[S, Controller],
  ) -> Frames[S, embed.Action]:
    return Frames(
        state_action=self.q_function.core_net.encode(frames.state_action),
        is_resetting=frames.is_resetting,
        reward=frames.reward,
    )

  def _unroll_q_function(
      self,
      q_function: q_lib.QFunction[embed.Action],
      bm_frames: Frames[Rank2, embed.Action],  # [B, T]
      initial_state: RecurrentState,  # [B]
      *,
      unroll_batch_size: tp.Optional[int] = None,
      lambda_: float = 1.0,
  ) -> tuple[Loss, dict, RecurrentState]:
    frames = jax.tree.map(jax_utils.swap_axes, bm_frames)
    frames = self._get_delayed_frames(frames)

    if unroll_batch_size is None:
      unroll_batch_size = frames.reward.shape[0]

    q_outputs, final_state = q_function.loss_batched(
        frames, initial_state, self.discount, unroll_batch_size,
        lambda_=lambda_)

    bm_loss = jnp.mean(q_outputs.loss, axis=0)  # [T, B] -> [B]
    bm_metrics = jax.tree.map(jax_utils.swap_axes, q_outputs.metrics)

    return bm_loss, bm_metrics, final_state

  def step(
      self,
      batch: Batch,
      initial_state: RecurrentState,
      train: bool = True,
  ) -> tuple[dict, RecurrentState]:
    frames = batch.to_frames(self.q_function.frame_skip)
    frames = self._encode_frames(frames)

    if train:
      metrics, final_state = self.train_q_function(
        frames, initial_state, lambda_=self.config.gae_lambda)
    else:
      metrics, final_state = self.run_q_function(
        frames, initial_state, unroll_batch_size=self.config.unroll_batch_size)

    return {Q_FUNCTION: metrics}, final_state
