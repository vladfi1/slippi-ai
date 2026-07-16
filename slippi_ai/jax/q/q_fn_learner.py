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

  # Number of epistemic indices per trajectory to train on. The loss is linear
  # in the expectation over indices, so this only affects gradient variance;
  # it needs to be at least 2 for the epistemic metrics to be defined.
  num_index_samples: int = 4
  # Number of epistemic indices to evaluate with (also as an ensemble);
  # defaults to num_index_samples.
  eval_num_index_samples: tp.Optional[int] = None

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
      rngs: tp.Optional[nnx.Rngs] = None,
      explicit_pmean: bool = False,
      smap_optimizer: bool = True,
  ):
    self.config = config
    self.q_function = q_function
    self.delay = delay
    assert delay == 0
    if config.num_index_samples < 1:
      raise ValueError('num_index_samples must be at least 1')
    self.eval_num_index_samples = config.eval_num_index_samples
    if self.eval_num_index_samples is None:
      self.eval_num_index_samples = config.num_index_samples
    if self.eval_num_index_samples < 1:
      raise ValueError('eval_num_index_samples must be at least 1')

    learning_rate = config.learning_rate
    self.q_function_optimizer = nnx.Optimizer(
        q_function, optax.adam(learning_rate), wrt=nnx.Param)

    self.discount = rl_lib.discount_from_halflife(config.reward_halflife)
    self.fs_discount = rl_lib.discount_from_halflife(
        config.reward_halflife, frame_skip=self.q_function.frame_skip)

    jax_utils.replicate_module(self, mesh)

    self.data_sharding = data_sharding
    sharding_kwargs = dict(
        mesh=mesh,
        explicit_pmean=explicit_pmean,
        smap_optimizer=smap_optimizer,
    )

    if rngs is None:
      rngs = nnx.Rngs(0)

    self.train_q_function = jax_utils.data_parallel_train_with_rngs(
        module=self.q_function,
        optimizer=self.q_function_optimizer,
        rngs=rngs.fork(),
        loss_fn=self._unroll_q_function,
        **sharding_kwargs,
        static_argnames=['unroll_batch_size', 'num_index_samples'],
    )

    self.run_q_function = jax_utils.shard_map_loss_fn_with_rngs(
        module=self.q_function,
        rngs=rngs.fork(),
        loss_fn=self._unroll_q_function,
        mesh=mesh,
        static_argnames=['unroll_batch_size', 'num_index_samples'],
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
      rngs: nnx.Rngs,
      *,
      unroll_batch_size: tp.Optional[int] = None,
      lambda_: float = 1.0,
      num_index_samples: int = 1,
  ) -> tuple[Loss, dict, RecurrentState]:
    frames = jax.tree.map(jax_utils.swap_axes, bm_frames)
    frames = self._get_delayed_frames(frames)

    if unroll_batch_size is None:
      unroll_batch_size = frames.reward.shape[0]

    q_outputs, final_state = q_function.loss_batched(
        frames, initial_state, self.fs_discount, unroll_batch_size,
        lambda_=lambda_, rngs=rngs,
        num_index_samples=num_index_samples)

    bm_loss = jnp.mean(q_outputs.loss, axis=0)  # [T, B] -> [B]
    bm_metrics = jax.tree.map(jax_utils.swap_axes, q_outputs.metrics)

    return bm_loss, bm_metrics, final_state

  def step(
      self,
      batch: Batch,
      initial_state: RecurrentState,
      train: bool = True,
  ) -> tuple[dict, RecurrentState]:
    frames = batch.to_frames(self.q_function.frame_skip, discount=self.discount)
    frames = self._encode_frames(frames)

    if train:
      metrics, final_state = self.train_q_function(
        frames, initial_state, lambda_=self.config.gae_lambda,
        num_index_samples=self.config.num_index_samples)
    else:
      metrics, final_state = self.run_q_function(
        frames, initial_state, unroll_batch_size=self.config.unroll_batch_size,
        num_index_samples=self.eval_num_index_samples)

    return {Q_FUNCTION: metrics}, final_state
