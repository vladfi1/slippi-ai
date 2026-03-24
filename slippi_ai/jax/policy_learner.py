
import dataclasses
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
from slippi_ai.jax import jax_utils
from slippi_ai.jax.jax_utils import swap_axes


Array = jax.Array

@dataclasses.dataclass
class LRDecayConfig:
  # Allow float for scientific notation at the command line.
  steps: tp.Optional[float] = None
  alpha: float = 0.1  # final lr multiplier

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  lr_decay: LRDecayConfig = dataclasses.field(default_factory=LRDecayConfig)
  weight_decay: float = 0
  nesterov_momentum: bool = False
  use_shard_map: bool = True
  combined: bool = False
  explicit_pmean: bool = False
  smap_optimizer: bool = True
  pack_data: bool = False

# TODO: move to jax_utils
P = tp.ParamSpec('P')
T = tp.TypeVar('T')

def jit_method(
    method: tp.Callable,
    *,
    donate_argnums: Optional[int | tuple[int, ...]] = None,
    static_argnames: tp.Optional[tp.Iterable[str]] = None,
) -> tp.Callable:
  if not isinstance(method, types.MethodType):
    raise TypeError('jit_method can only be applied to methods.')

  if donate_argnums is None:
    donate_argnums = ()
  elif isinstance(donate_argnums, int):
    donate_argnums = (donate_argnums,)

  jitted = nnx.jit(
      donate_argnums=(0,) + tuple(i+1 for i in donate_argnums),
      static_argnames=static_argnames,
  )(method.__func__)

  return nnx.cached_partial(jitted, method.__self__)

Metrics = dict[str, tp.Any]

def _policy_loss_fn(
    policy: Policy, frames: Frames, initial_states: RecurrentState,
) -> tuple[jax_utils.Loss, Metrics, RecurrentState]:
  tm_frames: Frames = jax.tree.map(swap_axes, frames)
  tm_loss, tm_metrics, final_states = policy.imitation_loss(tm_frames, initial_states)
  bm_loss = jnp.mean(tm_loss, axis=0)
  bm_metrics = jax.tree.map(swap_axes, tm_metrics)
  return bm_loss, bm_metrics, final_states


class PolicyLearner(nnx.Module):

  def __init__(
      self,
      policy: Policy,
      config: LearnerConfig,
      mesh: tp.Optional[jax.sharding.Mesh] = None,
      compile: bool = True,
  ):
    self.policy = policy
    self.compile = compile
    self.config = config

    if config.lr_decay.steps is None:
      schedule = config.learning_rate
    else:
      schedule = optax.cosine_decay_schedule(
          init_value=config.learning_rate,
          decay_steps=int(config.lr_decay.steps),
          alpha=config.lr_decay.alpha,
      )

    self.policy_optimizer = nnx.Optimizer(
        policy,
        optax.adamw(
            schedule,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov_momentum),
        wrt=nnx.Param)

    self.jit_step = jit_method(self._step, static_argnames=('train',))

    if mesh is not None:
      jax_utils.replicate_module(self, mesh)

    if config.use_shard_map:
      if mesh is None:
        raise ValueError('mesh must be provided when use_shard_map is True.')

      self.sharded_train = jax_utils.data_parallel_train(
          module=self.policy,
          optimizer=self.policy_optimizer,
          loss_fn=_policy_loss_fn,
          mesh=mesh,
          explicit_pmean=config.explicit_pmean,
          smap_optimizer=config.smap_optimizer,
          pack_data=config.pack_data,
      )

      self.sharded_run = jax_utils.shard_map_loss_fn(
          module=self.policy,
          loss_fn=_policy_loss_fn,
          mesh=mesh,
      )

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return self.policy.initial_state(batch_size, rngs)

  def _step(
      self,
      bm_frames: Frames,
      initial_states: RecurrentState,
      train: bool = True,
  ) -> tuple[Array, Metrics, RecurrentState]:
    """Single training/eval step for policy."""
    tm_frames: Frames = jax.tree.map(swap_axes, bm_frames)

    def loss_fn(policy: Policy):
      loss, metrics, final_states = policy.imitation_loss(tm_frames, initial_states)
      loss = jnp.mean(loss)
      return loss, (metrics, final_states)

    if train:
      (loss, (tm_metrics, final_states)), grads = nnx.value_and_grad(
          loss_fn, has_aux=True)(self.policy)
      self.policy_optimizer.update(self.policy, grads)
    else:
      loss, (tm_metrics, final_states) = loss_fn(self.policy)

    bm_metrics = jax.tree.map(swap_axes, tm_metrics)
    return loss, bm_metrics, final_states

  def step(
      self,
      frames: Frames,
      initial_states: RecurrentState,
      train: bool = True,
      compile: Optional[bool] = None,
  ) -> tuple[dict, RecurrentState]:
    compile = self.compile if compile is None else compile

    if compile and self.config.use_shard_map:
      if train:
        with jax.profiler.TraceAnnotation("sharded_train_policy"):
          metrics, final_states = self.sharded_train(frames, initial_states)
      else:
        with jax.profiler.TraceAnnotation("sharded_run_policy"):
          metrics, final_states = self.sharded_run(frames, initial_states)
    elif compile:
      _, metrics, final_states = self.jit_step(frames, initial_states, train=train)
    else:
      _, metrics, final_states = self._step(frames, initial_states, train=train)

    return {'policy': metrics}, final_states
