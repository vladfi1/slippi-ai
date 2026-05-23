
import dataclasses
from typing import Optional
import typing as tp
import types

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from slippi_ai.data import Frames
from slippi_ai.jax.policies import Policy, ControllerType
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
  bf16: bool = False

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


def check_compute_dtypes(policy: Policy[ControllerType]) -> dict[str, str]:
  """Return a dict of component name -> compute dtype for key parts of the policy.

  Useful to verify that bf16 is actually flowing through the intended components.
  Uses jax.eval_shape so no actual computation is done.
  Simulates bf16 compute by casting params to bf16 first.
  """
  policy = jax_utils.cast_params_to_dtype(policy, jnp.bfloat16)
  rngs = nnx.Rngs(0)
  B = (1,)
  dummy_sa = policy.network.dummy(B)

  results = {}

  from slippi_ai.jax import networks
  assert isinstance(policy.network, networks.SimpleEmbedNetwork)

  # Enhanced embed output before the compute_dtype cast
  results['enhanced_embed_out'] = str(
      jax.eval_shape(policy.network._embed_module, dummy_sa).dtype)

  # After compute_dtype cast in _embed
  results['embed_out'] = str(
      jax.eval_shape(policy.network._embed, dummy_sa).dtype)

  # Main network (TransformerLike) output
  init_state = policy.network.initial_state(B, rngs)
  net_out, _ = jax.eval_shape(policy.network.step, dummy_sa, init_state)
  results['network_out'] = str(net_out.dtype)

  # Controller head encoder output
  from slippi_ai.jax import controller_heads
  ctrl_head = policy._controller_head
  if isinstance(ctrl_head, controller_heads.AutoRegressive):
    residual = jax.eval_shape(ctrl_head.to_residual, net_out)
    results['controller_head_residual'] = str(residual.dtype)

  # ControllerRNN (if present)
  embed_mod = policy.network._embed_module
  if isinstance(embed_mod, networks.EnhancedEmbedModule) and embed_mod._use_controller_rnn:
    crnn = embed_mod._controller_rnn
    crnn_out = jax.eval_shape(crnn, dummy_sa.action)
    crnn_leaves = jax.tree_util.tree_leaves(crnn_out)
    if crnn_leaves:
      results['controller_rnn_out'] = str(crnn_leaves[0].dtype)

  return results


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

    if config.bf16:
      dtypes = check_compute_dtypes(policy)
      import logging
      logging.info('bf16 compute dtypes: %s', dtypes)

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

      loss_fn = (jax_utils.with_bf16_compute(_policy_loss_fn)
                 if config.bf16 else _policy_loss_fn)

      self.sharded_train = jax_utils.data_parallel_train(
          module=self.policy,
          optimizer=self.policy_optimizer,
          loss_fn=loss_fn,
          mesh=mesh,
          explicit_pmean=config.explicit_pmean,
          smap_optimizer=config.smap_optimizer,
          pack_data=config.pack_data,
      )

      self.sharded_run = jax_utils.shard_map_loss_fn(
          module=self.policy,
          loss_fn=loss_fn,
          mesh=mesh,
      )

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    state = self.policy.initial_state(batch_size, rngs)
    if self.config.bf16:
      state = jax_utils.cast_floats_to_dtype(state, jnp.bfloat16)
    return state

  def _step(
      self,
      bm_frames: Frames,
      initial_states: RecurrentState,
      train: bool = True,
  ) -> tuple[Array, Metrics, RecurrentState]:
    """Single training/eval step for policy."""
    tm_frames: Frames = jax.tree.map(swap_axes, bm_frames)

    def loss_fn(policy: Policy):
      if self.config.bf16:
        policy = jax_utils.cast_params_to_dtype(policy, jnp.bfloat16)
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
