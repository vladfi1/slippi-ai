import dataclasses
import functools
import logging
import typing as tp

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from slippi_ai import utils
from slippi_ai.types import S, Frames, Action, StateAction
from slippi_ai.jax.policies import Policy, RecurrentState
from slippi_ai.jax import embed, rl_lib, jax_utils, saving
from slippi_ai.jax.jax_utils import PS, DATA_AXIS
from slippi_ai.jax.agents import DType
from slippi_ai.nash import data as nash_data
from slippi_ai.jax.nash import (
    q_function as q_lib,
    utils as nash_utils,
)
from slippi_ai.jax.nash import nash
from slippi_ai.jax.rl.learner import FrameSkipTrajectory

T = tp.TypeVar('T')
Rank3 = tuple[int, int, int]

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  q_fn_learning_rate: tp.Optional[float] = None
  reward_halflife: float = 4
  gae_lambda: float = 0

  num_samples: int = 1
  sample_batch_size: int = 0  # 0 means full batch size, i.e. vmap
  include_action_taken_in_samples: bool = True
  subsample: tp.Optional[int] = None
  epoch_length: int = 100

  remat: bool = True

  # Number of epistemic indices to sample. The nash is solved once per index,
  # and the nash_policy regresses to the mixture of the per-index nash
  # distributions. Needs to be at least 2 for the epistemic metrics.
  num_index_samples: int = 4

  nash_weight: float = 1
  weight_by_advantage: bool = True

  initial_kl_weight: float = 3e-1
  kl_weight_lr: float = 3e-2
  target_teacher_kl: float = 0.05
  target_reverse_teacher_kl: float = 0.05

  value_burnin_steps: int = 0

  sample_policy_dtype: DType = DType.FP32
  teacher_dtype: DType = DType.FP32
  nash_policy_dtype: DType = DType.FP32
  q_fn_dtype: DType = DType.FP32

  microbatch_size: int = 0
  teacher_mbs: int = 0


_SAMPLE_AXIS = 0

Loss = jax.Array
Metrics = dict
Values = jax.Array
QValues = jax.Array

QFunctionOutputs = tuple[
    Loss,  # [B]
    Metrics,  # [B]
    RecurrentState,  # final state [B]
    Values,  # [N, T, B, 2] per epistemic index
    RecurrentState,  # action_init_state [T, B, 2, H]
    QValues,  # [N, S, S, T, B, 2] per epistemic index
    jax.Array,  # zs [N, B, 1, D_Z] epistemic indices
]


class ShardingKwargs(tp.TypedDict):
  mesh: jax.sharding.Mesh
  explicit_pmean: bool
  smap_optimizer: bool

class ShardingSpecs(tp.TypedDict):
  extra_in_specs: tp.Optional[tp.Sequence[PS]]
  extra_out_specs: tp.Optional[tp.Sequence[PS]]

SAMPLE_POLICY = 'sample_policy'
Q_FUNCTION = 'q_function'
NASH = 'nash'
NASH_POLICY = 'nash_policy'
TEACHER = 'teacher'

def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
  masked_sum = jnp.sum(x * mask, keepdims=True)
  count = jnp.sum(mask)
  return masked_sum / (count + 1e-8)

def p1_averaged_qs(two_player_qs: jax.Array) -> jax.Array:
  """Get Q-values from just player 1's perspective, assuming zero-sum."""
  # two_player_qs is [..., 2]
  return jnp.vecdot(
      two_player_qs, jnp.array([1, -1], dtype=two_player_qs.dtype),
      axis=-1) / 2

def get_frames(trajectory: FrameSkipTrajectory[Action]) -> Frames[Rank3, Action]:
  """Gives time-major frames with actions taken."""
  state_action = StateAction(
      state=trajectory.states,
      action=[so.controller_state for so in trajectory.actions],
      name=trajectory.name,
  )
  return Frames(state_action, trajectory.is_resetting, trajectory.rewards)

def batch_fs(xs: list[T]) -> T:
  return utils.map_nt(
      lambda *xs: jnp.stack(xs, axis=1),
      *xs
  )

def warmup_schedule(burnin_steps: int, base_value: float):
  burnin = optax.constant_schedule(0)
  normal = optax.constant_schedule(base_value)
  return optax.join_schedules([burnin, normal], [burnin_steps])

@jax.jit
def copy_struct(struct: T) -> T:
  return jax.tree.map(jnp.copy, struct)

class Learner(nnx.Module, tp.Generic[Action]):

  def __init__(
      self,
      config: LearnerConfig,
      q_function_config: q_lib.QFunctionConfig,
      policy_config: dict,
      state: dict,
      rngs: nnx.Rngs,  # used for sampling
      # mesh: jax.sharding.Mesh,
      # explicit_pmean: bool = False,
      # smap_optimizer: bool = True,
  ):
    self.config = config

    q_function = q_lib.build_q_function(rngs, q_function_config)

    self.q_function = q_function

    # We use the policy to sample and act, and regress the nash_policy towards
    # the resulting Nash distribution. At the end of each epoch, we copy the
    # nash_policy's weights to the policy.
    self.policy: Policy[Action] = saving.policy_from_config_dict(policy_config)
    self.nash_policy: Policy[Action] = saving.policy_from_config_dict(policy_config)
    self.teacher: Policy[Action] = saving.policy_from_config_dict(policy_config)

    if config.remat:
      # Only the nash_policy is trained.
      self.nash_policy.enable_remat()

    self._controller_embedding = self.policy.controller_head.controller_embedding

    # within-frame discount
    self.discount = rl_lib.discount_from_halflife(config.reward_halflife)
    # across-frame discount
    self.fs_discount = rl_lib.discount_from_halflife(
      config.reward_halflife, frame_skip=q_function.frame_skip)

    learning_rate = config.learning_rate

    self.policy_schedule = warmup_schedule(
        config.value_burnin_steps,
        config.learning_rate,
    )

    self.policy_optimizer = nnx.Optimizer(
        self.policy, optax.adam(self.policy_schedule), wrt=nnx.Param)

    q_fn_learning_rate = config.q_fn_learning_rate or learning_rate

    self.q_function_optimizer = nnx.Optimizer(
        q_function, optax.adam(q_fn_learning_rate), wrt=nnx.Param)

    kl_weight_schedule = warmup_schedule(
        config.value_burnin_steps, config.kl_weight_lr)
    self.kl_teacher_weights = jax_utils.KLTeacherWeights(config.initial_kl_weight)
    self.kl_teacher_weights_optimizer = nnx.Optimizer(
      self.kl_teacher_weights, optax.sgd(kl_weight_schedule), wrt=nnx.Param)

    # NOTE: some jax_utils functions expect jax arrays inside modules.
    jax_utils.set_module_state(self, utils.map_nt(jnp.asarray, state))

    if not config.include_action_taken_in_samples and config.num_samples < 2:
      raise ValueError('num_samples must be at least 2 if not including action taken in samples')

    if config.sample_batch_size > 0:
      ns = config.num_samples
      if config.include_action_taken_in_samples:
        ns += 1
      if ns % config.sample_batch_size != 0:
        logging.warning(f'sample_batch_size {config.sample_batch_size} does not divide num_samples {ns}')

    self.num_samples = config.num_samples
    self.delay = self.policy.delay
    self.frame_skip = self.policy.frame_skip
    assert self.q_function.frame_skip == self.frame_skip

    # jax_utils.replicate_module(self, mesh)

    # sharding_kwargs = ShardingKwargs(
    #     mesh=mesh,
    #     explicit_pmean=explicit_pmean,
    #     smap_optimizer=smap_optimizer,
    # )

    # BM = PS(DATA_AXIS)
    # tms_specs = [None, DATA_AXIS]
    # TM = PS(*tms_specs)  # time-major
    # tms_specs.insert(_SAMPLE_AXIS, None)
    # TMS = PS(*tms_specs)  # time-major with samples
    # tms_specs.insert(_SAMPLE_AXIS, None)
    # TMSS = PS(*tms_specs)  # time-major SxS

    # policy_samples = TMS
    # vs = TM
    # q_action_init = TM
    # qs = TMSS
    # nash_solution = TM
    # metrics = BM

    # sample_policy_specs = ShardingSpecs(
    #     extra_in_specs=None,
    #     extra_out_specs=(policy_samples,),
    # )

    jax_utils.cast_module_state_to_dtype(
      self.policy, config.sample_policy_dtype.dtype)

    self.run_sample_policy = jax_utils.cached_partial(
        jax_utils.nnx_jit(
            jax_utils.no_loss(self._unroll_sample_policy),
            donate_argnums=(0, 1, 3),
        ),
        self.policy, rngs,
    )

    # q_function_specs = ShardingSpecs(
    #     extra_in_specs=(policy_samples,),
    #     extra_out_specs=(vs, q_action_init, qs),
    # )

    # self.run_q_function_bm = jax_utils.shard_map_loss_fn(
    #     module=self.q_function,
    #     loss_fn=self._unroll_q_function,
    #     mesh=mesh,
    #     **q_function_specs,
    # )

    # self.train_q_function = jax_utils.data_parallel_train(
    #     module=self.q_function,
    #     optimizer=self.q_function_optimizer,
    #     loss_fn=self._unroll_q_function,
    #     **sharding_kwargs,
    #     **q_function_specs,
    # )

    unroll_q_function = jax_utils.with_compute_dtype(
      self._unroll_q_function, config.q_fn_dtype.dtype)

    self.train_q_function = jax_utils.train_fn_with_rngs(
        module=self.q_function,
        optimizer=self.q_function_optimizer,
        rngs=rngs.fork(),
        loss_fn=unroll_q_function,
    )

    self.run_q_function = jax_utils.cached_partial(
        jax_utils.nnx_jit(
            jax_utils.no_loss(unroll_q_function),
            donate_argnums=(0, 1, 3),
        ),
        self.q_function, rngs.fork(),
    )

    # We can't shard_map the qpax solver because of vma issues with while_loop.
    # The solution would be to insert a manual pvary inside qpax like we do in
    # our own ippd solver, but we can also just let jit handle running on
    # multiple devices as the solver is completely batch-parallel.

    # sharded_compute_nash = jax_utils.shard_map(
    #     self._compute_nash,
    #     mesh=mesh,
    #     in_specs=(qs,),
    #     out_specs=(nash_solution, metrics),
    # )
    # self.compute_nash = jax_utils.jit(sharded_compute_nash)
    self.compute_nash = jax_utils.jit(
      self._compute_nash,
      # in_shardings=jax.NamedSharding(mesh, qs),
      # out_shardings=(jax.NamedSharding(mesh, nash_solution), jax.NamedSharding(mesh, metrics)),
    )
    self.compute_nash = jax.profiler.annotate_function(self.compute_nash)

    def unroll_teacher(
        teacher: Policy[Action],
        frames: Frames[nash_data.Rank3, Action], /,
        initial_states: RecurrentState,  # [B, 2]
    ) -> tuple[jax.Array, RecurrentState]:
      # Only compute the core network outputs; the controller head is applied
      # in _unroll_nash_policy where the teacher's logits are evaluated on
      # actions sampled from the teacher and nash_policy.
      inputs = utils.map_nt(lambda t: t[:-1], frames.state_action)
      outputs, final_state = teacher.network.unroll(
          inputs, frames.is_resetting[:-1], initial_states)
      return outputs, final_state

    jax_utils.cast_module_state_to_dtype(
      self.teacher, config.teacher_dtype.dtype)

    self.run_teacher = jax_utils.cached_partial(
        jax_utils.nnx_jit(
            unroll_teacher,
            donate_argnums=(0, 2),
        ),
        self.teacher,
    )

    # nash_policy_specs = ShardingSpecs(
    #     extra_in_specs=(policy_samples, vs, q_action_init, nash_solution),
    #     extra_out_specs=None,
    # )

    unroll_nash_policy = jax_utils.with_compute_dtype(
        self._unroll_nash_policy, config.nash_policy_dtype.dtype)

    train_nash_policy = jax_utils.train_fn(unroll_nash_policy)

    self.train_nash_policy = jax_utils.cached_partial(
        jax_utils.nnx_jit(train_nash_policy, donate_argnums=(0, 1, 2, 3, 4, 5, 7)),
        self.nash_policy, self.policy_optimizer, rngs, self.q_function,
        self.teacher, self.kl_teacher_weights,
    )

    self.run_nash_policy = jax_utils.cached_partial(
        jax_utils.nnx_jit(
            jax_utils.no_loss(unroll_nash_policy),
            donate_argnums=(0, 1, 2, 3, 4, 6),
        ),
        self.nash_policy, rngs, self.q_function, self.teacher,
        self.kl_teacher_weights,
    )

    def post_update(
        policy: Policy[Action], /,
        frames: Frames[nash_data.Rank3, Action],
        initial_state: RecurrentState,
        fs_actor_logits: list[Action],  # FS x [T, B, 2]
    ) -> Metrics:
      policy_outputs = policy.unroll(frames, initial_state)
      policy_logits = batch_fs([
          do.logits for do in policy_outputs.distances])

      actor_logits = batch_fs(fs_actor_logits)
      actor_logits = utils.map_nt(lambda x: x[1:], actor_logits)

      actor_kl = self._compute_kl(actor_logits, policy_logits)

      metrics = {
          'post_update_actor_kl': actor_kl
      }
      return metrics

    post_update = jax_utils.with_compute_dtype(
        post_update, config.nash_policy_dtype.dtype)

    self.post_update = jax_utils.cached_partial(
        jax_utils.nnx_jit(post_update, donate_argnums=0),
        self.nash_policy,
    )

    train_kl_teacher_weights = jax_utils.nnx_jit(
        jax_utils.train_fn(self._unroll_kl_teacher_weights),
        donate_argnums=(0, 1),
    )
    self.train_kl_teacher_weights = jax_utils.cached_partial(
        train_kl_teacher_weights,
        self.kl_teacher_weights, self.kl_teacher_weights_optimizer,
    )

    @nnx.jit(donate_argnums=(0, 1))
    def update_policy(
      policy: Policy[Action],
      nash_policy: Policy[Action],
    ):
      nash_policy_state = nnx.state(nash_policy)
      policy_state = jax_utils.cast_floats_to_dtype(
          nash_policy_state, self.config.sample_policy_dtype.dtype)
      nnx.update(policy, policy_state)

    self._update_policy = jax_utils.cached_partial(
        update_policy, self.policy, self.nash_policy)

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    initial_states = {
        Q_FUNCTION: self.q_function.initial_state(batch_size, rngs),
        TEACHER: self.teacher.initial_state((batch_size, 2), rngs),
        NASH_POLICY: self.nash_policy.initial_state((batch_size, 2), rngs),
        # (sample) policy is also used by the actor
    }

    dtypes = {
        Q_FUNCTION: self.config.q_fn_dtype,
        TEACHER: self.config.teacher_dtype,
        NASH_POLICY: self.config.nash_policy_dtype,
    }

    for key, dtype in dtypes.items():
      initial_states[key] = jax_utils.cast_floats_to_dtype(initial_states[key], dtype.dtype)

    return initial_states

  def policy_variables(self):
    """Returns policy state for actor update via evaluators.update_variables."""
    return self.policy.get_state(to_numpy=False)

  def _sum_leaves(self, embedding: embed.Embedding[tp.Any, T], struct: T) -> jax.Array:
    return functools.reduce(jnp.add, embedding.flatten(struct))

  def _compute_kl(self, logits_p: Action, logits_q: Action) -> jax.Array:
    """Computes total KL(P||Q) summed over all controller components."""
    kls = self._controller_embedding.map(
        lambda e, lp, lq: e.kl_divergence(lp, lq),
        logits_p, logits_q)
    return self._sum_leaves(self._controller_embedding, kls)

  def _compute_entropy(self, logits: Action) -> jax.Array:
    """Computes total entropy H(P) summed over all controller components."""
    entropies = self._controller_embedding.map(
        lambda e, l: e.entropy(l), logits)
    return self._sum_leaves(self._controller_embedding, entropies)

  def _unroll_sample_policy(
      self,
      sample_policy: Policy[Action],
      rngs: nnx.Rngs,
      frames: Frames[nash_data.Rank3, Action],  # [T, B, 2]
      initial_states: RecurrentState,  # [B, 2]
  ) -> tuple[Loss, Metrics, RecurrentState, list[Action]]:
    action = frames.state_action.action
    prev_action = utils.map_nt(lambda t: t[:-1], action)

    # sample policy initial states come from the actor which might be in a different dtype
    initial_states = jax_utils.cast_floats_to_dtype(
        initial_states, self.config.sample_policy_dtype.dtype)
    sample_policy_outputs = sample_policy.unroll_with_outputs(frames, initial_states)

    # Because the action space is too large, we compute a finite subsample
    # using the sample_policy.

    @nnx.vmap(in_axes=(None, 0), out_axes=_SAMPLE_AXIS)
    def sample(sample_policy: Policy[Action], rngs: nnx.Rngs):
      sample_outputs = sample_policy.controller_head.sample(
          rngs=rngs,
          inputs=sample_policy_outputs.outputs,
          prev_controller_state=prev_action)
      return [so.controller_state for so in sample_outputs]

    policy_samples = sample(sample_policy, rngs.fork(split=self.num_samples))

    bm_loss = jnp.mean(sample_policy_outputs.imitation_loss, axis=[0, 2])
    bm_metrics = utils.map_single_structure(
      lambda x: jnp.mean(x, axis=0), sample_policy_outputs.metrics)

    return (
        bm_loss,
        bm_metrics,
        sample_policy_outputs.final_state,
        policy_samples,
    )

  def _unroll_q_function(
      self,
      q_function: q_lib.QFunction[Action],
      rngs: nnx.Rngs,
      frames: Frames[nash_data.Rank3, Action],  # [T, B, 2]
      initial_states: RecurrentState,  # [B, 2]
      policy_samples: list[Action],  # frame_skip x [S, T, B, 2]
      lambda_: float = 1.0,
  ) -> QFunctionOutputs:

    q_outputs, action_init_state, final_state, zs = q_function.loss_and_action_state(
        frames, initial_states, self.fs_discount, rngs=rngs, lambda_=lambda_,
        num_index_samples=self.config.num_index_samples)

    actions = policy_samples
    if self.config.include_action_taken_in_samples:
      actions = utils.map_nt(
        lambda samples, action_taken: jnp.concatenate(
          [samples, jnp.expand_dims(action_taken[1:], axis=_SAMPLE_AXIS)], axis=_SAMPLE_AXIS),
        policy_samples, frames.state_action.action)
    del policy_samples

    assert _SAMPLE_AXIS == 0
    # [N, S, S, T, B, 2]
    sample_q_values = q_function.multi_index_q_values_from_action_state(
        values=q_outputs.values,
        action_init_state=action_init_state,
        actions=actions,
        zs=zs,
        batch_size=self.config.sample_batch_size,
    )

    q_values = sample_q_values

    bm_loss = jnp.mean(q_outputs.loss, axis=[0, 2])

    payoff_matrices = nash_utils.mixed_payoff_matrices(q_values)  # [N, T, B, S, S]

    metrics = dict(
        q_outputs.metrics,
        information_fraction=jnp.mean(
            nash_utils.information_fraction(payoff_matrices), axis=0),  # [T, B]
    )

    bm_metrics = utils.map_single_structure(
      lambda x: jnp.mean(x, axis=0), metrics)

    return bm_loss, bm_metrics, final_state, q_outputs.values, action_init_state, q_values, zs

  def _compute_nash(
      self,
      q_values: jax.Array,  # [N, S, S, T, B, 2]
  ) -> tuple[nash.NashVariables, Metrics]:
    num_indices, s1, s2, t, b, n = q_values.shape
    assert n == 2

    payoff_matrices = nash_utils.mixed_payoff_matrices(q_values)  # [N, T, B, S, S]

    # Use separate vmaps over T and B to avoid an XLA SPMD partitioner
    # bug triggered by vmapping over the merged T*B sharded dimension.
    # Only triggered by qpax_fast, probably because it has matrices with
    # some dimensions equal to one.

    solve_vmap = jax_utils.multi_vmap(
        nash._solve_nash_simplex_impl, axes=[0, 1, 2])

    # One nash solution per epistemic index; leaves are [N, T, B, ...].
    nash_variables, nm_metrics = solve_vmap(payoff_matrices)

    nash_variables = utils.map_single_structure(
        lambda x: x.astype(jnp.float32), nash_variables)

    # Batch-major metrics; keep time and index dims so we can take max over num_steps.
    bm_metrics = utils.map_single_structure(
        lambda x: jnp.moveaxis(x, 2, 0), nm_metrics)  # [B, N, T, ...]

    bm_metrics['num_steps_max'] = jnp.max(bm_metrics['num_steps'], keepdims=True)

    return nash_variables, bm_metrics

  def _unroll_nash_policy(
      self,
      nash_policy: Policy[Action],
      rngs: nnx.Rngs,
      q_function: q_lib.QFunction[Action],
      teacher: Policy[Action],
      kl_teacher_weights: jax_utils.KLTeacherWeights,
      frames: Frames[nash_data.Rank3, Action],  # [T, B, 2]
      initial_states: RecurrentState,  # [B, 2]
      policy_samples: list[Action],  # FS x [S, T, B, 2]
      values: jax.Array,  # [N, T, B, 2]
      q_action_init_state: RecurrentState,  # [T, B, 2, H]
      q_values: jax.Array,  # [N, S, S, T, B, 2]
      zs: jax.Array,  # [N, B, 1, D_Z]
      nash_solution: nash.NashVariables,  # [N, T, B]
      teacher_core_outputs: jax.Array,  # [T, B, 2, O]
      fs_actor_logits: list[Action],  # FS x [T, B, 2]
  ) -> tuple[Loss, dict, RecurrentState]:

    metrics = dict()

    # Diagnostics are computed per epistemic index and averaged; the loss
    # regresses to the mixture of the per-index nash distributions.
    index_mean = lambda x: jnp.mean(x, axis=0)

    action = frames.state_action.action
    prev_action = utils.map_single_structure(lambda t: t[:-1], action)

    actions = policy_samples
    num_samples = self.num_samples

    if self.config.include_action_taken_in_samples:
      actions = utils.map_nt(
        lambda samples, action_taken: jnp.concatenate(
          [samples, jnp.expand_dims(action_taken[1:], axis=_SAMPLE_AXIS)], axis=_SAMPLE_AXIS),
        policy_samples, frames.state_action.action)
      num_samples += 1

    metrics['unique_fraction'] = nash_utils.compute_unique_fraction(actions)

    nash_probs = jnp.stack([nash_solution.p1, nash_solution.p2], axis=-2)  # [N, T, B, 2, S]
    nash_probs = nash_probs / jnp.sum(nash_probs, axis=-1, keepdims=True)  # re-normalize for numerical stability

    # The regression target is the mixture over epistemic indices.
    mixture_probs, indexed_metrics = nash_utils.indexed_nash_metrics(nash_probs)
    metrics.update(indexed_metrics)

    if self.config.include_action_taken_in_samples:
      metrics['action_taken_nash_prob'] = jax.lax.index_in_dim(
        mixture_probs, index=-1, axis=-1, keepdims=False)  # [T, B, 2]

    nash_values = jnp.stack([
        nash_solution.p1_nash_value, -nash_solution.p1_nash_value
    ], axis=-1)  # [N, T, B, 2]

    payoff_matrices = nash_utils.mixed_payoff_matrices(q_values)  # [N, T, B, S, S]
    diagnostics = nash_utils.nash_payoff_diagnostics(
        payoff_matrices, nash_probs, nash_values)
    metrics.update(diagnostics.metrics)
    nash_advantage = diagnostics.nash_advantage  # [T, B, 2]

    nash_policy_mbs = self.config.sample_batch_size

    # Save on computation by only training on the highest probability subsample.
    if self.config.subsample:
      if self.config.subsample > num_samples:
        raise ValueError(f'subsample {self.config.subsample} is greater than num_samples {num_samples}')

      # Select by mixture probability; subset the per-index distributions with
      # the same indices so the diagnostics below stay consistent.
      indices = jnp.argsort(
          mixture_probs, axis=-1, descending=True)[..., :self.config.subsample]
      mixture_probs = jnp.take_along_axis(mixture_probs, indices, axis=-1)
      mixture_probs = mixture_probs / jnp.sum(mixture_probs, axis=-1, keepdims=True)  # re-normalize

      nash_probs = jnp.take_along_axis(
          nash_probs, jnp.expand_dims(indices, 0), axis=-1)  # [N, T, B, 2, K]
      nash_probs = nash_probs / jnp.sum(nash_probs, axis=-1, keepdims=True)

      indices = jnp.moveaxis(indices, -1, _SAMPLE_AXIS)
      actions = utils.map_nt(
          lambda x: jnp.take_along_axis(x, indices, axis=_SAMPLE_AXIS),
          actions)
      num_samples = self.config.subsample

      if self.config.subsample < nash_policy_mbs:
        nash_policy_mbs = 0
      elif nash_policy_mbs > 0 and self.config.subsample % nash_policy_mbs != 0:
        raise ValueError(f'subsample {self.config.subsample} is not divisible by sample_batch_size {nash_policy_mbs}')

    nash_policy_outputs = nash_policy.unroll_with_outputs(
        frames, initial_states)
    nash_policy_imitation_loss = nash_policy_outputs.imitation_loss

    # Note that this inefficiently recomputes the controller head encoder
    # outputs for each sample.
    def nash_policy_distance_fn(nash_policy: Policy[Action], policy_sample: list[Action]):
      distances = nash_policy.controller_head.distance(
          inputs=nash_policy_outputs.outputs,
          prev_controller_state=prev_action,
          target_controller_state=policy_sample)
      return jax_utils.add_n(distances) / len(distances)

    if nash_policy_mbs > 0:
      nash_policy_distance_fn = nnx.remat(nash_policy_distance_fn)

    # [S, T, B, 2]
    nash_policy_log_probs = -jax_utils.lax_map_fn(
        nash_policy_distance_fn,
        input_batch_dims=(None, 0),
        output_batch_dims=0,
        microbatch_size=nash_policy_mbs,
    )(nash_policy, actions)

    nash_policy_log_probs = jnp.moveaxis(nash_policy_log_probs, _SAMPLE_AXIS, -1)  # [T, B, 2, S]
    # Cross-entropy to the mixture over epistemic indices of the per-index
    # nash distributions (the analog of the per-index argmax mixture in
    # q/rl_learner.py).
    nash_cross_entropy = -jnp.vecdot(mixture_probs, nash_policy_log_probs, axis=-1)  # [T, B, 2]

    if self.config.weight_by_advantage:
      # Weight the cross-entropy by how much better the nash distribution does
      # compared to the sample policy, i.e. how much we gain by using the nash
      # distribution for that state.
      nash_cross_entropy *= nash_advantage / nash_advantage.mean()

    # Estimate nash_policy vs computed nash
    nash_policy_sample_outputs = nash_policy.controller_head.sample(
        rngs=rngs,
        inputs=nash_policy_outputs.outputs,
        prev_controller_state=prev_action)
    nash_policy_samples = [  # list[Controller[T, B, 2]]
        so.controller_state for so in nash_policy_sample_outputs]

    q_function = jax_utils.cast_params_to_dtype(
        q_function, self.config.q_fn_dtype.dtype)

    # TODO: this is fairly inefficient -- we should instead pre-compute the
    # q-function's "outputs" on both the nash policy and the sampled actions,
    # the latter which we already have from the q-function unroll, and then use
    # QFunction._q_values_from_outputs.
    def compute_nash_policy_q_vs(
        q_function: q_lib.QFunction[Action],
        opponent_actions: list[Action],
    ) -> jax.Array:
      # Line up nash policy vs the other policy samples.
      def merge(nps: jax.Array, ps: jax.Array):
        # nps is [T, B, 2], ps is [T, B, 2]
        np1, np2 = jnp.unstack(nps, axis=2)
        p1, p2 = jnp.unstack(ps, axis=2)

        np1_vs_p2 = jnp.stack([np1, p2], axis=2)
        p1_vs_np2 = jnp.stack([p1, np2], axis=2)

        return jnp.stack([np1_vs_p2, p1_vs_np2], axis=0)  # [2, T, B, 2]

      merged_actions = utils.map_nt(  # [2, T, B, 2]
        merge, nash_policy_samples, opponent_actions)

      def q_fn(q_function: q_lib.QFunction[Action], actions: list[Action]):
        two_player_qs = q_function.indexed_q_values_from_action_state(
          values=values,
          action_init_state=q_action_init_state,
          actions=actions,
          zs=zs,
        )  # [N, T, B, 2]
        return p1_averaged_qs(two_player_qs)  # [N, T, B]

      merged_qs = nnx.vmap(q_fn, in_axes=(None, 0), out_axes=0)(q_function, merged_actions)  # [2, N, T, B]

      np1_vs_p2_qs, p1_vs_np2_qs = jnp.unstack(merged_qs, axis=0)  # [N, T, B]
      return jnp.stack([np1_vs_p2_qs, -p1_vs_np2_qs], axis=-1)  # [N, T, B, 2]

    # nash_policy vs the per-index nash distributions over sampled actions,
    # each evaluated under its own index's q-function. The module is passed
    # explicitly through each (nnx-aware) map so its graph state is lifted
    # properly; closing over it inside a raw jax transform breaks under grad.
    nash_policy_qs = jax_utils.lax_map_fn(  # [S, N, T, B, 2]
        compute_nash_policy_q_vs,
        microbatch_size=nash_policy_mbs,
        input_batch_dims=(None, 0),
        output_batch_dims=0,
    )(q_function, actions)
    nash_policy_qs = jnp.moveaxis(nash_policy_qs, 0, -1)  # [N, T, B, 2, S]
    nash_policy_vs_nash = jnp.vecdot(nash_policy_qs, jnp.flip(nash_probs, axis=-2))  # [N, T, B, 2]
    optimality_gap = nash_values - nash_policy_vs_nash

    mean_vs_nash = -jnp.flip(diagnostics.nash_vs_mean, axis=-1)  # [N, T, B, 2]
    nash_policy_advantage = nash_policy_vs_nash - mean_vs_nash

    metrics.update(
        nash_policy_vs_mean=index_mean(nash_policy_qs.mean(axis=-1)),
        nash_policy_vs_nash=index_mean(nash_policy_vs_nash),
        optimality_gap=index_mean(optimality_gap),  # nash-vs-nash - nash_policy-vs-nash
        nash_policy_advantage=index_mean(nash_policy_advantage),  # nash_policy-vs-nash - mean-vs-nash
    )

    nash_policy_logits = batch_fs([
        do.logits for do in nash_policy_outputs.distances])

    actor_logits = batch_fs(fs_actor_logits)
    actor_logits = utils.map_nt(lambda x: x[1:], actor_logits)

    # The exact KL between autoregressive policies is intractable because
    # later action components condition on earlier sampled ones. Instead we
    # sample actions from the "P" policy, condition both policies on them,
    # and take the analytic per-component KL, which is an unbiased
    # (Rao-Blackwellized) estimate of the true KL.

    # KL(nash_policy || teacher), sampling from the nash_policy.
    nash_policy_sample_logits = batch_fs(
        [so.logits for so in nash_policy_sample_outputs])
    teacher_on_nash_policy_samples = teacher.controller_head.distance_outputs(
        inputs=teacher_core_outputs,
        prev_controller_state=prev_action,
        target_controller_state=nash_policy_samples,
    )
    teacher_logits_on_nash_policy_samples = batch_fs(
        [do.logits for do in teacher_on_nash_policy_samples])
    teacher_kl = self._compute_kl(
        nash_policy_sample_logits, teacher_logits_on_nash_policy_samples)  # [T, FS, B, 2]

    # KL(teacher || nash_policy), sampling from the teacher.
    teacher_sample_outputs = teacher.controller_head.sample(
        rngs=rngs,
        inputs=teacher_core_outputs,
        prev_controller_state=prev_action)
    teacher_samples = [so.controller_state for so in teacher_sample_outputs]
    teacher_sample_logits = batch_fs(
        [so.logits for so in teacher_sample_outputs])
    nash_policy_on_teacher_samples = nash_policy.controller_head.distance_outputs(
        inputs=nash_policy_outputs.outputs,
        prev_controller_state=prev_action,
        target_controller_state=teacher_samples,
    )
    nash_policy_logits_on_teacher_samples = batch_fs(
        [do.logits for do in nash_policy_on_teacher_samples])
    reverse_teacher_kl = self._compute_kl(
        teacher_sample_logits, nash_policy_logits_on_teacher_samples)

    # The actor_kl is already such an estimate: the trajectory actions were
    # sampled from the actor, whose logits were recorded at sampling time, and
    # the nash_policy is teacher-forced on those same actions.
    actor_kl = self._compute_kl(actor_logits, nash_policy_logits)  # [T, FS, B, 2]
    # Like the KLs, the entropy conditions on prefixes sampled from the
    # nash_policy itself.
    entropy = self._compute_entropy(nash_policy_sample_logits)  # [T, FS, B, 2]

    def fs_mean(x: jax.Array) -> jax.Array:
      assert x.shape[1] == self.frame_skip
      return jnp.mean(x, axis=1)

    losses = [
        self.config.nash_weight * nash_cross_entropy,
        kl_teacher_weights.fwd_weight() * fs_mean(teacher_kl),
        kl_teacher_weights.bwd_weight() * fs_mean(reverse_teacher_kl),
    ]
    nash_policy_total_loss = jax_utils.add_n(losses)

    metrics.update(
        nash_cross_entropy=nash_cross_entropy,
        imitation_loss=nash_policy_imitation_loss,
        total_loss=nash_policy_total_loss,
        teacher_kl=teacher_kl,
        reverse_teacher_kl=reverse_teacher_kl,
        actor_kl=actor_kl,
        entropy=entropy,
    )

    bm_loss = jnp.mean(nash_policy_total_loss, axis=[0, 2])
    bm_metrics = utils.map_single_structure(
      lambda x: jnp.mean(x, axis=0), metrics)

    return bm_loss, bm_metrics, nash_policy_outputs.final_state

  def _unroll_kl_teacher_weights(
    self,
    kl_teacher_weights: jax_utils.KLTeacherWeights,
    teacher_kl: jax.Array,
    reverse_teacher_kl: jax.Array,
  ):
    fwd_weight = kl_teacher_weights.fwd_weight()
    bwd_weight = kl_teacher_weights.bwd_weight()

    # High weight lowers the KL, so if the KL is high, we want to increase the weight.
    fwd_loss = -fwd_weight * (jnp.mean(teacher_kl) - self.config.target_teacher_kl)
    bwd_loss = -bwd_weight * (jnp.mean(reverse_teacher_kl) - self.config.target_reverse_teacher_kl)
    total_loss = fwd_loss + bwd_loss

    metrics = dict(
        fwd_weight=fwd_weight,
        bwd_weight=bwd_weight,
        fwd_loss=fwd_loss,
        bwd_loss=bwd_loss,
        total_loss=total_loss,
    )

    return total_loss, metrics

  @jax_utils.annotate_function
  def step_sample_policy(
      self,
      tm_frames: Frames[nash_data.Rank3, Action], # [T, B, 2]
      initial_state: RecurrentState,
  ):
    return self.run_sample_policy(tm_frames, initial_state)

  @jax_utils.annotate_function
  def step_q_function(
      self,
      tm_frames: Frames[nash_data.Rank3, Action], # [T, B, 2]
      initial_state: RecurrentState,
      policy_samples: list[Action],
      train: bool,
  ):
    fn = self.train_q_function if train else self.run_q_function
    lambda_ = self.config.gae_lambda if train else 1.0
    return fn(tm_frames, initial_state, policy_samples, lambda_)

  @jax_utils.annotate_function
  def step_nash_policy(
      self,
      tm_frames: Frames[nash_data.Rank3, Action], # [T, B, 2]
      initial_state: RecurrentState,
      policy_samples: list[Action],  # frame_skip x [S, T, B, 2]
      values: jax.Array,  # [N, T, B, 2]
      q_action_init_state: RecurrentState,  # [T, B, 2, H]
      q_values: jax.Array,  # [N, S, S, T, B, 2]
      zs: jax.Array,  # [N, B, 1, D_Z]
      nash_solution: nash.NashVariables,  # [N, T, B]
      teacher_core_outputs: jax.Array,  # [T, B, 2, O]
      actor_logits: list[Action],  # FS x [T, B, 2]
      train: bool = True,
  ):
    fn = self.train_nash_policy if train else self.run_nash_policy

    return fn(
        tm_frames, initial_state, policy_samples, values, q_action_init_state,
        q_values, zs, nash_solution, teacher_core_outputs, actor_logits,
    )

  def step(
      self,
      # batch: nash_data.TwoPlayerBatch[Rank2],
      # tm_frames: Frames[nash_data.Rank3, Action], # [T, B, 2]
      trajectory: FrameSkipTrajectory[Action],  # [T, B, 2]
      initial_states: dict[str, RecurrentState],  # [B, 2]
      step: int,
      train: bool = True,
  ) -> tuple[dict, RecurrentState]:
    # TODO: take into account delay
    frames = get_frames(trajectory)
    frames = jax_utils.device_put(frames)

    final_states = dict(initial_states)
    metrics = {}

    (
      metrics[SAMPLE_POLICY],
      _,
      policy_samples,
    ) = self.step_sample_policy(
        frames, trajectory.initial_state)

    (
      metrics[Q_FUNCTION],
      final_states[Q_FUNCTION],
      values,
      q_action_init_state,
      q_values,
      zs,
    ) = self.step_q_function(
        frames, initial_states[Q_FUNCTION], policy_samples, train=train)

    (
      nash_variables,
      metrics[NASH],
    ) = self.compute_nash(q_values)

    (
      teacher_core_outputs,
      final_states[TEACHER],
    ) = self.run_teacher(frames, initial_states[TEACHER])

    actor_logits = [so.logits for so in trajectory.actions]
    # Need to make a copy since the original one gets donated
    initial_nash_state = copy_struct(initial_states[NASH_POLICY])
    (
      metrics[NASH_POLICY],
      final_states[NASH_POLICY],
    ) = self.step_nash_policy(
        frames, initial_states[NASH_POLICY], policy_samples, values,
        q_action_init_state, q_values, zs, nash_variables, teacher_core_outputs,
        actor_logits, train=train)

    post_update_metrics = self.post_update(
        frames, initial_nash_state, actor_logits)
    metrics[NASH_POLICY].update(post_update_metrics)

    for path, value in jax.tree.leaves_with_path(jax.device_get(post_update_metrics)):
      if np.any(np.isnan(value)):
        raise ValueError(f'NaN in post_update_metrics at {path}')

    if train:
      metrics['kl_teacher_weights'] = self.train_kl_teacher_weights(
          teacher_kl=metrics[NASH_POLICY]['teacher_kl'],
          reverse_teacher_kl=metrics[NASH_POLICY]['reverse_teacher_kl'],
      )[0]

    if train and step % self.config.epoch_length == 0:
      self._update_policy()

    return metrics, final_states
