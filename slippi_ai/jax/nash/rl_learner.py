import dataclasses
import functools
import logging
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from slippi_ai import utils
from slippi_ai.types import S, Frames, Action, StateAction
from slippi_ai.jax.policies import Policy, RecurrentState, DistanceOutputs
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
  gae_lambda: float = 1.0

  num_samples: int = 1
  sample_batch_size: int = 0  # 0 means full batch size, i.e. vmap
  include_action_taken_in_samples: bool = True
  subsample: tp.Optional[int] = None
  epoch_length: int = 100

  nash_weight: float = 1
  weight_by_advantage: bool = False
  kl_teacher_weight: float = 0
  reverse_kl_teacher_weight: float = 0

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
    Values,  # [T, B, 2]
    RecurrentState,  # action_init_state [T, B, 2, H]
    QValues,  # [S, S, T, B, 2]
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

    self._controller_embedding = self.policy.controller_head.controller_embedding

    self.discount = rl_lib.discount_from_halflife(
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

    unroll_sample_policy = jax_utils.with_compute_dtype(
      self._unroll_sample_policy, config.sample_policy_dtype.dtype)

    self.run_sample_policy = jax_utils.cached_partial(
        jax_utils.nnx_jit(
            jax_utils.no_loss(unroll_sample_policy),
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

    self.train_q_function = jax_utils.cached_train_fn(
        module=self.q_function,
        optimizer=self.q_function_optimizer,
        loss_fn=unroll_q_function,
    )

    self.run_q_function = jax_utils.cached_partial(
        jax_utils.nnx_jit(
            jax_utils.no_loss(unroll_q_function),
            donate_argnums=(0, 2),
        ),
        self.q_function,
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
    ) -> tuple[list[DistanceOutputs[Action]], RecurrentState]:
      teacher_outputs = teacher.unroll(frames, initial_states)
      return teacher_outputs.distances, teacher_outputs.final_state

    unroll_teacher = jax_utils.with_compute_dtype(
        unroll_teacher, config.teacher_dtype.dtype)

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
        jax_utils.nnx_jit(train_nash_policy, donate_argnums=(0, 1, 2, 3, 5)),
        self.nash_policy, self.policy_optimizer, rngs, self.q_function,
    )

    self.run_nash_policy = jax_utils.cached_partial(
        jax_utils.nnx_jit(
            jax_utils.no_loss(unroll_nash_policy),
            donate_argnums=(0, 1, 2, 4),
        ),
        self.nash_policy, rngs, self.q_function,
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
        jax_utils.nnx_jit(post_update),
        self.nash_policy,
    )

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
    return self.policy.get_state()

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
      frames: Frames[nash_data.Rank3, Action],  # [T, B, 2]
      initial_states: RecurrentState,  # [B, 2]
      policy_samples: list[Action],  # frame_skip x [S, T, B, 2]
      lambda_: float = 1.0,
  ) -> tuple[Loss, Metrics, RecurrentState, Values, RecurrentState, QValues]:

    q_outputs, action_init_state, final_state = q_function.loss_and_action_state(
        frames, initial_states, self.discount, lambda_=lambda_)

    actions = policy_samples
    if self.config.include_action_taken_in_samples:
      actions = utils.map_nt(
        lambda samples, action_taken: jnp.concatenate(
          [samples, jnp.expand_dims(action_taken[1:], axis=_SAMPLE_AXIS)], axis=_SAMPLE_AXIS),
        policy_samples, frames.state_action.action)
    del policy_samples

    assert _SAMPLE_AXIS == 0
     # [S, S, T, B, 2]
    sample_q_values = q_function.multi_q_values_from_action_state(
        values=q_outputs.values,
        action_init_state=action_init_state,
        actions=actions,
        batch_size=self.config.sample_batch_size,
    )

    q_values = sample_q_values

    bm_loss = jnp.mean(q_outputs.loss, axis=[0, 2])

    metrics = dict(
        q_outputs.metrics,
        information_fraction=nash_utils.information_fraction(q_values),
    )

    bm_metrics = utils.map_single_structure(
      lambda x: jnp.mean(x, axis=0), metrics)

    return bm_loss, bm_metrics, final_state, q_outputs.values, action_init_state, q_values

  def _compute_nash(
      self,
      q_values: jax.Array,  # [S, S, T, B, 2]
  ) -> tuple[nash.NashVariables, Metrics]:
    s1, s2, t, b, n = q_values.shape
    assert n == 2

    p1_qs, p2_qs = jnp.unstack(q_values, axis=-1)  # [S, S, T, B]
    mixed_values = (p1_qs - p2_qs) / 2  # [S, S, T, B]

    payoff_matrices = jnp.moveaxis(mixed_values, (0, 1), (-2, -1))  # [T, B, S, S]

    # Use separate vmaps over T and B to avoid an XLA SPMD partitioner
    # bug triggered by vmapping over the merged T*B sharded dimension.
    # Only triggered by qpax_fast, probably because it has matrices with
    # some dimensions equal to one.

    solve_vmap = jax_utils.multi_vmap(nash._solve_nash_simplex_impl, axes=[0, 1])

    nash_variables, tm_metrics = solve_vmap(payoff_matrices)

    nash_variables = utils.map_single_structure(
        lambda x: x.astype(jnp.float32), nash_variables)

    # Keep time dim so we can take max over num_steps
    bm_metrics = utils.map_single_structure(
        lambda x: jnp.swapaxes(x, 0, 1), tm_metrics)

    bm_metrics['num_steps_max'] = jnp.max(bm_metrics['num_steps'], keepdims=True)

    return nash_variables, bm_metrics

  def _unroll_nash_policy(
      self,
      nash_policy: Policy[Action],
      rngs: nnx.Rngs,
      q_function: q_lib.QFunction[Action],
      frames: Frames[nash_data.Rank3, Action],  # [T, B, 2]
      initial_states: RecurrentState,  # [B, 2]
      policy_samples: list[Action],  # FS x [S, T, B, 2]
      values: jax.Array,  # [T, B, 2]
      q_action_init_state: RecurrentState,  # [T, B, 2, H]
      q_values: jax.Array,  # [S, S, T, B, 2]
      nash_solution: nash.NashVariables,  # [T, B]
      teacher_outputs: list[DistanceOutputs[Action]],  # FS x [T, B, 2]
      fs_actor_logits: list[Action],  # FS x [T, B, 2]
  ) -> tuple[Loss, dict, RecurrentState]:

    metrics = dict()

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

    nash_probs = jnp.stack([nash_solution.p1, nash_solution.p2], axis=-2)  # [T, B, 2, S]
    nash_probs = nash_probs / jnp.sum(nash_probs, axis=-1, keepdims=True)  # re-normalize for numerical stability
    metrics['nash_entropy'] = jax_utils.entropy(nash_probs, axis=-1)  # [T, B, 2]

    nash_values = jnp.stack([
        nash_solution.p1_nash_value, -nash_solution.p1_nash_value
    ], axis=-1)  # [T, B, 2]

    p1_qs, p2_qs = jnp.unstack(q_values, axis=-1)  # [S, S, T, B]
    mixed_values = (p1_qs - p2_qs) / 2  # [S, S, T, B]
    payoff_matrices = jnp.moveaxis(mixed_values, (0, 1), (-2, -1))  # [T, B, S, S]
    p12_matrices = jnp.stack([
        payoff_matrices,
        -payoff_matrices.swapaxes(-1, -2)],
    axis=2)  # [T, B, 2, S, S]

    def payoffs(
      p: jax.Array,  # [T, B, 2, S]
      q: jax.Array,  # [T, B, 2, S]
    ) -> jax.Array:  # [T, B, 2]
      """Compute payoffs of policy p vs policy q."""
      return jnp.vecdot(p, jnp.matvec(p12_matrices, jnp.flip(q, axis=-2)))

    vs_mean = p12_matrices.mean(axis=-1)  # [T, B, 2, S]
    argmax_policy = jnp.argmax(vs_mean, axis=-1)  # [T, B, 2]
    argmax_policy_probs = jax.nn.one_hot(argmax_policy, num_classes=num_samples)  # [T, B, 2, S]
    argmax_vs_mean = jnp.max(vs_mean, axis=-1)  # [T, B, 2]

    nash_vs_mean = jnp.vecdot(nash_probs, vs_mean)  # [T, B, 2]
    argmax_advantage = argmax_vs_mean - nash_vs_mean

    nash_vs_argmax = payoffs(nash_probs, argmax_policy_probs)
    nash_vs_argmax_advantage = nash_vs_argmax - nash_values

    nash_advantage = nash_vs_mean - nash_values
    nash_advantage_std = jnp.std(nash_advantage, keepdims=True)
    nash_advantage_variation = nash_advantage_std / jnp.mean(nash_advantage)
    nash_advantantage_min = jnp.min(nash_advantage, keepdims=True)

    # Test nash solution; should maybe go in the nash computation itself
    nash_vs_nash = payoffs(nash_probs, nash_probs)
    nash_value_error = jnp.sqrt(jnp.square(nash_vs_nash - nash_values).mean(keepdims=True))
    nash_value_error_max = jnp.max(jnp.abs(nash_vs_nash - nash_values), keepdims=True)
    vs_nash = -jnp.vecmat(nash_probs, p12_matrices)  # [T, B, 2, S]
    best_vs_nash = jnp.max(vs_nash, axis=-1)  # [T, B, 2]
    nash_suboptimality = best_vs_nash - nash_vs_nash
    nash_suboptimality_max = jnp.max(nash_suboptimality, keepdims=True)

    metrics.update(
        nash_advantage=nash_advantage,  # nash-vs-mean - nash-vs-nash
        nash_advantage_std=nash_advantage_std,
        nash_advantage_variation=nash_advantage_variation,
        nash_advantantage_min=nash_advantantage_min,
        argmax_advantage=argmax_advantage,  # argmax-vs-mean - nash-vs-mean
        nash_vs_argmax_advantage=nash_vs_argmax_advantage,  # nash-vs-argmax - nash-vs-nash
        nash_value_error=nash_value_error,
        nash_value_error_max=nash_value_error_max,
        nash_suboptimality=nash_suboptimality,
        nash_suboptimality_max=nash_suboptimality_max,
    )

    nash_policy_mbs = self.config.sample_batch_size

    # Save on computation by only training on the highest probability subsample.
    if self.config.subsample:
      if self.config.subsample > num_samples:
        raise ValueError(f'subsample {self.config.subsample} is greater than num_samples {num_samples}')

      indices = jnp.argsort(
          nash_probs, axis=-1, descending=True)[..., :self.config.subsample]
      nash_probs = jnp.take_along_axis(nash_probs, indices, axis=-1)
      nash_probs = nash_probs / jnp.sum(nash_probs, axis=-1, keepdims=True)  # re-normalize

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
    nash_cross_entropy = -jnp.vecdot(nash_probs, nash_policy_log_probs, axis=-1)  # [T, B, 2]

    if self.config.weight_by_advantage:
      # Weight the cross-entropy by how much better the nash distribution does
      # compared to the sample policy, i.e. how much we gain by using the nash
      # distribution for that state.
      nash_cross_entropy *= nash_advantage / nash_advantage.mean()

    # Estimate nash_policy vs computed nash
    nash_policy_samples = [  # list[Controller[T, B, 2]]
        so.controller_state
        for so in nash_policy.controller_head.sample(
            rngs=rngs,
            inputs=nash_policy_outputs.outputs,
            prev_controller_state=prev_action)]

    q_function = jax_utils.cast_params_to_dtype(
        q_function, self.config.q_fn_dtype.dtype)

    # TODO: this is fairly inefficient -- we should instead pre-compute the
    # q-function's "outputs" on both the nash policy and the sampled actions,
    # the latter which we already have from the q-function unroll, and then use
    # QFunction._q_values_from_outputs.
    def compute_nash_policy_q_vs(opponent_actions: list[Action]) -> jax.Array:
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

      def q_fn(actions: list[Action]):
        two_player_qs = q_function.q_values_from_action_state(
          values=values,
          action_init_state=q_action_init_state,
          actions=actions,
        )
        return p1_averaged_qs(two_player_qs)  # [T, B]

      q_values = jax.vmap(q_fn, in_axes=0, out_axes=0)(merged_actions)  # [2, T, B]

      np1_vs_p2_qs, p1_vs_np2_qs = jnp.unstack(q_values, axis=0)  # [T, B], [T, B]
      return jnp.stack([np1_vs_p2_qs, -p1_vs_np2_qs], axis=-1)  # [T, B, 2]

    nash_policy_qs = jax_utils.lax_map(  # [S, T, B, 2]
        compute_nash_policy_q_vs, actions,
        batch_size=nash_policy_mbs,
    )
    nash_policy_qs = jnp.moveaxis(nash_policy_qs, 0, -1)  # [T, B, 2, S]
    nash_policy_qs = jnp.vecdot(nash_policy_qs, nash_probs)  # [T, B, 2]
    optimality_gap = nash_values - nash_policy_qs

    mean_vs_nash = -jnp.flip(nash_vs_mean, axis=-1)
    nash_policy_advantage = nash_policy_qs - mean_vs_nash

    metrics.update(
        optimality_gap=optimality_gap,  # nash-vs-nash - nash_policy-vs-nash
        nash_policy_advantage=nash_policy_advantage,  # nash_policy-vs-nash - mean-vs-nash
    )

    nash_policy_logits = batch_fs([
        do.logits for do in nash_policy_outputs.distances])

    teacher_logits = batch_fs([
        do.logits for do in teacher_outputs
    ])

    actor_logits = batch_fs(fs_actor_logits)
    actor_logits = utils.map_nt(lambda x: x[1:], actor_logits)

    teacher_kl = self._compute_kl(nash_policy_logits, teacher_logits)  # [T, FS, B, 2]
    reverse_teacher_kl = self._compute_kl(teacher_logits, nash_policy_logits)  # [T, FS, B, 2]
    actor_kl = self._compute_kl(actor_logits, nash_policy_logits)  # [T, FS, B, 2]
    entropy = self._compute_entropy(nash_policy_logits)  # [T, FS, B, 2]

    def fs_mean(x: jax.Array) -> jax.Array:
      assert x.shape[1] == self.frame_skip
      return jnp.mean(x, axis=1)

    losses = [
        self.config.nash_weight * nash_cross_entropy,
        self.config.kl_teacher_weight * fs_mean(teacher_kl),
        self.config.reverse_kl_teacher_weight * fs_mean(reverse_teacher_kl),
    ]
    nash_policy_total_loss = jax_utils.add_n(losses)

    metrics.update(
        nash_cross_entropy=nash_cross_entropy,
        nash_policy_qs=nash_policy_qs,
        imitation_loss=nash_policy_imitation_loss,
        total_loss=nash_policy_total_loss,
        teacher_kl=teacher_kl,
        reverse_teacher_kl=reverse_teacher_kl,
        actor_kl=actor_kl,
        entropy=entropy,
    )

    if self.config.include_action_taken_in_samples:
      metrics['action_taken_nash_prob'] = jax.lax.index_in_dim(
        nash_probs, index=-1, axis=-1, keepdims=False)  # [T, B, 2]

    bm_loss = jnp.mean(nash_policy_total_loss, axis=[0, 2])
    bm_metrics = utils.map_single_structure(
      lambda x: jnp.mean(x, axis=0), metrics)

    return bm_loss, bm_metrics, nash_policy_outputs.final_state

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
    return fn(tm_frames, initial_state, policy_samples, lambda_=lambda_)

  @jax_utils.annotate_function
  def step_nash_policy(
      self,
      tm_frames: Frames[nash_data.Rank3, Action], # [T, B, 2]
      initial_state: RecurrentState,
      policy_samples: list[Action],  # frame_skip x [S, T, B, 2]
      values: jax.Array,  # [T, B, 2]
      q_action_init_state: RecurrentState,  # [T, B, 2, H]
      q_values: jax.Array,  # [S, S, T, B, 2]
      nash_solution: nash.NashVariables,  # [T, B]
      teacher_outputs: list[DistanceOutputs[Action]],  # FS x [T, B, 2]
      actor_logits: list[Action],  # FS x [T, B, 2]
      train: bool = True,
  ):
    fn = self.train_nash_policy if train else self.run_nash_policy

    return fn(
        tm_frames, initial_state, policy_samples, values, q_action_init_state,
        q_values, nash_solution, teacher_outputs, actor_logits,
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
    ) = self.step_q_function(
        frames, initial_states[Q_FUNCTION], policy_samples, train=train)

    (
      nash_variables,
      metrics[NASH],
    ) = self.compute_nash(q_values)

    (
      teacher_outputs,
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
        q_action_init_state, q_values, nash_variables, teacher_outputs, actor_logits,
        train=train)

    post_update_metrics = self.post_update(
        frames, initial_nash_state, actor_logits)
    metrics[NASH_POLICY].update(post_update_metrics)

    if train and step % self.config.epoch_length == 0:
      jax_utils.set_module_state(self.policy, jax_utils.get_module_state(self.nash_policy))

    return metrics, final_states
