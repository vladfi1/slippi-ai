import contextlib
import dataclasses
import logging
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from slippi_ai import utils
from slippi_ai.types import S, Frames, Action
from slippi_ai.data import Rank2
from slippi_ai.jax.policies import Policy, RecurrentState
from slippi_ai.jax import embed, rl_lib, jax_utils
from slippi_ai.jax.jax_utils import PS, DATA_AXIS
from slippi_ai.nash import data as nash_data
from slippi_ai.jax.nash import (
    utils as nash_utils,
    q_function as q_lib,
)
from slippi_ai.jax.nash import nash
from slippi_ai.jax.agents import DType

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  reward_halflife: float = 4  # only for q_function metrics

  num_samples: int = 1
  sample_batch_size: int = 0  # 0 means full batch size, i.e. vmap
  include_action_taken_in_samples: bool = True

  nash_error: float = 1e-3

  nash_weight: float = 1
  imitation_weight: float = 0

  nash_solver: str = 'simplex'

  sample_policy_dtype: DType = DType.FP32
  nash_policy_dtype: DType = DType.FP32
  # q_fn_dtype: DType = DType.FP32

  compute_nash_policy_qs: bool = True

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


class Learner(nnx.Module, tp.Generic[Action]):

  def __init__(
      self,
      config: LearnerConfig,
      q_function: q_lib.QFunction[Action],
      sample_policy: Policy[Action],  # trained via imitation
      nash_policy: Policy[Action],
      rngs: nnx.Rngs,  # used for sampling
      mesh: jax.sharding.Mesh,
      explicit_pmean: bool = False,
      smap_optimizer: bool = True,
      nash_policy_optimizer_state: tp.Optional[tp.Any] = None,
  ):
    self.config = config
    self.q_function = q_function
    self.sample_policy = sample_policy
    self.nash_policy = nash_policy

    self.discount = rl_lib.discount_from_halflife(
      config.reward_halflife / q_function.frame_skip)

    learning_rate = config.learning_rate

    self.nash_policy_optimizer = nnx.Optimizer(
        nash_policy, optax.adam(learning_rate), wrt=nnx.Param)

    if nash_policy_optimizer_state is not None:
      jax_utils.set_module_state(self.nash_policy_optimizer, nash_policy_optimizer_state)

    if not config.include_action_taken_in_samples and config.num_samples < 2:
      raise ValueError('num_samples must be at least 2 if not including action taken in samples')

    if config.sample_batch_size > 0:
      ns = config.num_samples
      if config.include_action_taken_in_samples:
        ns += 1
      if ns % config.sample_batch_size != 0:
        logging.warning(f'sample_batch_size {config.sample_batch_size} does not divide num_samples {ns}')

    self.num_samples = config.num_samples

    self.delay = nash_policy.delay
    assert sample_policy.delay == self.delay

    jax_utils.replicate_module(self, mesh)

    sharding_kwargs = ShardingKwargs(
        mesh=mesh,
        explicit_pmean=explicit_pmean,
        smap_optimizer=smap_optimizer,
    )

    BM = PS(DATA_AXIS)
    tms_specs = [None, DATA_AXIS]
    TM = PS(*tms_specs)  # time-major
    tms_specs.insert(_SAMPLE_AXIS, None)
    TMS = PS(*tms_specs)  # time-major with samples
    tms_specs.insert(_SAMPLE_AXIS, None)
    TMSS = PS(*tms_specs)  # time-major SxS

    policy_samples = TMS
    vs = TM
    q_action_init = TM
    qs = TMSS
    nash_solution = TM
    metrics = BM

    sample_policy_specs = ShardingSpecs(
        extra_in_specs=None,
        extra_out_specs=(policy_samples,),
    )

    unroll_sample_policy = jax_utils.with_compute_dtype(
        self._unroll_sample_policy, config.sample_policy_dtype.dtype)

    self.run_sample_policy = jax_utils.shard_map_loss_fn_with_rngs(
        module=self.sample_policy,
        rngs=rngs,
        loss_fn=unroll_sample_policy,
        mesh=mesh,
        **sample_policy_specs,
    )

    q_function_specs = ShardingSpecs(
        extra_in_specs=(policy_samples,),
        extra_out_specs=(vs, q_action_init, qs),
    )

    # Keep q_function in fp32 so we can distinguish small differences in
    # q-values that lead to different nash solutions.
    unroll_q_function = self._unroll_q_function
    # if config.bf16:
    #   unroll_q_function = jax_utils.with_bf16_compute(unroll_q_function)

    self.run_q_function = jax_utils.shard_map_loss_fn(
        module=self.q_function,
        loss_fn=unroll_q_function,
        mesh=mesh,
        **q_function_specs,
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
      in_shardings=jax.NamedSharding(mesh, qs),
      out_shardings=(jax.NamedSharding(mesh, nash_solution), jax.NamedSharding(mesh, metrics)),
    )
    self.compute_nash = jax.profiler.annotate_function(self.compute_nash)

    nash_policy_specs = ShardingSpecs(
        extra_in_specs=(policy_samples, vs, q_action_init, nash_solution),
        extra_out_specs=None,
    )
    unroll_nash_policy = jax_utils.with_compute_dtype(
        self._unroll_nash_policy, config.nash_policy_dtype.dtype)

    self.train_nash_policy = jax_utils.data_parallel_train_with_rngs(
        module=self.nash_policy,
        optimizer=self.nash_policy_optimizer,
        rngs=rngs,
        loss_fn=unroll_nash_policy,
        **sharding_kwargs,
        **nash_policy_specs,
    )

    self.run_nash_policy = jax_utils.shard_map_loss_fn_with_rngs(
        module=self.nash_policy,
        rngs=rngs,
        loss_fn=unroll_nash_policy,
        mesh=mesh,
        **nash_policy_specs,
    )

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    nash_policy_state = jax_utils.cast_floats_to_dtype(
        self.nash_policy.initial_state((batch_size, 2), rngs),
        self.config.nash_policy_dtype.dtype)
    sample_policy_state = jax_utils.cast_floats_to_dtype(
        self.sample_policy.initial_state((batch_size, 2), rngs),
        self.config.sample_policy_dtype.dtype)

    state = {
        NASH_POLICY: nash_policy_state,
        SAMPLE_POLICY: sample_policy_state,
    }

    # q_function is in fp32
    state[Q_FUNCTION] = self.q_function.initial_state(batch_size, rngs)

    return state

  def _get_delayed_frames(self, frames: Frames[S, Action]) -> Frames[S, Action]:
    assert self.delay == 0
    return frames

    state_action = frames.state_action
    # Includes "overlap" frame.
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
        # Only use rewards that follow actions.
        reward=frames.reward[self.delay:],
    )

  def _unroll_sample_policy(
      self,
      sample_policy: Policy[Action],
      bm_frames: Frames[nash_data.Rank3, Action],
      initial_states: RecurrentState,
      rngs: nnx.Rngs,
  ) -> tuple[Loss, Metrics, RecurrentState, list[Action]]:
    frames = nash_utils.bm_to_tm(bm_frames)
    frames = self._get_delayed_frames(frames)

    action = frames.state_action.action
    prev_action = jax.tree.map(lambda t: t[:-1], action)

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
      bm_frames: Frames[nash_data.Rank3, Action],
      initial_states: RecurrentState,
      policy_samples: list[Action],
  ) -> tuple[Loss, Metrics, RecurrentState, Values, RecurrentState, QValues]:
    frames = nash_utils.bm_to_tm(bm_frames)
    frames = self._get_delayed_frames(frames)

    q_outputs, action_init_state, final_state = q_function.loss_and_action_state(
        frames, initial_states, self.discount)

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
    assert s1 == s2
    assert n == 2

    p1_qs, p2_qs = jnp.unstack(q_values, axis=-1)  # [S, S, T, B]
    mixed_values = (p1_qs - p2_qs) / 2  # [S, S, T, B]

    payoff_matrices = jnp.moveaxis(mixed_values, (0, 1), (-2, -1))  # [T, B, S, S]

    with contextlib.ExitStack() as stack:
      is_ip_solver = self.config.nash_solver in ['qpax', 'qpax_fast', 'ippd']

      if is_ip_solver:
        stack.enter_context(jax.enable_x64())

        payoff_matrices = payoff_matrices.astype(jnp.float64)
        assert payoff_matrices.dtype == jnp.float64
      else:
        payoff_matrices = payoff_matrices.astype(jnp.float32)

      if self.config.nash_solver == 'qpax':
        solver = nash._solve_zero_sum_nash_qpax
      elif self.config.nash_solver == 'qpax_fast':
        solver = nash._solve_zero_sum_nash_qpax_fast
      elif self.config.nash_solver == 'ippd':
        solver = nash._solve_zero_sum_nash_ippd
      elif self.config.nash_solver == 'simplex':
        solver = nash._solve_nash_simplex_impl
      else:
        raise ValueError(f'Unknown nash_solver {self.config.nash_solver}')

      solver_kwargs = {}
      if is_ip_solver:
        solver_kwargs['error'] = self.config.nash_error

      # Use separate vmaps over T and B to avoid an XLA SPMD partitioner
      # bug triggered by vmapping over the merged T*B sharded dimension.
      # Only triggered by qpax_fast, probably because it has matrices with
      # some dimensions equal to one.
      # https://github.com/jax-ml/jax/issues/35815
      def solve_one(pm: nash.PayoffMatrix):
        return solver(pm, **solver_kwargs)

      solve_vmap = jax_utils.multi_vmap(solve_one, axes=[0, 1])

      nash_variables, tm_metrics = solve_vmap(payoff_matrices)
      assert isinstance(nash_variables, nash.NashVariables)

    nash_variables = utils.map_single_structure(
        lambda x: x.astype(jnp.float32), nash_variables)

    ps = jnp.stack([nash_variables.p1, nash_variables.p2], axis=-2)  # [T, B, 2, S]
    assert ps.shape == (t, b, 2, s1)
    ps = ps / jnp.sum(ps, axis=-1, keepdims=True)  # re-normalize for numerical stability

    ps_stats = {}
    sorted_ps = jnp.sort(ps, descending=True, axis=-1)
    cumsum_ps = jnp.cumsum(sorted_ps, axis=-1)

    for count in range(1, min(s1 + 1, 6)):
      count_stats = {}
      mass_covered = cumsum_ps[..., count-1]
      count_stats['mean'] = mass_covered

      cutoff_stats = {}
      for cutoff in [0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.5]:
        cutoff_stats[cutoff] = mass_covered >= cutoff
      count_stats['above'] = cutoff_stats

      ps_stats[count] = count_stats

    tm_metrics['ps'] = ps_stats

    nash_entropy = jax_utils.entropy(ps, axis=-1)  # [T, B, 2]
    tm_metrics['entropy'] = nash_entropy
    entropy_stats = {}

    for cutoff in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.4, 2]:
      # Mean will be taken later
      entropy_stats[cutoff] = nash_entropy > cutoff

    tm_metrics['entropy_above'] = entropy_stats

    # Keep time dim so we can take max over num_steps
    bm_metrics = utils.map_single_structure(
        lambda x: jnp.swapaxes(x, 0, 1), tm_metrics)

    bm_metrics['sample_payoff_matrix'] = payoff_matrices[0].astype(jnp.float32)  # [B, S1, S2]

    return nash_variables, bm_metrics


  def _unroll_nash_policy(
      self,
      nash_policy: Policy[Action],
      bm_frames: Frames[nash_data.Rank3, Action],
      initial_states: RecurrentState,
      rngs: nnx.Rngs,
      policy_samples: list[Action],  # frame_skip x [S, T, B, 2]
      values: jax.Array,  # [T, B, 2]
      q_action_init_state: RecurrentState,  # [T, B, 2, H]
      nash_solution: nash.NashVariables,  # [T, B]
  ) -> tuple[Loss, dict, RecurrentState]:
    frames = nash_utils.bm_to_tm(bm_frames)
    frames = self._get_delayed_frames(frames)

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

    # TODO: this could belong in the sample policy unroll
    unique_fraction = nash_utils.compute_unique_fraction(actions)

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

    if self.config.sample_batch_size > 0:
      batch_distance_fn = jax_utils.lax_map_fn(
          nnx.remat(nash_policy_distance_fn),
          microbatch_size=self.config.sample_batch_size,
          input_batch_dims=(None, 0),
          output_batch_dims=0,
      )
    else:
      batch_distance_fn = nnx.vmap(
          nash_policy_distance_fn,
          in_axes=(None, 0), out_axes=0,
      )

    nash_policy_log_probs = -batch_distance_fn(nash_policy, actions)

    nash_policy_log_probs = jnp.moveaxis(nash_policy_log_probs, _SAMPLE_AXIS, -1)  # [T, B, 2, S]

    nash_probs = jnp.stack([nash_solution.p1, nash_solution.p2], axis=-2)  # [T, B, 2, S]
    nash_probs = nash_probs / jnp.sum(nash_probs, axis=-1, keepdims=True)  # re-normalize for numerical stability
    nash_entropy = jax_utils.entropy(nash_probs, axis=-1)  # [T, B, 2]

    nash_cross_entropy = -jnp.vecdot(nash_probs, nash_policy_log_probs, axis=-1)  # [T, B, 2]

    losses = [
        self.config.nash_weight * nash_cross_entropy,
        self.config.imitation_weight * nash_policy_imitation_loss,
    ]
    nash_policy_total_loss = jax_utils.add_n(losses)

    metrics = dict(
        nash_entropy=nash_entropy,
        nash_cross_entropy=nash_cross_entropy,
        imitation_loss=nash_policy_imitation_loss,
        total_loss=nash_policy_total_loss,
        unique_fraction=unique_fraction,
    )

    if self.config.compute_nash_policy_qs:
      # Estimate nash_policy vs computed nash
      nash_policy_samples = [  # list[Controller[T, B, 2]]
          so.controller_state
          for so in nash_policy.controller_head.sample(
              rngs=rngs,
              inputs=nash_policy_outputs.outputs,
              prev_controller_state=prev_action)]

      # Technically q_function should be an input to this method.
      q_function = self.q_function  # stays in fp32 even if nash_policy is in bf16

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
          batch_size=self.config.sample_batch_size,
      )
      nash_policy_qs = jnp.moveaxis(nash_policy_qs, 0, -1)  # [T, B, 2, S]
      nash_policy_qs = jnp.vecdot(nash_policy_qs, nash_probs)  # [T, B, 2]

      metrics['nash_policy_qs'] = nash_policy_qs

    if self.config.include_action_taken_in_samples:
      metrics['action_taken_nash_prob'] = jax.lax.index_in_dim(
        nash_probs, index=-1, axis=-1, keepdims=False)  # [T, B, 2]

    bm_loss = jnp.mean(nash_policy_total_loss, axis=[0, 2])
    bm_metrics = utils.map_single_structure(
      lambda x: jnp.mean(x, axis=0), metrics)

    return bm_loss, bm_metrics, nash_policy_outputs.final_state

  @jax.profiler.annotate_function
  def step_sample_policy(
      self,
      zipped_frames: nash_data.ZippedFrames,  # [B, 2, T]
      initial_state: RecurrentState,
  ):
    frames = Frames[nash_data.Rank3, Action](
        state_action=self.sample_policy.network.encode(zipped_frames.state_action),
        is_resetting=zipped_frames.is_resetting,
        reward=zipped_frames.reward,
    )

    return self.run_sample_policy(frames, initial_state)

  @jax.profiler.annotate_function
  def step_q_function(
      self,
      zipped_frames: nash_data.ZippedFrames,  # [B, 2, T]
      initial_state: RecurrentState,
      policy_samples: list[Action],
  ):
    frames = Frames[nash_data.Rank3, Action](
        state_action=self.q_function.core_net.encode(zipped_frames.state_action),
        is_resetting=zipped_frames.is_resetting,
        reward=zipped_frames.reward,
    )

    return self.run_q_function(frames, initial_state, policy_samples)

  @jax.profiler.annotate_function
  def step_nash_policy(
      self,
      zipped_frames: nash_data.ZippedFrames,  # [B, 2, T]
      initial_state: RecurrentState,
      policy_samples: list[Action],  # frame_skip x [S, T, B, 2]
      values: jax.Array,  # [T, B, 2]
      q_action_init_state: RecurrentState,  # [T, B, 2, H]
      nash_solution: nash.NashVariables,  # [T, B]
      train: bool = True,
  ):
    frames = Frames[nash_data.Rank3, Action](
        state_action=self.nash_policy.network.encode(zipped_frames.state_action),
        is_resetting=zipped_frames.is_resetting,
        reward=zipped_frames.reward,
    )

    fn = self.train_nash_policy if train else self.run_nash_policy
    return fn(
        frames, initial_state, policy_samples,
        values, q_action_init_state, nash_solution,
    )

  def step(
      self,
      batch: nash_data.TwoPlayerBatch[Rank2],
      initial_states: RecurrentState,
      train: bool = True,
  ) -> tuple[dict, RecurrentState]:
    # TODO: take into account delay

    zipped_frames = nash_data.batch_to_frames(batch)
    final_states = initial_states  # GC initial states as they are replaced
    metrics = {}

    (
      metrics[SAMPLE_POLICY],
      final_states[SAMPLE_POLICY],
      policy_samples,
    ) = self.step_sample_policy(
        zipped_frames, initial_states[SAMPLE_POLICY])

    (
      metrics[Q_FUNCTION],
      final_states[Q_FUNCTION],
      values,
      q_action_init_state,
      q_values,
    ) = self.step_q_function(
        zipped_frames, initial_states[Q_FUNCTION], policy_samples)

    (
      nash_variables,
      metrics[NASH],
    ) = self.compute_nash(q_values)

    (
      metrics[NASH_POLICY],
      final_states[NASH_POLICY],
    ) = self.step_nash_policy(
        zipped_frames, initial_states[NASH_POLICY], policy_samples,
        values, q_action_init_state, nash_variables, train=train)

    return metrics, final_states
