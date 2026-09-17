import contextlib
import dataclasses
import logging
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from slippi_ai import utils
from slippi_ai.types import S, Frames, Action, SkipAction
from slippi_ai.data import Rank2
from slippi_ai.jax.policies import Policy, RecurrentState
from slippi_ai.jax import embed, rl_lib, jax_utils, networks
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
  # Only train on the highest probability subsample.
  subsample: tp.Optional[int] = None

  # Number of epistemic indices to sample. The nash is solved once per index,
  # and the nash_policy regresses to the mixture of the per-index nash
  # distributions. Needs to be at least 2 for the epistemic metrics.
  num_index_samples: int = 4
  eval_num_index_samples: tp.Optional[int] = None

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

# Shape glossary:
#   B   batch size; two-player frames have batch shape [B, 2]
#   T   steps in a chunk after the delayed slice (unroll length + Ds)
#   Ds  skip-delay (delay // frame_skip)
#   T'  T - Ds valid indices of the chain game, the unroll length; every loss
#       is computed over these same indices (see nash_utils.chain_context)
#   S   sampled chains per player (+ 1 with include_action_taken_in_samples)
#   N   epistemic indices (num_index_samples)
#   H   action_net hidden size
# Chains (nash_utils.Chain) are (Ds + 1) x frame_skip nested lists of
# controllers with leaves [S, T', B, 2]; time-major ("tm") arrays put T first,
# batch-major ("bm") ones put B first.

class QFunctionOutputs(tp.NamedTuple):
  """What the q_function unroll hands to the nash solver, the nash_policy
  unroll and the nash_policy diagnostics (besides its loss)."""
  metrics: Metrics  # [B]
  final_state: RecurrentState  # carried state [B]
  values: Values  # [N, T', B, 2] per epistemic index
  sample_action_init: RecurrentState  # per-sample action_init_state [S, T', B, 2, H]
  core_outputs: jax.Array  # core_net outputs [T, B, 2, O_core]
  core_states: RecurrentState  # core_net states after each step [T, B, 2]
  q_values: QValues  # [N, S, S, T', B, 2] per epistemic index
  zs: jax.Array  # [N, B, 1, D_Z] epistemic indices


class QFunctionOutputSpecs(tp.NamedTuple):
  """Partition specs of QFunctionOutputs."""
  metrics: PS
  final_state: PS
  values: PS
  sample_action_init: PS
  core_outputs: PS
  core_states: PS
  q_values: PS
  zs: PS

  def as_prefix(self) -> tp.Self:
    """As a pytree prefix of a QFunctionOutputs (which must have the same
    named-tuple type for jax to match them)."""
    return QFunctionOutputs(*self)  # type: ignore


class NashTargets(tp.NamedTuple, tp.Generic[Action]):
  """The nash_policy's regression target at each game index."""
  mixture_probs: jax.Array  # [T', B, 2, K] mixture over epistemic indices
  chains: nash_utils.Chain[Action]  # leaves [K, T', B, 2], the K chains
  num_samples: int  # K
  microbatch_size: int  # for mapping over the K chains
  metrics: dict  # [T', B, ...]


class ShardingKwargs(tp.TypedDict):
  mesh: jax.sharding.Mesh
  explicit_pmean: bool
  smap_optimizer: bool

class ShardingSpecs(tp.TypedDict):
  extra_in_specs: tp.Optional[jax_utils.Specs]
  extra_out_specs: tp.Optional[jax_utils.Specs]

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
    if sample_policy.delay != self.delay:
      raise ValueError(
          f'Sample policy delay {sample_policy.delay} does not match '
          f'nash policy delay {self.delay}.')
    if q_function.frame_skip != nash_policy.frame_skip:
      raise ValueError(
          f'Q-function frame skip {q_function.frame_skip} does not match '
          f'nash policy frame skip {nash_policy.frame_skip}.')
    # With delay, the game at each index is over chains of skip_delay + 1
    # actions; see docs/plans/nash_policy_delay.md and nash_utils.
    self.layout = nash_utils.ChunkLayout(self.delay, nash_policy.frame_skip)
    self.skip_delay = self.layout.skip_delay

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

    # Per-epistemic-index arrays carry a leading index axis N.
    NTM = PS(None, None, DATA_AXIS)  # [N, T, B, ...]
    NTMSS = PS(None, None, None, None, DATA_AXIS)  # [N, S, S, T, B, ...]
    NB = PS(None, DATA_AXIS)  # [N, B, ...]

    policy_samples = TMS
    q_specs = QFunctionOutputSpecs(
        metrics=BM,
        final_state=BM,
        values=NTM,
        sample_action_init=TMS,
        core_outputs=TM,
        core_states=TM,
        q_values=NTMSS,
        zs=NB,
    )
    qs = q_specs.q_values
    nash_solution = NTM  # [N, T, B]
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

    # The metrics and final state are the loss function's standard outputs.
    q_function_specs = ShardingSpecs(
        extra_in_specs=(policy_samples,),
        extra_out_specs=q_specs[2:],
    )

    # Keep q_function in fp32 so we can distinguish small differences in
    # q-values that lead to different nash solutions.
    unroll_q_function = self._unroll_q_function
    # if config.bf16:
    #   unroll_q_function = jax_utils.with_bf16_compute(unroll_q_function)

    self.run_q_function = jax_utils.shard_map_loss_fn_with_rngs(
        module=self.q_function,
        rngs=rngs,
        loss_fn=unroll_q_function,
        mesh=mesh,
        **q_function_specs,
        static_argnames=('num_index_samples',),
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

    nash_chain = TM  # leaves [T, B, 2]
    nash_policy_specs = ShardingSpecs(
        extra_in_specs=(policy_samples, qs, nash_solution),
        extra_out_specs=(nash_chain,),
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

    # Scores the nash_policy's sampled chain with the q_function. This runs
    # on the q_function (not closed over) because its networks may contain
    # nnx transforms, which fail on a closed-over module inside the
    # nash_policy's loss.
    nash_policy_qs_specs = ShardingSpecs(
        extra_in_specs=(
            policy_samples, q_specs.as_prefix(), nash_chain, nash_solution),
        extra_out_specs=None,
    )
    self.run_nash_policy_qs = jax_utils.shard_map_loss_fn(
        module=self.q_function,
        loss_fn=self._nash_policy_qs,
        mesh=mesh,
        **nash_policy_qs_specs,
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

  def _chains(
      self,
      policy_samples: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      frames: Frames[nash_data.Rank3, Action],
  ) -> tuple[nash_utils.Chain[Action], int]:  # leaves [S (+ 1), T', B, 2]
    """Appends the chain actually taken to the sampled ones, if configured.

    Returns the chains and their number. Must be applied identically wherever
    the sample axis is interpreted (q_function and nash_policy unrolls).
    """
    num_samples = self.num_samples
    if self.config.include_action_taken_in_samples:
      taken = self.layout.taken_chain(frames)
      policy_samples = utils.map_nt(
          lambda samples, action_taken: jnp.concatenate(
              [samples, jnp.expand_dims(action_taken, axis=_SAMPLE_AXIS)],
              axis=_SAMPLE_AXIS),
          policy_samples, taken)
      num_samples += 1
    return policy_samples, num_samples

  def _unroll_sample_policy(
      self,
      sample_policy: Policy[Action],
      bm_frames: Frames[nash_data.Rank3, Action],
      initial_states: RecurrentState,
      rngs: nnx.Rngs,
  ) -> tuple[Loss, Metrics, RecurrentState, nash_utils.Chain[Action]]:
    """Unrolls the sample policy and samples S chains per player and index.

    Returns the (game-index) imitation loss and metrics, the carried state,
    and the sampled chains with leaves [S, T', B, 2].
    """
    frames = nash_utils.bm_to_tm(bm_frames)
    frames = self.layout.delayed_frames(frames)  # T + 1 = U + Ds + 1 states

    sample_policy_outputs = sample_policy.scan_with_outputs(frames, initial_states)
    context = self.layout.context(
        sample_policy_outputs.outputs, sample_policy_outputs.hidden_states,
        frames)

    # Because the action space is too large, we compute a finite subsample
    # using the sample_policy. With delay, each sample is a chain of actions.

    @nnx.vmap(in_axes=(None, 0), out_axes=_SAMPLE_AXIS)
    def sample(sample_policy: Policy[Action], rngs: nnx.Rngs):
      return nash_utils.sample_chain(sample_policy, rngs, context)

    policy_samples = sample(sample_policy, rngs.fork(split=self.num_samples))

    imitation_loss = self.layout.game_slice(sample_policy_outputs.imitation_loss)
    bm_loss = jnp.mean(imitation_loss, axis=[0, 2])
    bm_metrics = utils.map_single_structure(
      lambda x: jnp.mean(x, axis=0),
      self.layout.game_slice(sample_policy_outputs.metrics))

    return (
        bm_loss,
        bm_metrics,
        self.layout.carried_state(sample_policy_outputs.hidden_states, frames),
        policy_samples,
    )

  def _unroll_q_function(
      self,
      q_function: q_lib.QFunction[Action],
      bm_frames: Frames[nash_data.Rank3, Action],
      initial_states: RecurrentState,
      rngs: nnx.Rngs,
      policy_samples: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      *,
      num_index_samples: int,
  ) -> tuple[Loss, ...]:  # (loss, *QFunctionOutputs)
    """Unrolls the q_function and scores every pair of sampled chains.

    Returns the q_function's own loss over the game indices (as a
    diagnostic; it is not trained here) followed by the QFunctionOutputs.
    """
    frames = nash_utils.bm_to_tm(bm_frames)
    frames = self.layout.delayed_frames(frames)  # T + 1 = U + Ds + 1 states
    layout = self.layout
    batch_size = self.config.sample_batch_size

    core = q_function.scan_core(
        frames, initial_states, rngs=rngs, num_index_samples=num_index_samples)
    zs = core.zs

    # The loss and the game only cover the valid indices.
    values = layout.game_slice(core.values, axis=1)  # [N, T', B, 2]
    q_outputs = q_function.ensemble_outputs(
        layout.game_slice(frames), values,
        layout.game_slice(core.q_values, axis=1),
        core.last_value, self.discount, lambda_=1.0)

    chains, num_samples = self._chains(policy_samples, frames)
    del policy_samples

    # Each chain's last action is scored by the action_net from the core
    # state after the chain's prefix: re-run the core_net over the real
    # inputs at indices [t - Ds + 1, t] with the chain's actions, per sample.
    context = layout.context(core.core_outputs, core.core_states, frames)

    def init_for_chain(
        q_function: q_lib.QFunction[Action], /,
        chain: nash_utils.Chain[Action],  # (Ds + 1) x frame_skip x [T', B, 2]
    ) -> RecurrentState:  # [T', B, 2, H]
      return q_function.chain_action_init_state(context, chain)

    # Maps over the sample axis of the chains. This must be an nnx-aware map
    # with q_function as an argument (not a closure): the core_net may
    # contain nnx transforms (e.g. the ControllerRNN embedding's scan), which
    # fail under a raw jax.lax.map.
    init_mbs = batch_size
    if init_mbs > 0 and num_samples % init_mbs != 0:
      init_mbs = 0  # fall back to a full vmap
    batch_init_for_chain = jax_utils.lax_map_fn(
        init_for_chain,
        microbatch_size=init_mbs,
        input_batch_dims=(None, 0),
        output_batch_dims=0,
    )
    sample_action_init = batch_init_for_chain(  # [S, T', B, 2, H]
        q_function, chains)

    assert _SAMPLE_AXIS == 0
    sample_q_values = q_function.multi_index_q_values_from_action_state(
        values=values,
        action_init_state=sample_action_init,
        actions=chains[-1],
        zs=zs,
        batch_size=batch_size,
        per_sample_init=True,
    )

    q_values = sample_q_values  # [N, S, S, T', B, 2]

    bm_loss = jnp.mean(q_outputs.loss, axis=[0, 2])

    payoff_matrices = nash_utils.mixed_payoff_matrices(q_values)  # [N, T', B, S, S]

    metrics = dict(
        q_outputs.metrics,
        information_fraction=jnp.mean(
            nash_utils.information_fraction(payoff_matrices), axis=0),  # [T', B]
    )

    bm_metrics = utils.map_single_structure(
      lambda x: jnp.mean(x, axis=0), metrics)

    outputs = QFunctionOutputs(
        metrics=bm_metrics,
        final_state=layout.carried_state(core.core_states, frames),
        values=values,
        sample_action_init=sample_action_init,
        core_outputs=core.core_outputs,
        core_states=core.core_states,
        q_values=q_values,
        zs=zs,
    )
    return (bm_loss, *outputs)

  def _compute_nash(
      self,
      q_values: jax.Array,  # [N, S, S, T, B, 2]
  ) -> tuple[nash.NashVariables, Metrics]:
    num_indices, s1, s2, t, b, n = q_values.shape
    assert s1 == s2
    assert n == 2

    payoff_matrices = nash_utils.mixed_payoff_matrices(q_values)  # [N, T, B, S, S]

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

      solve_vmap = jax_utils.multi_vmap(solve_one, axes=[0, 1, 2])

      # One nash solution per epistemic index; leaves are [N, T, B, ...].
      nash_variables, nm_metrics = solve_vmap(payoff_matrices)
      assert isinstance(nash_variables, nash.NashVariables)

    nash_variables = utils.map_single_structure(
        lambda x: x.astype(jnp.float32), nash_variables)

    ps = jnp.stack([nash_variables.p1, nash_variables.p2], axis=-2)  # [N, T, B, 2, S]
    assert ps.shape == (num_indices, t, b, 2, s1)
    ps = ps / jnp.sum(ps, axis=-1, keepdims=True)  # re-normalize for numerical stability

    # The mass-coverage and entropy statistics are computed on the mixture over
    # epistemic indices, which is the distribution the nash_policy regresses
    # to. The per-index disagreement metrics are computed in the nash_policy
    # unroll.
    mixture_ps = jnp.mean(ps, axis=0)  # [T, B, 2, S]

    ps_stats = {}
    sorted_ps = jnp.sort(mixture_ps, descending=True, axis=-1)
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

    mixture_metrics = {'ps': ps_stats}

    nash_entropy = jax_utils.entropy(mixture_ps, axis=-1)  # [T, B, 2]
    mixture_metrics['entropy'] = nash_entropy
    entropy_stats = {}

    for cutoff in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.4, 2]:
      # Mean will be taken later
      entropy_stats[cutoff] = nash_entropy > cutoff

    mixture_metrics['entropy_above'] = entropy_stats

    # Batch-major metrics; keep index and time dims so we can take max over
    # num_steps.
    bm_metrics = utils.map_single_structure(
        lambda x: jnp.moveaxis(x, 2, 0), nm_metrics)  # [B, N, T, ...]

    # The mixture metrics have no index axis.
    bm_metrics.update(utils.map_single_structure(
        lambda x: jnp.moveaxis(x, 1, 0), mixture_metrics))  # [B, T, ...]

    # First epistemic index, first timestep.
    bm_metrics['sample_payoff_matrix'] = payoff_matrices[0, 0].astype(jnp.float32)  # [B, S1, S2]

    return nash_variables, bm_metrics


  def _unroll_nash_policy(
      self,
      nash_policy: Policy[Action],
      bm_frames: Frames[nash_data.Rank3, Action],
      initial_states: RecurrentState,
      rngs: nnx.Rngs,
      policy_samples: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      q_values: jax.Array,  # [N, S, S, T', B, 2]
      nash_solution: nash.NashVariables,  # [N, T', B]
  ) -> tuple[Loss, dict, RecurrentState, nash_utils.Chain[Action]]:
    """Trains the nash_policy towards the nash distributions over chains.

    The loss at each game index is the cross-entropy from the mixture (over
    epistemic indices) of the nash distributions over the S chains to the
    nash_policy's chain log-probs, plus an optional imitation term. Also
    returns a chain sampled from the nash_policy (leaves [T', B, 2]), which
    _nash_policy_qs scores against the nash distribution.
    """
    frames = nash_utils.bm_to_tm(bm_frames)
    frames = self.layout.delayed_frames(frames)  # T + 1 = U + Ds + 1 states

    chains, num_samples = self._chains(policy_samples, frames)
    del policy_samples
    targets = self._nash_targets(chains, num_samples, q_values, nash_solution)

    nash_policy_outputs = nash_policy.scan_with_outputs(
        frames, initial_states)
    nash_policy_imitation_loss = self.layout.game_slice(
        nash_policy_outputs.imitation_loss)  # [T', B, 2]
    context = self.layout.context(
        nash_policy_outputs.outputs, nash_policy_outputs.hidden_states, frames)

    nash_cross_entropy = self._chain_cross_entropy(
        nash_policy, context, targets)  # [T', B, 2]

    losses = [
        self.config.nash_weight * nash_cross_entropy,
        self.config.imitation_weight * nash_policy_imitation_loss,
    ]
    nash_policy_total_loss = jax_utils.add_n(losses)

    metrics = dict(
        targets.metrics,
        nash_cross_entropy=nash_cross_entropy,
        imitation_loss=nash_policy_imitation_loss,
        total_loss=nash_policy_total_loss,
    )

    # Sampled as the nash_policy would act online; scored in _nash_policy_qs.
    nash_policy_chain = nash_utils.sample_chain(nash_policy, rngs, context)

    bm_loss = jnp.mean(nash_policy_total_loss, axis=[0, 2])
    bm_metrics = utils.map_single_structure(
      lambda x: jnp.mean(x, axis=0), metrics)

    return (
        bm_loss, bm_metrics,
        self.layout.carried_state(nash_policy_outputs.hidden_states, frames),
        nash_policy_chain)

  def _nash_targets(
      self,
      chains: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      num_samples: int,  # S
      q_values: jax.Array,  # [N, S, S, T', B, 2]
      nash_solution: nash.NashVariables,  # [N, T', B]
  ) -> NashTargets:
    """Builds the regression target from the nash solution.

    The target is the mixture over epistemic indices of the per-index nash
    distributions over the S chains, optionally restricted to the K =
    subsample most likely chains (re-normalized). Also computes the
    diagnostics of the nash solution itself (its metrics are per epistemic
    index and averaged).
    """
    # TODO: this could belong in the sample policy unroll
    unique_fraction = nash_utils.compute_unique_fraction(
        nash_utils.flatten_chain(chains))

    nash_probs, nash_values = nash_utils.nash_solution_probs(nash_solution)
    mixture_probs, metrics = nash_utils.indexed_nash_metrics(nash_probs)
    metrics['unique_fraction'] = unique_fraction

    payoff_matrices = nash_utils.mixed_payoff_matrices(q_values)  # [N, T', B, S, S]
    diagnostics = nash_utils.nash_payoff_diagnostics(
        payoff_matrices, nash_probs, nash_values)
    metrics.update(diagnostics.metrics)

    # Computed before any subsampling reorders the sample axis: the action
    # taken is the last sample.
    if self.config.include_action_taken_in_samples:
      metrics['action_taken_nash_prob'] = jax.lax.index_in_dim(
          mixture_probs, index=-1, axis=-1, keepdims=False)  # [T', B, 2]

    microbatch_size = self.config.sample_batch_size

    # Save on computation by only training on the highest probability subsample.
    if self.config.subsample:
      if self.config.subsample > num_samples:
        raise ValueError(f'subsample {self.config.subsample} is greater than num_samples {num_samples}')

      indices = jnp.argsort(
          mixture_probs, axis=-1, descending=True)[..., :self.config.subsample]
      mixture_probs = jnp.take_along_axis(mixture_probs, indices, axis=-1)
      mixture_probs = mixture_probs / jnp.sum(mixture_probs, axis=-1, keepdims=True)  # re-normalize

      indices = jnp.moveaxis(indices, -1, _SAMPLE_AXIS)  # [K, T', B, 2]
      chains = utils.map_nt(
          lambda x: jnp.take_along_axis(x, indices, axis=_SAMPLE_AXIS), chains)
      num_samples = self.config.subsample

      if self.config.subsample < microbatch_size:
        microbatch_size = 0
      elif microbatch_size > 0 and self.config.subsample % microbatch_size != 0:
        raise ValueError(f'subsample {self.config.subsample} is not divisible by sample_batch_size {microbatch_size}')

    return NashTargets(
        mixture_probs=mixture_probs,
        chains=chains,
        num_samples=num_samples,
        microbatch_size=microbatch_size,
        metrics=metrics,
    )

  def _chain_cross_entropy(
      self,
      nash_policy: Policy[Action],
      context: nash_utils.ChainContext,  # from the nash_policy's scan
      targets: NashTargets,
  ) -> jax.Array:  # [T', B, 2]
    """Cross-entropy from the target mixture over chains to the
    nash_policy's chain log-probs."""

    # Note that this inefficiently recomputes the controller head encoder
    # outputs for each sample.
    def nash_policy_distance_fn(
        nash_policy: Policy[Action], /,
        chain: nash_utils.Chain[Action],  # leaves [T', B, 2]
    ) -> jax.Array:  # [T', B, 2]
      return -nash_utils.chain_log_prob(nash_policy, context, chain)

    if targets.microbatch_size > 0:
      batch_distance_fn = jax_utils.lax_map_fn(
          nnx.remat(nash_policy_distance_fn),
          microbatch_size=targets.microbatch_size,
          input_batch_dims=(None, 0),
          output_batch_dims=0,
      )
    else:
      batch_distance_fn = nnx.vmap(
          nash_policy_distance_fn,
          in_axes=(None, 0), out_axes=0,
      )

    log_probs = -batch_distance_fn(nash_policy, targets.chains)
    log_probs = jnp.moveaxis(log_probs, _SAMPLE_AXIS, -1)  # [T', B, 2, K]
    return -jnp.vecdot(targets.mixture_probs, log_probs, axis=-1)

  def _nash_policy_qs(
      self,
      q_function: q_lib.QFunction[Action],
      bm_frames: Frames[nash_data.Rank3, Action],  # encoded for the q_function
      initial_states: RecurrentState,  # unused, passed through
      policy_samples: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      q_outputs: QFunctionOutputs,
      nash_policy_chain: nash_utils.Chain[Action],  # leaves [T', B, 2]
      nash_solution: nash.NashVariables,  # [N, T', B]
  ) -> tuple[Loss, dict, RecurrentState]:
    """Diagnostics: how the nash_policy's sampled chain fares against the
    (per-index) nash distributions over the sampled chains, under the
    q_function. Uses all S samples, regardless of subsampling in training.
    """
    values = q_outputs.values
    sample_action_init = q_outputs.sample_action_init
    q_values = q_outputs.q_values
    zs = q_outputs.zs
    frames = nash_utils.bm_to_tm(bm_frames)
    frames = self.layout.delayed_frames(frames)  # T + 1 = U + Ds + 1 states
    index_mean = lambda x: jnp.mean(x, axis=0)

    chains, _ = self._chains(policy_samples, frames)
    del policy_samples

    nash_probs, nash_values = nash_utils.nash_solution_probs(nash_solution)
    payoff_matrices = nash_utils.mixed_payoff_matrices(q_values)  # [N, T', B, S, S]
    diagnostics = nash_utils.nash_payoff_diagnostics(
        payoff_matrices, nash_probs, nash_values)

    # The q_function's core state after the nash_policy chain's prefix.
    context = self.layout.context(
        q_outputs.core_outputs, q_outputs.core_states, frames)
    nash_policy_action_init = q_function.chain_action_init_state(
        context, nash_policy_chain)

    nash_policy_last_action = nash_policy_chain[-1]  # frame_skip x [T', B, 2]

    # TODO: this is fairly inefficient -- we should instead pre-compute the
    # q-function's "outputs" on both the nash policy and the sampled actions,
    # the latter which we already have from the q-function unroll, and then use
    # QFunction._q_values_from_outputs.
    def compute_nash_policy_q_vs(
        opponent: tuple[RecurrentState, SkipAction[Action]],  # one sample: [T', B, 2, H], frame_skip x [T', B, 2]
    ) -> jax.Array:  # [N, T', B, 2]
      """Q-values of the nash_policy's chain against one opponent sample,
      from each player's perspective (nash_policy as p1 vs sample as p2,
      and sample as p1 vs nash_policy as p2)."""
      opponent_action_init, opponent_last_action = opponent

      # Line up nash policy vs the other policy samples.
      merged_action_init = nash_utils.merge_players(
          nash_policy_action_init, opponent_action_init)  # [2, T', B, 2, H]
      merged_actions = nash_utils.merge_players(
          nash_policy_last_action, opponent_last_action)  # [2, T', B, 2]

      def q_fn(
          action_init_state: RecurrentState,  # [T', B, 2, H]
          actions: SkipAction[Action],  # frame_skip x [T', B, 2]
      ) -> jax.Array:  # [N, T', B]
        two_player_qs = q_function.indexed_q_values_from_action_state(
          values=values,
          action_init_state=action_init_state,
          actions=actions,
          zs=zs,
        )  # [N, T', B, 2]
        return p1_averaged_qs(two_player_qs)  # [N, T', B]

      merged_qs = jax.vmap(q_fn, in_axes=0, out_axes=0)(
          merged_action_init, merged_actions)  # [2, N, T', B]

      np1_vs_p2_qs, p1_vs_np2_qs = jnp.unstack(merged_qs, axis=0)  # [N, T', B]
      return jnp.stack([np1_vs_p2_qs, -p1_vs_np2_qs], axis=-1)  # [N, T', B, 2]

    # nash_policy vs the per-index nash distributions over sampled actions,
    # each evaluated under its own index's q-function. A raw jax map is fine
    # here as indexed_q_values_from_action_state has no nnx transforms inside.
    nash_policy_qs = jax_utils.lax_map(  # [S, N, T', B, 2]
        compute_nash_policy_q_vs, (sample_action_init, chains[-1]),
        batch_size=self.config.sample_batch_size,
    )
    nash_policy_qs = jnp.moveaxis(nash_policy_qs, 0, -1)  # [N, T', B, 2, S]
    # Evaluated against the opponent's nash distribution.
    nash_policy_vs_nash = jnp.vecdot(
        nash_policy_qs, jnp.flip(nash_probs, axis=-2))  # [N, T', B, 2]
    optimality_gap = nash_values - nash_policy_vs_nash

    mean_vs_nash = -jnp.flip(diagnostics.nash_vs_mean, axis=-1)  # [N, T', B, 2]
    nash_policy_advantage = nash_policy_vs_nash - mean_vs_nash

    metrics = dict(
        nash_policy_vs_mean=index_mean(nash_policy_qs.mean(axis=-1)),
        nash_policy_vs_nash=index_mean(nash_policy_vs_nash),
        optimality_gap=index_mean(optimality_gap),  # nash-vs-nash - nash_policy-vs-nash
        nash_policy_advantage=index_mean(nash_policy_advantage),  # nash_policy-vs-nash - mean-vs-nash
    )

    bm_loss = jnp.zeros(frames.reward.shape[1], dtype=jnp.float32)  # [B]
    bm_metrics = utils.map_single_structure(
      lambda x: jnp.mean(x, axis=0), metrics)
    return bm_loss, bm_metrics, initial_states

  def _encode(
      self,
      network: networks.StateActionNetwork[Action],
      zipped_frames: nash_data.ZippedFrames,  # [B, 2, T]
  ) -> Frames[nash_data.Rank3, Action]:
    return Frames[nash_data.Rank3, Action](
        state_action=network.encode(zipped_frames.state_action),
        is_resetting=zipped_frames.is_resetting,
        reward=zipped_frames.reward,
    )

  @jax.profiler.annotate_function
  def step_sample_policy(
      self,
      zipped_frames: nash_data.ZippedFrames,  # [B, 2, T]
      initial_state: RecurrentState,
  ):
    frames = self._encode(self.sample_policy.network, zipped_frames)
    return self.run_sample_policy(frames, initial_state)

  @jax.profiler.annotate_function
  def step_q_function(
      self,
      zipped_frames: nash_data.ZippedFrames,  # [B, 2, T]
      initial_state: RecurrentState,
      policy_samples: nash_utils.Chain,
      train: bool = True,
  ) -> QFunctionOutputs:
    frames = self._encode(self.q_function.core_net, zipped_frames)

    num_index_samples = self.config.num_index_samples
    if not train:
      num_index_samples = self.config.eval_num_index_samples or num_index_samples

    outputs = self.run_q_function(
        frames, initial_state, policy_samples,
        num_index_samples=num_index_samples)
    return QFunctionOutputs(*outputs)

  @jax.profiler.annotate_function
  def step_nash_policy(
      self,
      zipped_frames: nash_data.ZippedFrames,  # [B, 2, T]
      initial_state: RecurrentState,
      policy_samples: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      q_values: jax.Array,  # [N, S, S, T', B, 2]
      nash_solution: nash.NashVariables,  # [N, T', B]
      train: bool = True,
  ):
    frames = self._encode(self.nash_policy.network, zipped_frames)
    fn = self.train_nash_policy if train else self.run_nash_policy
    return fn(frames, initial_state, policy_samples, q_values, nash_solution)

  @jax.profiler.annotate_function
  def step_nash_policy_qs(
      self,
      zipped_frames: nash_data.ZippedFrames,  # [B, 2, T]
      initial_state: RecurrentState,  # q_function state, unused
      policy_samples: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      q_outputs: QFunctionOutputs,
      nash_policy_chain: nash_utils.Chain[Action],  # leaves [T', B, 2]
      nash_solution: nash.NashVariables,  # [N, T', B]
  ) -> tuple[dict, RecurrentState]:
    """Returns the diagnostics and the (passed-through) q_function state; the
    input state is donated, so the caller must keep the returned one."""
    frames = self._encode(self.q_function.core_net, zipped_frames)
    return self.run_nash_policy_qs(
        frames, initial_state, policy_samples, q_outputs, nash_policy_chain,
        nash_solution)

  def step(
      self,
      batch: nash_data.TwoPlayerBatch[Rank2],
      initial_states: RecurrentState,
      train: bool = True,
  ) -> tuple[dict, RecurrentState]:
    zipped_frames = nash_data.batch_to_frames(batch)
    final_states = initial_states  # GC initial states as they are replaced
    metrics = {}

    (
      metrics[SAMPLE_POLICY],
      final_states[SAMPLE_POLICY],
      policy_samples,
    ) = self.step_sample_policy(
        zipped_frames, initial_states[SAMPLE_POLICY])

    q_outputs = self.step_q_function(
        zipped_frames, initial_states[Q_FUNCTION], policy_samples)
    metrics[Q_FUNCTION] = q_outputs.metrics
    final_states[Q_FUNCTION] = q_outputs.final_state

    (
      nash_variables,
      metrics[NASH],
    ) = self.compute_nash(q_outputs.q_values)

    (
      metrics[NASH_POLICY],
      final_states[NASH_POLICY],
      nash_policy_chain,
    ) = self.step_nash_policy(
        zipped_frames, initial_states[NASH_POLICY], policy_samples,
        q_outputs.q_values, nash_variables, train=train)

    if self.config.compute_nash_policy_qs:
      # The q_function's state is donated and passed through.
      nash_policy_qs_metrics, final_states[Q_FUNCTION] = self.step_nash_policy_qs(
          zipped_frames, final_states[Q_FUNCTION], policy_samples,
          q_outputs, nash_policy_chain, nash_variables)
      metrics[NASH_POLICY].update(nash_policy_qs_metrics)

    return metrics, final_states
