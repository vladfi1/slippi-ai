import contextlib
import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from slippi_ai import utils
from slippi_ai.types import Frames, Action
from slippi_ai.data import Rank2
from slippi_ai.jax.policies import Policy, RecurrentState
from slippi_ai.jax import rl_lib, jax_utils, networks
from slippi_ai.jax.jax_utils import PS, DATA_AXIS
from slippi_ai.nash import data as nash_data
from slippi_ai.jax.nash import (
    chain_game,
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

# The chain game (sampling, scoring, targets) is shared with the nash RL
# learner; see chain_game.py for the shape glossary.
QFunctionOutputs = chain_game.QFunctionOutputs
QFunctionOutputSpecs = chain_game.QFunctionOutputSpecs


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
    # actions; see docs/plans/nash_policy_delay.md and chain_game.
    self.layout = nash_utils.ChunkLayout(self.delay, nash_policy.frame_skip)
    self.skip_delay = self.layout.skip_delay
    self.game = chain_game.ChainGame[Action](
        self.layout,
        num_samples=config.num_samples,
        include_action_taken_in_samples=config.include_action_taken_in_samples,
        subsample=config.subsample,
        sample_batch_size=config.sample_batch_size,
    )

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

  def _delayed_frames(
      self,
      bm_frames: Frames[nash_data.Rank3, Action],  # [B, 2, T]
  ) -> Frames[nash_data.Rank3, Action]:  # [T + 1, B, 2], T + 1 = U + Ds + 1 states
    return self.layout.delayed_frames(nash_utils.bm_to_tm(bm_frames))

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
    return self.game.unroll_sample_policy(
        sample_policy, rngs, self._delayed_frames(bm_frames), initial_states)

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
    bm_loss, outputs = self.game.unroll_q_function(
        q_function, rngs, self._delayed_frames(bm_frames), initial_states,
        policy_samples, num_index_samples=num_index_samples,
        discount=self.discount, lambda_=1.0)
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
    frames = self._delayed_frames(bm_frames)

    chains, num_samples = self.game.chains(policy_samples, frames)
    del policy_samples
    targets = self.game.targets(chains, num_samples, q_values, nash_solution)

    nash_policy_outputs = nash_policy.scan_with_outputs(
        frames, initial_states)
    nash_policy_imitation_loss = self.layout.game_slice(
        nash_policy_outputs.imitation_loss)  # [T', B, 2]
    context = self.layout.context(
        nash_policy_outputs.outputs, nash_policy_outputs.hidden_states, frames)

    nash_cross_entropy = self.game.cross_entropy(
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
    """Diagnostics of the nash_policy's sampled chain against the nash
    distributions (chain_game.ChainGame.nash_policy_qs), in the loss
    function format with a zero loss."""
    frames = self._delayed_frames(bm_frames)
    metrics = self.game.nash_policy_qs(
        q_function, frames, policy_samples, q_outputs, nash_policy_chain,
        nash_solution)

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
