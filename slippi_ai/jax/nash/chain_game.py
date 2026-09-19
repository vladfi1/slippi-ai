"""The game over action chains shared by the nash-policy and nash-RL learners.

With frame delay D and frame skip FS, the nash at each index is over chains
of Ds + 1 = D / FS + 1 actions per player (docs/plans/nash_policy_delay.md).
ChainGame holds the chunk layout (nash_utils.ChunkLayout) and the sampling
configuration, and provides the pieces both learners compose: sampling chains
from the sample policy, scoring every pair of sampled chains with the
q-function, building the nash policy's regression target from the nash
solution, the cross-entropy of the nash policy's chains to it, and the
nash-policy-versus-nash diagnostics. Every frames argument is time-major and
already delayed (T + 1 = U + Ds + 1 states); the learners convert and encode
before calling in.

Shape glossary:
  B   batch size; two-player frames have batch shape [B, 2]
  T   steps in a chunk after the delayed slice (unroll length + Ds)
  Ds  skip-delay (delay // frame_skip)
  T'  T - Ds valid indices of the chain game, the unroll length; every loss
      is computed over these same indices (see nash_utils.chain_context)
  S   sampled chains per player (+ 1 with include_action_taken_in_samples)
  K   chains the nash policy trains on (S, or subsample)
  N   epistemic indices (num_index_samples)
  H   action_net hidden size
Chains (nash_utils.Chain) are (Ds + 1) x frame_skip nested lists of
controllers with leaves [S, T', B, 2]; time-major ("tm") arrays put T first,
batch-major ("bm") ones put B first.
"""

import logging
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx

from slippi_ai import utils
from slippi_ai.types import Frames, Action, SkipAction
from slippi_ai.jax.policies import Policy, RecurrentState
from slippi_ai.jax import jax_utils
from slippi_ai.jax.jax_utils import PS
from slippi_ai.nash import data as nash_data
from slippi_ai.jax.nash import (
    nash,
    q_function as q_lib,
    utils as nash_utils,
)

_SAMPLE_AXIS = 0

Loss = jax.Array
Metrics = dict
Values = jax.Array
QValues = jax.Array


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
  # How much better the nash does than the sample policy (index-averaged).
  nash_advantage: jax.Array  # [T', B, 2]
  metrics: dict  # [T', B, ...]


def p1_averaged_qs(two_player_qs: jax.Array) -> jax.Array:
  """Get Q-values from just player 1's perspective, assuming zero-sum."""
  # two_player_qs is [..., 2]
  return jnp.vecdot(
      two_player_qs, jnp.array([1, -1], dtype=two_player_qs.dtype),
      axis=-1) / 2


class ChainGame(tp.Generic[Action]):
  """The chain game at every valid index of a chunk; see the module doc.

  Not an nnx.Module: the networks are passed to each method so that the
  nnx transforms inside see them as arguments rather than closures.
  """

  def __init__(
      self,
      layout: nash_utils.ChunkLayout,
      *,
      num_samples: int,
      include_action_taken_in_samples: bool = True,
      # Only train the nash policy on the highest probability subsample.
      subsample: tp.Optional[int] = None,
      sample_batch_size: int = 0,  # 0 means full batch size, i.e. vmap
  ):
    if not include_action_taken_in_samples and num_samples < 2:
      raise ValueError(
          'num_samples must be at least 2 if not including action taken in samples')

    if sample_batch_size > 0:
      ns = num_samples
      if include_action_taken_in_samples:
        ns += 1
      if ns % sample_batch_size != 0:
        logging.warning(
            f'sample_batch_size {sample_batch_size} does not divide num_samples {ns}')

    self.layout = layout
    self.num_samples = num_samples
    self.include_action_taken_in_samples = include_action_taken_in_samples
    self.subsample = subsample
    self.sample_batch_size = sample_batch_size

  def chains(
      self,
      policy_samples: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      frames: Frames[nash_data.Rank3, Action],
  ) -> tuple[nash_utils.Chain[Action], int]:  # leaves [S (+ 1), T', B, 2]
    """Appends the chain actually taken to the sampled ones, if configured.

    Returns the chains and their number. Must be applied identically wherever
    the sample axis is interpreted (q_function and nash_policy unrolls).
    """
    num_samples = self.num_samples
    if self.include_action_taken_in_samples:
      taken = self.layout.taken_chain(frames)
      policy_samples = utils.map_nt(
          lambda samples, action_taken: jnp.concatenate(
              [samples, jnp.expand_dims(action_taken, axis=_SAMPLE_AXIS)],
              axis=_SAMPLE_AXIS),
          policy_samples, taken)
      num_samples += 1
    return policy_samples, num_samples

  def unroll_sample_policy(
      self,
      sample_policy: Policy[Action],
      rngs: nnx.Rngs,
      frames: Frames[nash_data.Rank3, Action],  # [T + 1, B, 2]
      initial_states: RecurrentState,  # [B, 2]
  ) -> tuple[Loss, Metrics, RecurrentState, nash_utils.Chain[Action]]:
    """Unrolls the sample policy and samples S chains per player and index.

    Returns the (game-index) imitation loss and metrics, the carried state,
    and the sampled chains with leaves [S, T', B, 2].
    """
    layout = self.layout
    sample_policy_outputs = sample_policy.scan_with_outputs(frames, initial_states)
    context = layout.context(
        sample_policy_outputs.outputs, sample_policy_outputs.hidden_states,
        frames)

    # Because the action space is too large, we compute a finite subsample
    # using the sample_policy. With delay, each sample is a chain of actions.

    @nnx.vmap(in_axes=(None, 0), out_axes=_SAMPLE_AXIS)
    def sample(sample_policy: Policy[Action], rngs: nnx.Rngs):
      return nash_utils.sample_chain(sample_policy, rngs, context)

    policy_samples = sample(sample_policy, rngs.fork(split=self.num_samples))

    imitation_loss = layout.game_slice(sample_policy_outputs.imitation_loss)
    bm_loss = jnp.mean(imitation_loss, axis=[0, 2])
    bm_metrics = utils.map_single_structure(
      lambda x: jnp.mean(x, axis=0),
      layout.game_slice(sample_policy_outputs.metrics))

    return (
        bm_loss,
        bm_metrics,
        layout.carried_state(sample_policy_outputs.hidden_states, frames),
        policy_samples,
    )

  def unroll_q_function(
      self,
      q_function: q_lib.QFunction[Action],
      rngs: nnx.Rngs,
      frames: Frames[nash_data.Rank3, Action],  # [T + 1, B, 2]
      initial_states: RecurrentState,  # [B, 2]
      policy_samples: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      *,
      num_index_samples: int,
      discount: float,
      lambda_: float | jax.Array = 1.0,
  ) -> tuple[Loss, QFunctionOutputs]:
    """Unrolls the q_function and scores every pair of sampled chains.

    Returns the q_function's own loss over the game indices (which the RL
    learner trains on) and the QFunctionOutputs.
    """
    layout = self.layout
    batch_size = self.sample_batch_size

    core = q_function.scan_core(
        frames, initial_states, rngs=rngs, num_index_samples=num_index_samples)
    zs = core.zs

    # The loss and the game only cover the valid indices.
    values = layout.game_slice(core.values, axis=1)  # [N, T', B, 2]
    q_outputs = q_function.ensemble_outputs(
        layout.game_slice(frames), values,
        layout.game_slice(core.q_values, axis=1),
        core.last_value, discount, lambda_=lambda_)

    chains, num_samples = self.chains(policy_samples, frames)
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
    q_values = q_function.multi_index_q_values_from_action_state(
        values=values,
        action_init_state=sample_action_init,
        actions=chains[-1],
        zs=zs,
        batch_size=batch_size,
        per_sample_init=True,
    )  # [N, S, S, T', B, 2]

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
    return bm_loss, outputs

  def targets(
      self,
      chains: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      num_samples: int,  # S
      q_values: jax.Array,  # [N, S, S, T', B, 2]
      nash_solution: nash.NashVariables,  # [N, T', B]
  ) -> NashTargets[Action]:
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
    if self.include_action_taken_in_samples:
      metrics['action_taken_nash_prob'] = jax.lax.index_in_dim(
          mixture_probs, index=-1, axis=-1, keepdims=False)  # [T', B, 2]

    microbatch_size = self.sample_batch_size

    # Save on computation by only training on the highest probability subsample.
    if self.subsample:
      if self.subsample > num_samples:
        raise ValueError(f'subsample {self.subsample} is greater than num_samples {num_samples}')

      indices = jnp.argsort(
          mixture_probs, axis=-1, descending=True)[..., :self.subsample]
      mixture_probs = jnp.take_along_axis(mixture_probs, indices, axis=-1)
      mixture_probs = mixture_probs / jnp.sum(mixture_probs, axis=-1, keepdims=True)  # re-normalize

      indices = jnp.moveaxis(indices, -1, _SAMPLE_AXIS)  # [K, T', B, 2]
      chains = utils.map_nt(
          lambda x: jnp.take_along_axis(x, indices, axis=_SAMPLE_AXIS), chains)
      num_samples = self.subsample

      if self.subsample < microbatch_size:
        microbatch_size = 0
      elif microbatch_size > 0 and self.subsample % microbatch_size != 0:
        raise ValueError(f'subsample {self.subsample} is not divisible by sample_batch_size {microbatch_size}')

    return NashTargets(
        mixture_probs=mixture_probs,
        chains=chains,
        num_samples=num_samples,
        microbatch_size=microbatch_size,
        nash_advantage=diagnostics.nash_advantage,
        metrics=metrics,
    )

  def cross_entropy(
      self,
      nash_policy: Policy[Action],
      context: nash_utils.ChainContext,  # from the nash_policy's scan
      targets: NashTargets[Action],
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

  def nash_policy_qs(
      self,
      q_function: q_lib.QFunction[Action],
      frames: Frames[nash_data.Rank3, Action],  # [T + 1, B, 2], encoded for the q_function
      policy_samples: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      q_outputs: QFunctionOutputs,
      nash_policy_chain: nash_utils.Chain[Action],  # leaves [T', B, 2]
      nash_solution: nash.NashVariables,  # [N, T', B]
  ) -> Metrics:  # [T', B, 2]
    """Diagnostics: how the nash_policy's sampled chain fares against the
    (per-index) nash distributions over the sampled chains, under the
    q_function. Uses all S samples, regardless of subsampling in training.
    """
    values = q_outputs.values
    sample_action_init = q_outputs.sample_action_init
    q_values = q_outputs.q_values
    zs = q_outputs.zs
    index_mean = lambda x: jnp.mean(x, axis=0)

    chains, _ = self.chains(policy_samples, frames)
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
        batch_size=self.sample_batch_size,
    )
    nash_policy_qs = jnp.moveaxis(nash_policy_qs, 0, -1)  # [N, T', B, 2, S]
    # Evaluated against the opponent's nash distribution.
    nash_policy_vs_nash = jnp.vecdot(
        nash_policy_qs, jnp.flip(nash_probs, axis=-2))  # [N, T', B, 2]
    optimality_gap = nash_values - nash_policy_vs_nash

    mean_vs_nash = -jnp.flip(diagnostics.nash_vs_mean, axis=-1)  # [N, T', B, 2]
    nash_policy_advantage = nash_policy_vs_nash - mean_vs_nash

    return dict(
        nash_policy_vs_mean=index_mean(nash_policy_qs.mean(axis=-1)),
        nash_policy_vs_nash=index_mean(nash_policy_vs_nash),
        optimality_gap=index_mean(optimality_gap),  # nash-vs-nash - nash_policy-vs-nash
        nash_policy_advantage=index_mean(nash_policy_advantage),  # nash_policy-vs-nash - mean-vs-nash
    )
