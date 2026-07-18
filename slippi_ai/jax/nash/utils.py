import typing as tp

import jax
import jax.numpy as jnp

from slippi_ai import utils
from slippi_ai.jax import jax_utils
from slippi_ai.types import Action

T = tp.TypeVar('T')

def bm_to_tm(nest: T) -> T:
  """Converts [B, 2, T] to [T, B, 2]."""
  return utils.map_single_structure(
      lambda x: jnp.moveaxis(x, 2, 0), nest)

def tm_to_bm(nest: T) -> T:
  """Converts [T, B, 2] to [B, 2, T]."""
  return utils.map_single_structure(
      lambda x: jnp.moveaxis(x, 0, 2), nest)


def compute_unique_fraction(actions: list[Action]) -> jax.Array:
  # Compute fraction of actions that are unique

  # We assume that the action components are scalars
  stacked_actions = utils.map_nt(  # [S, T, FS, B, 2]
      lambda *xs: jnp.stack(xs, axis=2), *actions)
  combined_actions = jnp.stack(
      jax.tree.leaves(stacked_actions), axis=-1)  # [S, T, FS, B, 2, C]
  num_samples = combined_actions.shape[0]

  actions_eq = combined_actions == jnp.expand_dims(combined_actions, axis=1)  # [S, S, T, FS, B, 2, C]
  actions_eq = jnp.all(actions_eq, axis=[2, -1])  # [S, S, T, B, 2]

  ns = jnp.arange(num_samples)
  # is_first[i, j] = i < j for i, j in [0, S)
  is_first = jnp.expand_dims(ns, 1) < ns  # [S, S]
  is_first = jnp.expand_dims(is_first, axis=[2, 3, 4])  # [S, S, 1, 1, 1]
  # i is disqualified by j if i < j and action[i] == action[j]
  disqualified = jnp.logical_and(actions_eq, is_first)  # [S, S, T, B, 2]
  is_unique = ~jnp.any(disqualified, axis=1)  # [S, T, B, 2]
  unique_fraction = jnp.mean(is_unique, axis=0)  # [T, B, 2]

  return unique_fraction

def information_fraction(
  payoff_matrices: jax.Array,  # [..., S, S]
  eps: float = 1e-8,
) -> jax.Array:
  """Fraction of a payoff matrix that can't be explained by additive interactions."""
  P = payoff_matrices
  m = P.mean(axis=[-2, -1], keepdims=True)
  P_m = P - m
  r = P_m.mean(axis=-1, keepdims=True)
  c = P_m.mean(axis=-2, keepdims=True)
  I = P_m - r - c

  return I.var(axis=[-2, -1]) / (P_m.var(axis=[-2, -1]) + eps)


def mixed_payoff_matrices(q_values: jax.Array) -> jax.Array:
  """Converts two-player q-values to player-1 payoff matrices.

  Args:
    q_values: [N, S, S, T, B, 2] per-index two-player q-values, indexed by
      (epistemic index, p1 action sample, p2 action sample).
  Returns:
    [N, T, B, S, S] player-1 payoff matrices, averaging the two players'
    estimates assuming a zero-sum game.
  """
  p1_qs, p2_qs = jnp.unstack(q_values, axis=-1)  # [N, S, S, T, B]
  mixed_values = (p1_qs - p2_qs) / 2  # [N, S, S, T, B]
  return jnp.moveaxis(mixed_values, (1, 2), (-2, -1))  # [N, T, B, S, S]


def indexed_nash_metrics(
    nash_probs: jax.Array,  # [N, ..., S] per-index nash distributions
) -> tuple[jax.Array, dict]:
  """Mixture and disagreement metrics for per-index nash distributions.

  Returns (mixture_probs [..., S], metrics).

  Mixture entropy alone conflates a genuinely mixed nash (all indices agree
  on a high-entropy strategy) with epistemic disagreement. The mutual
  information I(action; index) = H(mixture) - E_index[H(nash)] isolates the
  disagreement: it is 0 iff all indices give the same distribution, however
  mixed. Note it is estimated from N index samples and can only underestimate
  the true MI (by concavity of entropy).

  As a complementary, conjugate-prior style dispersion measure, we
  moment-match a Dirichlet to the per-index nash strategies. For
  Dirichlet(alpha) with precision alpha0 = sum(alpha),
  Var(p_s) = pbar_s (1 - pbar_s) / (alpha0 + 1), so
  alpha0 = sum_s pbar_s (1 - pbar_s) / sum_s Var(p_s) - 1. High precision
  means the indices agree tightly; logged on a log scale as the precision
  diverges when the variance vanishes. Unlike the MI it is not bounded by
  log(N), so it still resolves the near-agreement regime.
  """
  mixture_probs = jnp.mean(nash_probs, axis=0)  # [..., S]

  index_entropy = jnp.mean(jax_utils.entropy(nash_probs, axis=-1), axis=0)
  mixture_entropy = jax_utils.entropy(mixture_probs, axis=-1)

  nash_index_var = jnp.sum(jnp.var(nash_probs, axis=0, ddof=1), axis=-1)
  allocated_var = jnp.sum(mixture_probs * (1 - mixture_probs), axis=-1)
  dirichlet_precision = allocated_var / (nash_index_var + 1e-8) - 1

  metrics = dict(
      nash_entropy=mixture_entropy,
      nash_index_entropy=index_entropy,
      nash_index_mi=mixture_entropy - index_entropy,
      nash_index_var=nash_index_var,
      nash_dirichlet_log_precision=jnp.log1p(
          jnp.maximum(dirichlet_precision, 0)),
  )
  return mixture_probs, metrics


class NashPayoffDiagnostics(tp.NamedTuple):
  metrics: dict  # index-averaged, [...] (batch-shaped)
  nash_vs_mean: jax.Array  # [N, ..., 2] per-index
  nash_advantage: jax.Array  # [..., 2] index-averaged


def nash_payoff_diagnostics(
    payoff_matrices: jax.Array,  # [N, T, B, S, S] player-1 payoffs
    nash_probs: jax.Array,  # [N, T, B, 2, S] per-index nash distributions
    nash_values: jax.Array,  # [N, T, B, 2] per-index nash values
) -> NashPayoffDiagnostics:
  """Diagnostics of per-index nash solutions against their payoff matrices.

  All logged metrics are averaged over the epistemic index axis N; the
  per-index nash_vs_mean and the index-averaged nash_advantage are also
  returned for further use (e.g. advantage weighting).
  """
  index_mean = lambda x: jnp.mean(x, axis=0)

  p12_matrices = jnp.stack([
      payoff_matrices,
      -payoff_matrices.swapaxes(-1, -2)],
  axis=-3)  # [N, T, B, 2, S, S]

  def payoffs(
    p: jax.Array,  # [N, T, B, 2, S]
    q: jax.Array,  # [N, T, B, 2, S]
  ) -> jax.Array:  # [N, T, B, 2]
    """Compute payoffs of policy p vs policy q, per epistemic index."""
    return jnp.vecdot(p, jnp.matvec(p12_matrices, jnp.flip(q, axis=-2)))

  num_samples = payoff_matrices.shape[-1]

  vs_mean = p12_matrices.mean(axis=-1)  # [N, T, B, 2, S]
  argmax_policy = jnp.argmax(vs_mean, axis=-1)  # [N, T, B, 2]
  argmax_policy_probs = jax.nn.one_hot(argmax_policy, num_classes=num_samples)
  argmax_vs_mean = jnp.max(vs_mean, axis=-1)  # [N, T, B, 2]

  nash_vs_mean = jnp.vecdot(nash_probs, vs_mean)  # [N, T, B, 2]
  argmax_advantage = argmax_vs_mean - nash_vs_mean

  nash_vs_argmax = payoffs(nash_probs, argmax_policy_probs)
  nash_vs_argmax_advantage = nash_vs_argmax - nash_values

  # Ensemble (index-mean) advantage; also used for advantage weighting.
  nash_advantage = index_mean(nash_vs_mean - nash_values)  # [T, B, 2]
  nash_advantage_std = jnp.std(nash_advantage, keepdims=True)
  nash_advantage_variation = nash_advantage_std / jnp.mean(nash_advantage)
  nash_advantantage_min = jnp.min(nash_advantage, keepdims=True)

  # Test nash solutions; should maybe go in the nash computation itself
  nash_vs_nash = payoffs(nash_probs, nash_probs)  # [N, T, B, 2]
  nash_value_error = jnp.sqrt(
      jnp.square(nash_vs_nash - nash_values).mean(keepdims=True))
  nash_value_error_max = jnp.max(
      jnp.abs(nash_vs_nash - nash_values), keepdims=True)
  vs_nash = jnp.matvec(p12_matrices, jnp.flip(nash_probs, axis=-2))  # [N, T, B, 2, S]
  best_vs_nash = jnp.max(vs_nash, axis=-1)  # [N, T, B, 2]
  nash_suboptimality = best_vs_nash - nash_vs_nash
  nash_suboptimality_max = jnp.max(nash_suboptimality, keepdims=True)

  # Disagreement between indices about the value of the game.
  nash_value_epistemic_std = jnp.std(nash_values, axis=0, ddof=1)  # [T, B, 2]

  metrics = dict(
      nash_advantage=nash_advantage,  # nash-vs-mean - nash-vs-nash
      nash_advantage_std=nash_advantage_std,
      nash_advantage_variation=nash_advantage_variation,
      nash_advantantage_min=nash_advantantage_min,
      argmax_advantage=index_mean(argmax_advantage),  # argmax-vs-mean - nash-vs-mean
      nash_vs_argmax_advantage=index_mean(nash_vs_argmax_advantage),  # nash-vs-argmax - nash-vs-nash
      nash_value_error=nash_value_error,
      nash_value_error_max=nash_value_error_max,
      nash_suboptimality=index_mean(nash_suboptimality),
      nash_suboptimality_max=nash_suboptimality_max,
      nash_value_epistemic_std=nash_value_epistemic_std,
  )

  return NashPayoffDiagnostics(
      metrics=metrics,
      nash_vs_mean=nash_vs_mean,
      nash_advantage=nash_advantage,
  )
