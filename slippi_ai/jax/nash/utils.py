import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx

from slippi_ai import utils
from slippi_ai.jax import jax_utils, networks
from slippi_ai.jax.policies import Policy
from slippi_ai.types import Action, Frames, StateAction, S, SkipAction

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


# Chains of actions for delayed nash training (docs/plans/nash_policy_delay.md).
#
# With skip-delay Ds, the game at index t is over chains of Ds + 1 actions
# replacing the committed actions [t + 1, t + Ds + 1]. In the delayed-frames
# alignment (data.delayed_frames) the action at index j is the original action
# j + Ds, so a chain at index t replaces the delayed-frame actions
# [t - Ds + 1, t + 1]. A chain is scored or sampled by re-running the network
# from its hidden state after index t - Ds over the real inputs at indices
# [t - Ds + 1, t] with the chain's first Ds actions in place of the real ones,
# after which the network's output at index t predicts or scores the last
# action. Indices t < Ds have no such hidden state within the chunk, so the
# game is over the T' = T - Ds valid indices t = Ds + t' for t' in [0, T').
# The nash policy trainer overlaps consecutive chunks by Ds extra steps so
# that a chunk's first Ds indices are the previous chunk's last Ds valid ones
# (nothing is lost), and the learner carries the hidden state after index
# T' - 1, where the next chunk starts.
#
# Shapes below: T is the number of (delayed) steps in a chunk, T' = T - Ds
# the number of valid indices, and "..." the batch shape ([B, 2] for
# two-player frames). Chains are lists of skip_delay + 1 actions, each a list
# of frame_skip controllers, whose leaves are [T', ...] for a single chain
# and [S, T', ...] for S sampled chains (the sample axis is the leading one).

type Chain[Action] = list[SkipAction[Action]]  # (skip_delay + 1) x (frame_skip x Controller)


def time_slices(nest: T, length: int, offsets: tp.Iterable[int]) -> list[T]:
  """Slices [offset, offset + length) along the leading (time) axis.

  Args:
    nest: Leaves of shape [T, ...].
    length: Length of each slice.
    offsets: Starting index of each slice.

  Returns:
    One nest per offset, with leaves of shape [length, ...].
  """
  return [jax.tree.map(lambda x: x[o:o + length], nest) for o in offsets]


def flatten_chain(chain: Chain[Action]) -> list[Action]:
  """Flattens a chain into its (skip_delay + 1) * frame_skip controllers."""
  return [controller for action in chain for controller in action]


class RerunStep(tp.NamedTuple, tp.Generic[Action]):
  """A network's output at one index of a chain re-run, with the action it
  was fed there (the previous action in the chain)."""
  outputs: jax.Array  # [T', ..., O]
  prev_action: SkipAction[Action]  # [T', ...]


class ChainContext(tp.NamedTuple, tp.Generic[S, T, Action]):
  """What a network needs to re-run over a chain at each valid index.

  All leaves have leading shape [T', ...], one entry per valid index
  t = Ds + t'. To re-run at index t, start from hidden_states (after index
  t - Ds), sample or score the chain's first action against outputs, then
  for k in [0, Ds) step the network on inputs[k] with the chain's k-th action
  in place of the real one and sample or score the (k + 1)-th action against
  the new output. See rerun.
  """
  outputs: jax.Array  # [T', ..., O] network outputs at index t - Ds
  hidden_states: T  # [T', ...] hidden states after index t - Ds
  prev_action: SkipAction[Action]  # [T', ...] the (real) action at index t - Ds
  inputs: list[StateAction[S, Action]]  # Ds x [T', ...] real inputs at indices [t - Ds + 1, t]
  resets: list[jax.Array]  # Ds x [T', ...] is_resetting at indices [t - Ds + 1, t]

  @property
  def skip_delay(self) -> int:
    return len(self.inputs)

  def rerun(
      self,
      network: networks.StateActionNetwork[Action],
      act: tp.Callable[[int, jax.Array, SkipAction[Action]], SkipAction[Action]],
  ) -> list[RerunStep[Action]]:
    """Re-runs the network over a chain chosen one action at a time.

    Starting from the output and hidden state at index t - Ds, for k in
    [0, Ds) the chain's k-th action is act(k, outputs, prev_action), given
    the output at index t - Ds + k and the action before it, and the network
    is then stepped on inputs[k] with that action in place of the real one
    (honouring resets[k]). Returns the Ds + 1 outputs at indices [t - Ds, t],
    each paired with the action the network was fed there, i.e. what the
    chain's k-th action is sampled or scored against; the last output
    conditions on the whole chain prefix. With Ds = 0 nothing is re-run and
    the single step is the context's own output and previous action.
    """
    step = RerunStep(self.outputs, self.prev_action)
    hidden_state = self.hidden_states
    steps = [step]
    for k, (inputs, reset) in enumerate(zip(self.inputs, self.resets)):
      action = act(k, step.outputs, step.prev_action)
      outputs, hidden_state = network.step_with_reset(
          inputs._replace(action=action), reset, hidden_state)
      step = RerunStep(outputs, action)
      steps.append(step)
    return steps

  def rerun_chain(
      self,
      network: networks.StateActionNetwork[Action],
      chain: tp.Sequence[SkipAction[Action]],  # at least Ds actions
  ) -> list[RerunStep[Action]]:
    """Re-runs the network over a given chain's first Ds actions."""
    if len(chain) < self.skip_delay:
      raise ValueError(
          f'Expected at least {self.skip_delay} actions, got {len(chain)}.')
    return self.rerun(network, lambda k, outputs, prev_action: chain[k])


def chain_context(
    outputs: jax.Array,  # [T, ..., O]
    hidden_states: T,  # [T, ...]
    frames: Frames[S, Action],  # T + 1 states and actions, T rewards
    skip_delay: int,
) -> ChainContext[S, T, Action]:
  """Builds the ChainContext from a network's scan over T steps of frames."""
  num_valid = frames.reward.shape[0] - skip_delay
  keep = lambda x: x[:num_valid]
  return ChainContext(
      outputs=keep(outputs),
      hidden_states=jax.tree.map(keep, hidden_states),
      prev_action=jax.tree.map(keep, frames.state_action.action),
      inputs=time_slices(
          frames.state_action, num_valid, range(1, skip_delay + 1)),
      resets=time_slices(
          frames.is_resetting, num_valid, range(1, skip_delay + 1)),
  )


def taken_chain(
    frames: Frames[S, Action],  # T + 1 states and actions, T rewards
    skip_delay: int,
) -> Chain[Action]:  # leaves [T', ...]
  """The chain of actions actually taken at each valid index t, i.e. the
  frames' actions at indices [t - Ds + 1, t + 1]."""
  num_valid = frames.reward.shape[0] - skip_delay
  return time_slices(
      frames.state_action.action, num_valid, range(1, skip_delay + 2))


def sample_chain(
    policy: Policy[Action],
    rngs: nnx.Rngs,
    context: ChainContext[S, T, Action],
) -> Chain[Action]:  # leaves [T', ...]
  """Samples a chain from the policy by re-running it on its own samples.

  Each of the Ds + 1 actions is sampled from the network's output at index
  t - Ds + k given the previous (sampled) action, exactly as the policy
  would act online; only the states are the real ones. With Ds = 0 this is
  a single head sample from context.outputs.
  """
  chain: Chain[Action] = []

  def sample(k: int, outputs: jax.Array, prev_action: SkipAction[Action]):
    del k
    sample_outputs = policy.controller_head.sample(rngs, outputs, prev_action)
    chain.append([so.controller_state for so in sample_outputs])
    return chain[-1]

  steps = context.rerun(policy.network, sample)
  # The last action, from the output at index t.
  sample(len(chain), *steps[-1])
  return chain


def chain_log_prob(
    policy: Policy[Action],
    context: ChainContext[S, T, Action],
    chain: Chain[Action],  # leaves [T', ...]
) -> jax.Array:  # [T', ...]
  """Log-probability of a chain under the policy, re-running it on the chain.

  Mirrors sample_chain: the k-th action is scored against the network's
  output at index t - Ds + k given the previous action. Each action's
  log-prob is the mean over its frame-skipped controllers (as in the
  policy's imitation loss); the chain's is their sum. With the chain
  actually taken this equals the sum of the per-step imitation log-probs
  over indices [t - Ds, t] (tests/nash_chain_test.py).
  """
  if len(chain) != context.skip_delay + 1:
    raise ValueError(
        f'Expected a chain of {context.skip_delay + 1} actions, got {len(chain)}.')
  steps = context.rerun_chain(policy.network, chain)

  distances = []
  for step, action in zip(steps, chain):
    controller_distances = policy.controller_head.distance(
        step.outputs, step.prev_action, action)
    distances.append(
        jax_utils.add_n(controller_distances) / len(controller_distances))

  return -jax_utils.add_n(distances)
