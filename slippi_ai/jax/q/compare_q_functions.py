"""Compare two q-functions on actions sampled from a policy.

Given a (sample) policy and two q-functions, this samples a finite set of
actions from the policy at each frame and evaluates both q-functions on those
actions. We then measure agreement at two levels:

* Relative ordering of the sampled actions (rather than the absolute q-values),
  via Spearman rank correlation, Kendall's tau, and top-1 (argmax) agreement.
* The actual q-values, via ``value_agreement``: how much better one
  q-function's q-values predict the other's q-values than the other's
  state-only value does. Since a state's value is constant across actions, the
  only information one q-function can contribute is its *advantage*
  (``Q(s, a) - V(s)``), so this measures how well the two q-functions'
  advantages line up (see ``value_agreement`` for details).

This reuses the same sampling / q-evaluation machinery as
``q_policy_learner.Learner`` but without any training.
"""

import dataclasses
import typing as tp

from absl import logging
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import tqdm

from slippi_ai import flag_utils, utils, data as data_lib
from slippi_ai.data import Batch, Frames
from slippi_ai.jax import saving, jax_utils, rl_lib, train_policy
from slippi_ai.jax.policies import Policy
from slippi_ai.jax.q import train_q_fn, q_function as q_lib

_field = utils.field

SAMPLE_POLICY = 'sample_policy'
Q_FUNCTION_A = 'q_function_a'
Q_FUNCTION_B = 'q_function_b'

_SAMPLE_AXIS = 0


@dataclasses.dataclass
class Config:
  dataset: data_lib.DatasetConfig = _field(data_lib.DatasetConfig)
  data: data_lib.DataConfig = _field(data_lib.DataConfig)

  # Checkpoints to load. The policy is an imitation-style checkpoint; the
  # q-functions are train_q_fn checkpoints.
  sample_policy: tp.Optional[str] = None
  q_function_a: tp.Optional[str] = None
  q_function_b: tp.Optional[str] = None

  num_samples: int = 8
  # 0 means vmap over all samples at once (see multi_q_values_from_action_state).
  sample_batch_size: int = 0
  # Whether to include the action actually taken among the ranked actions.
  include_action_taken: bool = True

  # Only affects QOutputs.returns/advantages, which we don't use here, but
  # loss_and_action_state requires a discount.
  reward_halflife: float = 4

  num_steps: int = 20  # number of batches to evaluate
  jit_compile: bool = True
  seed: int = 0


def load_q_function(path: str) -> tuple[q_lib.QFunction, train_q_fn.Config]:
  """Loads a q-function and its (train_q_fn) config dict from a checkpoint."""
  state = saving.load_state_from_disk(path)
  config = flag_utils.dataclass_from_dict(
      train_q_fn.Config, state['config'])
  q_function = q_lib.build_q_function(nnx.Rngs(0), config.q_function)
  # load_state_from_disk stores params as numpy; convert to jax arrays.
  params = jax.tree.map(jnp.asarray, state['state']['q_function'])
  jax_utils.set_module_state(q_function, params)
  return q_function, config


def _ranks(x: jax.Array) -> jax.Array:
  """0-indexed ranks along the sample axis."""
  order = jnp.argsort(x, axis=_SAMPLE_AXIS)
  ranks = jnp.argsort(order, axis=_SAMPLE_AXIS)
  return ranks.astype(x.dtype)


def _pearson(a: jax.Array, b: jax.Array) -> jax.Array:
  """Pearson correlation along the sample axis. Returns [T, B]."""
  am = a - jnp.mean(a, axis=_SAMPLE_AXIS, keepdims=True)
  bm = b - jnp.mean(b, axis=_SAMPLE_AXIS, keepdims=True)
  cov = jnp.sum(am * bm, axis=_SAMPLE_AXIS)
  denom = jnp.sqrt(
      jnp.sum(jnp.square(am), axis=_SAMPLE_AXIS)
      * jnp.sum(jnp.square(bm), axis=_SAMPLE_AXIS))
  return cov / (denom + 1e-8)


def rank_agreement(qa: jax.Array, qb: jax.Array) -> dict[str, jax.Array]:
  """Agreement between two q-functions' orderings of S samples.

  Args:
    qa: [S, T, B] q-values from the first q-function.
    qb: [S, T, B] q-values from the second q-function.

  Returns:
    Dict of [T, B] metrics: spearman, kendall_tau, top1_agreement.
  """
  spearman = _pearson(_ranks(qa), _ranks(qb))

  # Kendall's tau-b over the S axis: (n_c - n_d) / sqrt((n0 - n1)(n0 - n2)),
  # where n0 is the number of unordered pairs, n1/n2 the number of tied pairs
  # in qa/qb. tau-b correctly handles ties (identical inputs give tau-b == 1).
  num_samples = qa.shape[_SAMPLE_AXIS]
  idx = jnp.arange(num_samples)
  pair_mask = (idx[:, None] < idx[None, :])[:, :, None, None]  # [S, S, 1, 1]
  sign_a = jnp.sign(qa[:, None] - qa[None, :])  # [S, S, T, B]
  sign_b = jnp.sign(qb[:, None] - qb[None, :])
  concordance = sign_a * sign_b  # +1 concordant, -1 discordant, 0 if either ties
  nc_minus_nd = jnp.sum(concordance * pair_mask, axis=(0, 1))  # [T, B]
  n0 = jnp.sum(pair_mask)  # scalar
  n1 = jnp.sum((sign_a == 0) * pair_mask, axis=(0, 1))  # tied pairs in qa
  n2 = jnp.sum((sign_b == 0) * pair_mask, axis=(0, 1))  # tied pairs in qb
  kendall_tau = nc_minus_nd / (jnp.sqrt((n0 - n1) * (n0 - n2)) + 1e-8)

  top1_agreement = (
      jnp.argmax(qa, axis=_SAMPLE_AXIS) == jnp.argmax(qb, axis=_SAMPLE_AXIS)
  ).astype(jnp.float32)

  return dict(
      spearman=spearman,
      kendall_tau=kendall_tau,
      top1_agreement=top1_agreement,
  )


def value_agreement(
    qa: jax.Array,
    qb: jax.Array,
    va: jax.Array,
    vb: jax.Array,
) -> dict[str, jax.Array]:
  """Agreement between two q-functions' actual q-values.

  We ask how much better one q-function's q-values predict the other's q-values
  than the other's *state-only* value does. Because a state's value V(s) is
  constant across actions, the only thing one q-function can contribute to
  predicting another's q-values (beyond V) is its advantage A(s, a) = Q - V.

  This assumes the two q-functions are on the same reward scale (true when they
  share a discount / reward halflife; ``compare`` warns when they don't). To
  predict q-function B's q-values we compare two predictors *at unit scale*:
    - baseline: ``vb`` (B's state value, constant across the S sampled actions),
      with error ``sum_S (qb - vb)^2 = sum_S adv_b^2``;
    - candidate: ``vb + adv_a`` (B's state value plus A's advantage, unscaled).

  This returns per-element second moments (summed over the sample axis) rather
  than a finished R^2, because R^2 must be *pooled* across states -- i.e. summed
  numerator over summed denominator -- not averaged per state. A per-state R^2
  is unbounded below and blows up at states where B values the sampled actions
  almost equally (``sq_b ~ 0``), so a plain mean of per-state R^2 is dominated
  by those states. ``compare`` forms the pooled ``r2_b_from_a`` /
  ``r2_a_from_b`` (fraction of one q-function's action-induced variance
  explained by the other's q-values; negative when worse than the state value),
  ``advantage_cosine_pooled`` (the variance-weighted cosine; the cosine of the
  advantage vectors stacked across all states), and ``advantage_rms_a`` /
  ``advantage_rms_b`` (advantage scales) from these.

  ``advantage_cosine`` is the scale-invariant, signed alignment of the advantage
  vectors and is bounded, so it is reported as a per-state mean directly. It
  weights every state equally, whereas ``advantage_cosine_pooled`` weights each
  state by its advantage magnitude (so high-stakes states dominate).

  Args:
    qa: [S, T, B] q-values from the first q-function.
    qb: [S, T, B] q-values from the second q-function.
    va: [T, B] state-only values from the first q-function.
    vb: [T, B] state-only values from the second q-function.

  Returns:
    Dict of [T, B] metrics: advantage_cosine plus the second moments
    _sq_residual, _sq_a, _sq_b (combined into R^2 / RMS by ``compare``).
  """
  adv_a = qa - jnp.expand_dims(va, axis=_SAMPLE_AXIS)  # [S, T, B]
  adv_b = qb - jnp.expand_dims(vb, axis=_SAMPLE_AXIS)

  dot = jnp.sum(adv_a * adv_b, axis=_SAMPLE_AXIS)  # [T, B]
  sq_a = jnp.sum(jnp.square(adv_a), axis=_SAMPLE_AXIS)
  sq_b = jnp.sum(jnp.square(adv_b), axis=_SAMPLE_AXIS)
  # Residual sum of squares from predicting one advantage with the other at unit
  # scale; the same for both directions.
  residual = jnp.sum(jnp.square(adv_a - adv_b), axis=_SAMPLE_AXIS)

  cosine = dot / (jnp.sqrt(sq_a * sq_b) + 1e-8)

  q_loss = jnp.square(qa - qb)
  v_loss = jnp.square(va - vb)

  q_rel_v_ab = jnp.log(q_loss / jnp.square(adv_b)).mean()
  q_rel_v_ba = jnp.log(q_loss / jnp.square(adv_a)).mean()

  return dict(
      advantage_cosine=cosine,
      _sq_residual=residual,
      _sq_a=sq_a,
      _sq_b=sq_b,
      q_loss=q_loss,
      v_loss=v_loss,
      q_rel_v_ab=q_rel_v_ab,
      q_rel_v_ba=q_rel_v_ba,
  )


def policy_regret(qa: jax.Array, qb: jax.Array) -> dict[str, jax.Array]:
  """Regret from acting by one q-function but scoring with the other.

  For each state we pick the action that maximizes the *chooser* q-function
  among the S samples, then measure its regret under the *truth* q-function: how
  much worse it is than the action the truth would have picked. We normalize by
  the truth's own spread between its best and its mean sampled q-value, so 0
  means the chooser selects the truth's best action and 1 means it does no
  better than an average (random) sample. (Per state the ratio can exceed 1 when
  the chooser picks worse-than-average actions.)

  This is scale-invariant in the truth q-function (numerator and denominator
  share its units), so unlike value_agreement it needs no shared-scale
  assumption. Returns per-[T, B] numerators and denominators; ``compare`` pools
  them (sum numerator / sum denominator) across all states.

  Args:
    qa: [S, T, B] q-values from the first q-function.
    qb: [S, T, B] q-values from the second q-function.

  Returns:
    Dict of [T, B] second moments: _regret_num_a_b, _regret_den_b (choose by A,
    truth B) and _regret_num_b_a, _regret_den_a (choose by B, truth A).
  """
  def regret(chooser: jax.Array, truth: jax.Array):
    chosen_idx = jnp.argmax(chooser, axis=_SAMPLE_AXIS, keepdims=True)
    truth_chosen = jnp.take_along_axis(
        truth, chosen_idx, axis=_SAMPLE_AXIS)[0]  # [T, B]
    truth_best = jnp.max(truth, axis=_SAMPLE_AXIS)
    truth_mean = jnp.mean(truth, axis=_SAMPLE_AXIS)
    return truth_best - truth_chosen, truth_best - truth_mean

  num_a_b, den_b = regret(qa, qb)  # choose by A, evaluate under truth B
  num_b_a, den_a = regret(qb, qa)  # choose by B, evaluate under truth A
  return dict(
      _regret_num_a_b=num_a_b,
      _regret_den_b=den_b,
      _regret_num_b_a=num_b_a,
      _regret_den_a=den_a,
  )


class Comparator(nnx.Module):

  def __init__(
      self,
      config: Config,
      sample_policy: Policy,
      q_function_a: q_lib.QFunction,
      q_function_b: q_lib.QFunction,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.sample_policy = sample_policy
    self.q_function_a = q_function_a
    self.q_function_b = q_function_b
    self.rngs = rngs

    self.frame_skip = sample_policy.frame_skip
    assert q_function_a.frame_skip == self.frame_skip
    assert q_function_b.frame_skip == self.frame_skip

    self.delay = sample_policy.delay
    assert self.delay == 0, 'Only delay == 0 is supported.'

    self.discount = rl_lib.discount_from_halflife(
        config.reward_halflife, frame_skip=self.frame_skip)

    self.num_samples = config.num_samples
    self.include_action_taken = config.include_action_taken
    self.sample_batch_size = config.sample_batch_size

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> dict:
    return {
        SAMPLE_POLICY: self.sample_policy.initial_state(batch_size, rngs),
        Q_FUNCTION_A: self.q_function_a.initial_state(batch_size, rngs),
        Q_FUNCTION_B: self.q_function_b.initial_state(batch_size, rngs),
    }

  def _encode(self, network, frames: Frames) -> Frames:
    return Frames(
        state_action=network.encode(frames.state_action),
        is_resetting=frames.is_resetting,
        reward=frames.reward,
    )

  def prepare_frames(self, batch: Batch) -> dict[str, Frames]:
    # batch-major frames; encoded separately for each network.
    frames = batch.to_frames(self.frame_skip)
    return {
        SAMPLE_POLICY: self._encode(self.sample_policy.network, frames),
        Q_FUNCTION_A: self._encode(self.q_function_a.core_net, frames),
        Q_FUNCTION_B: self._encode(self.q_function_b.core_net, frames),
    }

  def _q_values_for_samples(
      self,
      q_function: q_lib.QFunction,
      frames: Frames,  # time-major, encoded for this q-function
      initial_state,
      policy_samples: list,  # frame_skip x [S, T, B]
  ):
    q_outputs, action_init_state, final_state = q_function.loss_and_action_state(
        frames, initial_state, self.discount)

    sample_q_values = q_function.multi_q_values_from_action_state(
        values=q_outputs.values,
        action_init_state=action_init_state,
        actions=policy_samples,
        batch_size=self.sample_batch_size,
    )  # [S, T, B]

    if self.include_action_taken:
      # q_outputs.q_values is the q-value of the action actually taken; the
      # taken action is the same for both q-functions, so this index aligns.
      sample_q_values = jnp.concatenate(
          [sample_q_values,
           jnp.expand_dims(q_outputs.q_values, axis=_SAMPLE_AXIS)],
          axis=_SAMPLE_AXIS)

    return sample_q_values, q_outputs.values, final_state

  def step(
      self,
      bm_frames: dict[str, Frames],  # batch-major
      initial_states: dict,
  ) -> tuple[dict, dict]:
    final_states = {}

    frames = {
        k: jax.tree.map(jax_utils.swap_axes, v) for k, v in bm_frames.items()
    }

    # Unroll the sample policy and draw num_samples actions per frame.
    sp_frames = frames[SAMPLE_POLICY]
    sp_outputs = self.sample_policy.unroll_with_outputs(
        sp_frames, initial_states[SAMPLE_POLICY])
    final_states[SAMPLE_POLICY] = sp_outputs.final_state

    prev_action = jax.tree.map(
        lambda t: t[:-1], sp_frames.state_action.action)

    @nnx.vmap(in_axes=(None, 0), out_axes=_SAMPLE_AXIS)
    def sample(sample_policy: Policy, rngs: nnx.Rngs):
      sample_outputs = sample_policy.controller_head.sample(
          rngs=rngs,
          inputs=sp_outputs.outputs,
          prev_controller_state=prev_action)
      return [so.controller_state for so in sample_outputs]

    policy_samples = sample(
        self.sample_policy, self.rngs.fork(split=self.num_samples))

    qa, va, final_states[Q_FUNCTION_A] = self._q_values_for_samples(
        self.q_function_a, frames[Q_FUNCTION_A],
        initial_states[Q_FUNCTION_A], policy_samples)
    qb, vb, final_states[Q_FUNCTION_B] = self._q_values_for_samples(
        self.q_function_b, frames[Q_FUNCTION_B],
        initial_states[Q_FUNCTION_B], policy_samples)

    metrics = rank_agreement(qa, qb)
    metrics.update(value_agreement(qa, qb, va, vb))
    metrics.update(policy_regret(qa, qb))
    return metrics, final_states


def _check_compatible(name: str, a, b):
  if a != b:
    raise ValueError(f'Incompatible {name}: {a} vs {b}')


def compare(config: Config) -> dict[str, float]:
  if not config.sample_policy:
    raise ValueError('Must specify sample_policy_path.')
  if not config.q_function_a or not config.q_function_b:
    raise ValueError('Must specify both q_function_a_path and q_function_b_path.')

  rngs = nnx.Rngs(config.seed)

  # Load policy.
  policy_state = saving.load_state_from_disk(config.sample_policy)
  sample_policy = saving.load_policy_from_state(policy_state)
  policy_config = flag_utils.dataclass_from_dict(
      train_policy.Config, saving.upgrade_config(policy_state['config']))
  name_map = policy_state['name_map']

  # Load q-functions.
  q_function_a, q_config_a = load_q_function(config.q_function_a)
  q_function_b, q_config_b = load_q_function(config.q_function_b)

  # value_agreement assumes both q-functions share a reward scale, which is set
  # by the training discount (reward halflife). Warn if they differ, since the
  # R^2 metrics would then be conflated with a scale mismatch.
  halflife_a = q_config_a.learner.reward_halflife
  halflife_b = q_config_b.learner.reward_halflife
  if halflife_a != halflife_b:
    logging.warning(
        'q-functions trained with different reward halflives (%s vs %s); their '
        'q-value scales likely differ, so value_agreement R^2 metrics will be '
        'distorted. advantage_cosine and advantage_rms_a/b remain meaningful.',
        halflife_a, halflife_b)

  config.dataset.copy_characteristics_from(policy_config.dataset)

  # Sanity checks: all networks must agree on the raw observation so that the
  # data pipeline produces frames each network can encode.
  _check_compatible('observation', policy_config.observation, q_config_a.observation)
  _check_compatible('observation', policy_config.observation, q_config_b.observation)
  _check_compatible(
      'frame_skip', sample_policy.frame_skip, q_function_a.frame_skip)
  _check_compatible(
      'frame_skip', sample_policy.frame_skip, q_function_b.frame_skip)

  comparator = Comparator(
      config=config,
      sample_policy=sample_policy,
      q_function_a=q_function_a,
      q_function_b=q_function_b,
      rngs=rngs,
  )

  frame_skip = comparator.frame_skip
  config.data.random_offset = frame_skip

  sources = data_lib.build_sources(
      dataset_config=config.dataset,
      train_data_config=config.data,
      name_map=name_map,
      extra_frames=comparator.delay + frame_skip,
      observation_config=policy_config.observation,
  )
  data_source = sources.test

  if config.jit_compile:
    step_fn = jax_utils.cached_partial(
      jax_utils.nnx_jit(Comparator.step, donate_argnums=(0, 2)),
      comparator,
    )
  else:
    step_fn = comparator.step

  hidden_state = comparator.initial_state(data_source.batch_size, rngs)

  per_step: list[dict] = []

  def pull_last_step_metrics():
    if not per_step:
      return

    jax_metrics = per_step.pop()
    np_metrics = jax.device_get(jax_metrics)
    means = {k: float(np.mean(v)) for k, v in np_metrics.items()}
    per_step.append(means)

  try:
    for _ in tqdm.trange(config.num_steps):
      pull_last_step_metrics()

      batch_with_meta, _ = next(data_source)
      frames = comparator.prepare_frames(batch_with_meta.batch)
      metrics, hidden_state = step_fn(frames, hidden_state)
      per_step.append(metrics)
  finally:
    data_source.shutdown()

  pull_last_step_metrics()

  keys = per_step[0].keys()
  summary = {
      k: float(sum(s[k] for s in per_step) / len(per_step)) for k in keys
  }

  # Pool the value-agreement second moments into R^2 / RMS. Each step contains
  # the same number of [T, B] elements, so the mean of a second moment over all
  # steps equals its global mean, and a ratio of these means is the pooled
  # quantity (summed numerator over summed denominator).
  num_samples = config.num_samples + int(config.include_action_taken)
  sq_residual = summary.pop('_sq_residual')
  sq_a = summary.pop('_sq_a')
  sq_b = summary.pop('_sq_b')
  summary['r2_b_from_a'] = 1.0 - sq_residual / (sq_b + 1e-8)
  summary['r2_a_from_b'] = 1.0 - sq_residual / (sq_a + 1e-8)
  # Pooled (variance-weighted) cosine: the cosine of the two advantage vectors
  # stacked across all states, equivalently each state weighted by its advantage
  # magnitude. Recovered from the pooled second moments via the polarization
  # identity dot = (sq_a + sq_b - residual) / 2. Unlike advantage_cosine (a flat
  # mean of per-state cosines), this is dominated by high-advantage states.
  dot = 0.5 * (sq_a + sq_b - sq_residual)
  summary['advantage_cosine_pooled'] = dot / ((sq_a * sq_b) ** 0.5 + 1e-8)
  summary['advantage_rms_a'] = (sq_a / num_samples) ** 0.5
  summary['advantage_rms_b'] = (sq_b / num_samples) ** 0.5

  # Pooled normalized regret: sum of regrets over sum of (best - mean) spreads.
  # regret_a_b = argmax by A, scored under truth B (and vice versa).
  summary['regret_a_b'] = (
      summary.pop('_regret_num_a_b') / (summary.pop('_regret_den_b') + 1e-8))
  summary['regret_b_a'] = (
      summary.pop('_regret_num_b_a') / (summary.pop('_regret_den_a') + 1e-8))

  print('=== q-function agreement (mean over %d steps) ===' % len(per_step))
  for k, v in summary.items():
    print(f'  {k}: {v:.4f}')

  return summary
