"""Compare two q-functions by how they rank actions sampled from a policy.

Given a (sample) policy and two q-functions, this samples a finite set of
actions from the policy at each frame and evaluates both q-functions on those
actions. We then measure how much the two q-functions agree on the *relative
ordering* of the sampled actions (rather than the absolute q-values), via
Spearman rank correlation, Kendall's tau, and top-1 (argmax) agreement.

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

    return sample_q_values, final_state

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

    qa, final_states[Q_FUNCTION_A] = self._q_values_for_samples(
        self.q_function_a, frames[Q_FUNCTION_A],
        initial_states[Q_FUNCTION_A], policy_samples)
    qb, final_states[Q_FUNCTION_B] = self._q_values_for_samples(
        self.q_function_b, frames[Q_FUNCTION_B],
        initial_states[Q_FUNCTION_B], policy_samples)

    metrics = rank_agreement(qa, qb)
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

  print('=== q-function ordering agreement (mean over %d steps) ===' % len(per_step))
  for k, v in summary.items():
    print(f'  {k}: {v:.4f}')

  return summary
