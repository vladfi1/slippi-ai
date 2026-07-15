import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from slippi_ai import utils
from slippi_ai.data import Batch, Frames, Controller
from slippi_ai.jax.agents import DType
from slippi_ai.jax.policies import Policy, RecurrentState
from slippi_ai.jax.q import q_function as q_lib
from slippi_ai.jax import embed, rl_lib, jax_utils, networks
from slippi_ai.jax.jax_utils import PS, DATA_AXIS

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  reward_halflife: float = 4

  num_samples: int = 1
  sample_batch_size: int = 0  # 0 means full batch size, i.e. vmap
  include_action_taken_in_samples: bool = True

  # Number of epistemic indices to sample. The q_policy regresses to the
  # uniform distribution over the per-index argmax actions.
  num_index_samples: int = 1

  q_policy_argmax_weight: float = 1
  q_policy_imitation_weight: float = 0

  sample_policy_dtype: DType = DType.FP32
  q_fn_dtype: DType = DType.FP32
  q_policy_dtype: DType = DType.FP32

_SAMPLE_AXIS = 0

Loss = jax.Array
Rank2 = tuple[int, int]

class ShardingKwargs(tp.TypedDict):
  mesh: jax.sharding.Mesh
  explicit_pmean: bool
  smap_optimizer: bool

class ShardingSpecs(tp.TypedDict):
  extra_in_specs: tp.Optional[tp.Sequence[PS]]
  extra_out_specs: tp.Optional[tp.Sequence[PS]]

class ShardingSpecs2(tp.TypedDict):
  in_specs: tp.Sequence[jax_utils.Axis]
  out_specs: tp.Sequence[jax_utils.Axis]


SAMPLE_POLICY = 'sample_policy'
Q_FUNCTION = 'q_function'
Q_POLICY = 'q_policy'

def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
  masked_sum = jnp.sum(x * mask, keepdims=True)
  count = jnp.sum(mask)
  return masked_sum / (count + 1e-8)

class Learner(nnx.Module, tp.Generic[embed.Action]):

  def __init__(
      self,
      config: LearnerConfig,
      q_function: q_lib.QFunction[embed.Action],
      sample_policy: Policy[embed.Action],  # trained via imitation
      q_policy: Policy[embed.Action],  # trained to maximize q_function outputs
      rngs: nnx.Rngs,  # used for sampling
      mesh: jax.sharding.Mesh,
      data_sharding: jax.sharding.NamedSharding,
      explicit_pmean: bool = False,
      smap_optimizer: bool = True,
      q_policy_optimizer_state: tp.Optional[tp.Any] = None,
  ):
    self.config = config
    self.q_function = q_function
    self.sample_policy = sample_policy
    self.q_policy = q_policy

    self.frame_skip = q_function.frame_skip
    assert sample_policy.frame_skip == self.frame_skip
    assert q_policy.frame_skip == self.frame_skip

    self.discount = rl_lib.discount_from_halflife(
      config.reward_halflife, frame_skip=self.frame_skip)

    learning_rate = config.learning_rate

    self.q_policy_optimizer = nnx.Optimizer(
        q_policy, optax.adam(learning_rate), wrt=nnx.Param)

    if q_policy_optimizer_state is not None:
      jax_utils.set_module_state(self.q_policy_optimizer, q_policy_optimizer_state)

    if not config.include_action_taken_in_samples and config.num_samples < 2:
      raise ValueError('num_samples must be at least 2 if not including action taken in samples')

    if config.num_index_samples < 1:
      raise ValueError('num_index_samples must be at least 1')

    self.num_samples = config.num_samples
    self.include_action_taken_in_samples = config.include_action_taken_in_samples
    self.q_policy_argmax_weight = config.q_policy_argmax_weight
    self.q_policy_imitation_weight = config.q_policy_imitation_weight

    self.delay = q_policy.delay
    assert sample_policy.delay == self.delay
    assert self.delay == 0

    self.cast_module_dtypes()

    jax_utils.replicate_module(self, mesh)

    self.data_sharding = data_sharding
    sharding_kwargs = ShardingKwargs(
        mesh=mesh,
        explicit_pmean=explicit_pmean,
        smap_optimizer=smap_optimizer,
    )

    tms_specs = [None, DATA_AXIS]
    TM = PS(*tms_specs)  # time-major
    tms_specs.insert(_SAMPLE_AXIS, None)
    TMS = PS(*tms_specs)  # time-major with samples

    sample_policy_specs = ShardingSpecs(
        extra_in_specs=None,
        extra_out_specs=(TMS,),  # policy samples
    )

    self.run_sample_policy = jax_utils.shard_map_loss_fn_with_rngs(
        module=self.sample_policy,
        rngs=rngs,
        loss_fn=self._unroll_sample_policy,
        mesh=mesh,
        **sample_policy_specs,
    )

    q_function_specs = ShardingSpecs(
        extra_in_specs=(TMS,),  # policy samples
        # core_outputs, distribution, mean_taken_q_value
        extra_out_specs=(TM, TMS, TM),
    )
    self.run_q_function = jax_utils.shard_map_loss_fn_with_rngs(
        module=self.q_function,
        rngs=rngs,
        loss_fn=self._unroll_q_function,
        mesh=mesh,
        **q_function_specs,
    )

    q_policy_specs = ShardingSpecs2(
        # q_function, frames, state, core_outputs, policy_samples, distribution, mean_taken_q_value
        in_specs=(None, 0, 0, TM, TMS, TMS, TM),
        # (loss omitted), metrics, state
        out_specs=(0, 0),
    )

    # The q_policy's master weights stay fp32; they are cast to the compute
    # dtype at each step, with gradients flowing back as fp32.
    unroll_q_policy = jax_utils.with_compute_dtype(
        self._unroll_q_policy, config.q_policy_dtype.dtype)

    train_q_policy = jax_utils.sharded_train_with_rngs(
        loss_fn=unroll_q_policy,
        **sharding_kwargs,
        **q_policy_specs,
    )
    jit_train_q_policy = nnx.jit(
        train_q_policy,
        donate_argnums=(0, 1, 2, 3, 5),
    )
    self.train_q_policy = jax_utils.cached_partial(
        jit_train_q_policy,
        self.q_policy, self.q_policy_optimizer, rngs, self.q_function,
    )

    run_q_policy = jax_utils.sharded_run_with_rngs(
        fn=jax_utils.no_loss(unroll_q_policy),
        mesh=mesh,
        **q_policy_specs,
    )
    jit_run_q_policy = nnx.jit(
        run_q_policy,
        donate_argnums=(0, 1, 2, 4),
    )

    self.run_q_policy = jax_utils.cached_partial(
        jit_run_q_policy,
        self.q_policy, rngs, self.q_function,
    )

  def cast_module_dtypes(self):
    """Casts the frozen (non-trained) modules to their configured dtypes.

    The q_policy is trained, so its master weights stay fp32 and are cast to
    q_policy_dtype at each step instead (see with_compute_dtype).

    Must be called again if the module states are overwritten, e.g. when
    restoring from a checkpoint.
    """
    jax_utils.cast_module_state_to_dtype(
        self.sample_policy, self.config.sample_policy_dtype.dtype)
    jax_utils.cast_module_state_to_dtype(
        self.q_function, self.config.q_fn_dtype.dtype)

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    initial_states = {
        Q_FUNCTION: self.q_function.initial_state(batch_size, rngs),
        Q_POLICY: self.q_policy.initial_state(batch_size, rngs),
        SAMPLE_POLICY: self.sample_policy.initial_state(batch_size, rngs),
    }

    dtypes = {
        Q_FUNCTION: self.config.q_fn_dtype,
        Q_POLICY: self.config.q_policy_dtype,
        SAMPLE_POLICY: self.config.sample_policy_dtype,
    }
    for key, dtype in dtypes.items():
      initial_states[key] = jax_utils.cast_floats_to_dtype(
          initial_states[key], dtype.dtype)

    return initial_states

  def _get_delayed_frames(self, frames: Frames[Rank2, embed.Action]) -> Frames[Rank2, embed.Action]:
    # delay == 0, so this is a no-op; kept for parity with nash.
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

  def _encode(
      self,
      network: networks.StateActionNetwork[embed.Action],
      frames: Frames[Rank2, Controller],
  ) -> Frames[Rank2, embed.Action]:
    return Frames(
        state_action=network.encode(frames.state_action),
        is_resetting=frames.is_resetting,
        reward=frames.reward,
    )

  def prepare_frames(self, batch: Batch[Rank2]) -> dict[str, Frames[Rank2, embed.Action]]:
    frames = batch.to_frames(self.frame_skip)

    return {
        SAMPLE_POLICY: self._encode(self.sample_policy.network, frames),
        Q_FUNCTION: self._encode(self.q_function.core_net, frames),
        Q_POLICY: self._encode(self.q_policy.network, frames),
    }

  def _unroll_sample_policy(
      self,
      sample_policy: Policy[embed.Action],
      bm_frames: Frames[Rank2, embed.Action],
      initial_states: RecurrentState,
      rngs: nnx.Rngs,
  ) -> tuple[Loss, dict, RecurrentState, list[embed.Action]]:
    frames = jax.tree.map(jax_utils.swap_axes, bm_frames)
    frames = self._get_delayed_frames(frames)

    action = frames.state_action.action
    prev_action = jax.tree.map(lambda t: t[:-1], action)

    sample_policy_outputs = sample_policy.unroll_with_outputs(frames, initial_states)

    # Because the action space is too large, we compute a finite subsample
    # using the sample_policy.

    @nnx.vmap(in_axes=(None, 0), out_axes=_SAMPLE_AXIS)
    def sample(sample_policy: Policy[embed.Action], rngs: nnx.Rngs):
      sample_outputs = sample_policy.controller_head.sample(
          rngs=rngs,
          inputs=sample_policy_outputs.outputs,
          prev_controller_state=prev_action)
      return [so.controller_state for so in sample_outputs]

    policy_samples = sample(sample_policy, rngs.fork(split=self.num_samples))

    bm_loss = jnp.mean(sample_policy_outputs.imitation_loss, axis=0)
    bm_metrics = jax.tree.map(jax_utils.swap_axes, sample_policy_outputs.metrics)

    return (
        bm_loss,
        bm_metrics,
        sample_policy_outputs.final_state,
        policy_samples,
    )

  def _unroll_q_function(
      self,
      q_function: q_lib.QFunction[embed.Action],
      bm_frames: Frames[Rank2, embed.Action],
      initial_states: RecurrentState,
      rngs: nnx.Rngs,
      policy_samples: list[embed.Action],  # frame_skip x [S, T, B]
  ) -> tuple[Loss, dict, RecurrentState, jax.Array, jax.Array, RecurrentState]:
    frames = jax.tree.map(jax_utils.swap_axes, bm_frames)
    frames = self._get_delayed_frames(frames)

    q_outputs, core_outputs, final_state = q_function.loss_and_core_outputs(
        frames, initial_states, rngs, self.config.num_index_samples, self.discount)

    q_bias = jnp.mean(q_outputs.q_values - q_outputs.values, axis=0)  # [T, B]

    assert _SAMPLE_AXIS == 0
    actions = policy_samples
    num_samples = self.num_samples
    if self.include_action_taken_in_samples:
      num_samples += 1
      actions = utils.map_nt(
        lambda samples, action_taken: jnp.concatenate(
          [samples, jnp.expand_dims(action_taken[1:], axis=_SAMPLE_AXIS)], axis=_SAMPLE_AXIS),
        policy_samples, frames.state_action.action)  # frame_skip x [S + 1, T, B]

    # TODO: don't recompute q-values for the action taken?
    indexed_q_values = q_function.multi_index_q_values_from_core_outputs(
        core_outputs=core_outputs,
        actions=actions,
        rngs=rngs,
        num_index_samples=self.config.num_index_samples,
        batch_size=self.config.sample_batch_size,
    )  # [N, S, T, B]
    # The ensemble (mean over indices) is the point estimate of the q-values.
    mean_q_values = jnp.mean(indexed_q_values, axis=0)  # [S, T, B]

    # Just the policy samples, without the action taken.
    mean_sample_q_values = mean_q_values[:self.num_samples]  # [S (- 1), T, B]
    if self.include_action_taken_in_samples:
      mean_taken_q_value = mean_q_values[-1]
    else:
      # TODO: we are using *different* epistemic indices for the taken actions
      # than for the sampled actions. We should instead use the same indices for both.
      mean_taken_q_value = jnp.mean(q_outputs.q_values, axis=0)  # [T, B]

    sample_policy_expected_return = jnp.mean(
        mean_sample_q_values, axis=_SAMPLE_AXIS)
    sample_policy_advantages = sample_policy_expected_return - mean_taken_q_value

    argmaxes = jnp.argmax(indexed_q_values, axis=1)  # [N, T, B]
    one_hots = jax.nn.one_hot(argmaxes, num_samples, axis=1)  # [N, S, T, B]
    distribution = jnp.mean(one_hots, axis=0)  # [S, T, B]

    entropy = jax_utils.entropy(distribution, axis=0)  # [T, B]

    optimal_expected_return = jnp.max(mean_q_values, axis=_SAMPLE_AXIS)
    optimal_advantages = optimal_expected_return - mean_taken_q_value

    action_taken_is_optimal = optimal_expected_return <= mean_taken_q_value

    optimal_sample_policy_advantage = masked_mean(
        sample_policy_advantages,
        mask=~action_taken_is_optimal,
    )

    non_optimal_sample_policy_advantage = masked_mean(
        sample_policy_advantages,
        mask=action_taken_is_optimal,
    )

    bm_loss = jnp.mean(q_outputs.loss, axis=0)
    metrics = dict(
        q_outputs.metrics,
        sample_policy_advantages=sample_policy_advantages,
        optimal_advantages=optimal_advantages,
        action_taken_is_optimal=action_taken_is_optimal,
        optimal_sample_policy_advantage=optimal_sample_policy_advantage,
        non_optimal_sample_policy_advantage=non_optimal_sample_policy_advantage,
        q_bias=q_bias,
        entropy=entropy,
    )

    bm_metrics = jax.tree.map(jax_utils.swap_axes, metrics)

    return (
        bm_loss, bm_metrics, final_state,
        core_outputs, distribution, mean_taken_q_value)

  def _unroll_q_policy(
      self,
      q_policy: Policy[embed.Action],
      rngs: nnx.Rngs,
      q_function: q_lib.QFunction[embed.Action],
      bm_frames: Frames[Rank2, embed.Action],
      initial_states: RecurrentState,
      q_fn_core_outputs: jax.Array,  # [T, B, O_core]
      policy_samples: list[embed.Action],  # frame_skip x [S, T, B]
      distribution: jax.Array,  # [S, T, B]
      mean_taken_q_value: jax.Array,  # [T, B]
  ) -> tuple[Loss, dict, RecurrentState]:
    frames = jax.tree.map(jax_utils.swap_axes, bm_frames)
    frames = self._get_delayed_frames(frames)

    prev_action = jax.tree.map(lambda t: t[:-1], frames.state_action.action)

    # Train the q_policy by argmaxing the q_function over the sample_policy
    q_policy_outputs = q_policy.unroll_with_outputs(
        frames, initial_states)
    q_policy_imitation_loss = q_policy_outputs.imitation_loss

    actions = policy_samples
    num_samples = self.num_samples
    if self.include_action_taken_in_samples:
      num_samples += 1
      actions = utils.map_nt(
        lambda samples, action_taken: jnp.concatenate(
          [samples, jnp.expand_dims(action_taken[1:], axis=_SAMPLE_AXIS)], axis=_SAMPLE_AXIS),
        policy_samples, frames.state_action.action)  # frame_skip x [S + 1, T, B]

    assert distribution.shape[0] == num_samples

    # Distance to the best (highest-q) frame_skip action sequence, averaged
    # over epistemic indices, i.e. cross-entropy to the uniform distribution
    # over the per-index argmax actions.
    @nnx.vmap(in_axes=(None, 0), out_axes=0)
    def distances_to_target(
        q_policy: Policy[embed.Action], target: list[embed.Action],
    ) -> jax.Array:
      fs_distances = q_policy.controller_head.distance(
          inputs=q_policy_outputs.outputs,
          prev_controller_state=prev_action,
          target_controller_state=target,
      )
      return jax_utils.add_n(fs_distances) / len(fs_distances)

    q_policy_distances = distances_to_target(q_policy, actions)  # [S, T, B]
    q_policy_argmax_loss = jnp.sum(q_policy_distances * distribution, axis=0)  # [T, B]

    # Estimate q_policy returns
    q_policy_samples = [
        so.controller_state
        for so in q_policy.controller_head.sample(
            rngs=rngs,
            inputs=q_policy_outputs.outputs,
            prev_controller_state=prev_action)]

    # Note: q_function will already be in the correct dtype.
    q_policy_sample_q_values = q_function.q_values_from_core_outputs(
        core_outputs=q_fn_core_outputs,
        rngs=rngs,
        num_index_samples=self.config.num_index_samples,
        actions=q_policy_samples,
    )  # [N, T, B]
    # TODO: there is some extra noise from using different indices for the q_policy
    # compared to the taken (or sampled) actions.
    mean_q_policy_sample_q_values = jnp.mean(q_policy_sample_q_values, axis=0)
    q_policy_advantages = mean_q_policy_sample_q_values - mean_taken_q_value

    losses = [
        self.q_policy_argmax_weight * q_policy_argmax_loss,
        self.q_policy_imitation_weight * q_policy_imitation_loss,
    ]
    q_policy_total_loss = jax_utils.add_n(losses)

    q_policy_metrics = dict(
        q_loss=q_policy_argmax_loss,
        imitation_loss=q_policy_imitation_loss,
        total_loss=q_policy_total_loss,
        q_policy_advantages=q_policy_advantages,
    )

    bm_loss = jnp.mean(q_policy_total_loss, axis=0)
    bm_metrics = jax.tree.map(jax_utils.swap_axes, q_policy_metrics)

    return bm_loss, bm_metrics, q_policy_outputs.final_state

  def step(
      self,
      frames: dict[str, Frames[Rank2, embed.Action]],
      initial_states: RecurrentState,
      train: bool = True,
  ) -> tuple[dict, RecurrentState]:
    final_states = initial_states  # GC initial states as they are replaced
    metrics = {}

    (
      metrics[SAMPLE_POLICY],
      final_states[SAMPLE_POLICY],
      policy_samples,
    ) = self.run_sample_policy(
        frames[SAMPLE_POLICY], initial_states[SAMPLE_POLICY])

    (
      metrics[Q_FUNCTION],
      final_states[Q_FUNCTION],
      core_outputs,
      distribution,
      mean_taken_q_value,
    ) = self.run_q_function(
        frames[Q_FUNCTION], initial_states[Q_FUNCTION], policy_samples)

    step_q_policy = self.train_q_policy if train else self.run_q_policy
    (
      metrics[Q_POLICY],
      final_states[Q_POLICY],
    ) = step_q_policy(
        frames[Q_POLICY], initial_states[Q_POLICY],
        core_outputs, policy_samples, distribution, mean_taken_q_value)

    return metrics, final_states
