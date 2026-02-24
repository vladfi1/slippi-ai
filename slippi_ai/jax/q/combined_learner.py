import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from slippi_ai import utils
from slippi_ai.data import Batch, Frames, StateAction
from slippi_ai.jax.policies import Policy, RecurrentState
from slippi_ai.jax.q import q_function as q_lib
from slippi_ai.jax import embed, rl_lib, jax_utils
from slippi_ai.jax.jax_utils import PS, DATA_AXIS

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  reward_halflife: float = 4

  train_sample_policy: bool = False

  num_samples: int = 1
  sample_batch_size: int = 0  # 0 means full batch size, i.e. vmap
  include_action_taken_in_samples: bool = True

  q_policy_argmax_weight: float = 1
  q_policy_imitation_weight: float = 0
  q_policy_expected_return_weight: float = 0

_SAMPLE_AXIS = 0

def replicate(n: int, axis: int = _SAMPLE_AXIS) -> tp.Callable[[jax.Array], jax.Array]:
  def fn(x: jax.Array) -> jax.Array:
    x0 = jnp.expand_dims(x, axis)
    reps = [1] * x0.ndim
    reps[axis] = n
    return jnp.tile(x0, reps)
  return fn

T = tp.TypeVar('T')

def identity(x: T) -> T:
  return x

class QFunctionOutputs(tp.NamedTuple):
  values: jax.Array  # [T, B]
  sample_q_values: jax.Array  # [num_samples, T, B]

Loss = jax.Array
Rank2 = tuple[int, int]


class ShardingKwargs(tp.TypedDict):
  mesh: jax.sharding.Mesh
  explicit_pmean: bool
  smap_optimizer: bool

class ShardingSpecs(tp.TypedDict):
  extra_in_specs: tp.Optional[tp.Sequence[PS]]
  extra_out_specs: tp.Optional[tp.Sequence[PS]]

SAMPLE_POLICY = 'sample_policy'
Q_FUNCTION = 'q_function'
Q_POLICY = 'q_policy'

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
  ):
    self.config = config
    self.q_function = q_function
    self.sample_policy = sample_policy
    self.q_policy = q_policy

    learning_rate = config.learning_rate
    self.q_function_optimizer = nnx.Optimizer(
        q_function, optax.adam(learning_rate), wrt=nnx.Param)

    self.sample_policy_optimizer = nnx.Optimizer(
        sample_policy, optax.adam(learning_rate), wrt=nnx.Param)
    self.q_policy_optimizer = nnx.Optimizer(
        q_policy, optax.adam(learning_rate), wrt=nnx.Param)

    self.discount = rl_lib.discount_from_halflife(config.reward_halflife)

    if not config.include_action_taken_in_samples and config.num_samples < 2:
      raise ValueError('num_samples must be at least 2 if not including action taken in samples')

    self.num_samples = config.num_samples
    self.include_action_taken_in_samples = config.include_action_taken_in_samples
    self.q_policy_argmax_weight = config.q_policy_argmax_weight
    self.q_policy_imitation_weight = config.q_policy_imitation_weight
    self.q_policy_expected_return_weight = config.q_policy_expected_return_weight

    self.delay = q_policy.delay
    assert sample_policy.delay == self.delay

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

    self.train_sample_policy = jax_utils.data_parallel_train_with_rngs(
        module=self.sample_policy,
        optimizer=self.sample_policy_optimizer,
        rngs=rngs,
        loss_fn=self._unroll_sample_policy,
        **sharding_kwargs,
        **sample_policy_specs,
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
        # best_action, values, q_fn hidden states
        extra_out_specs=(TM, TM, TM),
    )

    self.train_q_function = jax_utils.data_parallel_train(
        module=self.q_function,
        optimizer=self.q_function_optimizer,
        loss_fn=self._unroll_q_function,
        **sharding_kwargs,
        **q_function_specs,
    )

    self.run_q_function = jax_utils.shard_map_loss_fn(
        module=self.q_function,
        loss_fn=self._unroll_q_function,
        mesh=mesh,
        **q_function_specs,
    )

    q_policy_specs = ShardingSpecs(
        # best_action, values, q_fn hidden states
        extra_in_specs=(TM, TM, TM),
        extra_out_specs=None,
    )

    self.train_q_policy = jax_utils.data_parallel_train_with_rngs(
        module=self.q_policy,
        optimizer=self.q_policy_optimizer,
        rngs=rngs,
        loss_fn=self._unroll_q_policy,
        **sharding_kwargs,
        **q_policy_specs,
    )

    self.run_q_policy = jax_utils.shard_map_loss_fn_with_rngs(
        module=self.q_policy,
        rngs=rngs,
        loss_fn=self._unroll_q_policy,
        mesh=mesh,
        **q_policy_specs,
    )

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return {
        Q_FUNCTION: self.q_function.initial_state(batch_size, rngs),
        Q_POLICY: self.q_policy.initial_state(batch_size, rngs),
        SAMPLE_POLICY: self.sample_policy.initial_state(batch_size, rngs),
    }

  def _get_delayed_frames(self, frames: Frames[Rank2, embed.Action]) -> Frames[Rank2, embed.Action]:
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

  def _shard_frames(self, frames: Frames[Rank2, embed.Action]) -> Frames[Rank2, embed.Action]:
    return utils.map_single_structure(lambda x: jax.device_put(x, self.data_sharding), frames)

  def prepare_frames(self, batch: Batch[Rank2]) -> Frames[Rank2, embed.Action]:
    # Note: this assumes that the sample_policy, q_function, and q_policy all
    # use the same embedding for the state_action.
    state_action = StateAction(
        batch.game, batch.game.p0.controller, batch.name)
    frames = Frames(
        state_action=self.q_function.core_net.encode(state_action),
        is_resetting=batch.is_resetting,
        reward=batch.reward,
    )
    return self._shard_frames(frames)

  def _unroll_sample_policy(
      self,
      sample_policy: Policy[embed.Action],
      bm_frames: Frames[Rank2, embed.Action],
      initial_states: RecurrentState,
      rngs: nnx.Rngs,
  ) -> tuple[Loss, dict, RecurrentState, embed.Action]:
    frames = jax.tree.map(jax_utils.swap_axes, bm_frames)
    frames = self._get_delayed_frames(frames)

    action = frames.state_action.action
    prev_action = jax.tree.map(lambda t: t[:-1], action)
    next_action = jax.tree.map(lambda t: t[1:], action)

    sample_policy_outputs = sample_policy.unroll_with_outputs(frames, initial_states)

    # Because the action space is too large, we compute a finite subsample
    # using the sample_policy.

    @nnx.vmap(in_axes=0, out_axes=_SAMPLE_AXIS)
    def sample(rngs: nnx.Rngs):
      return sample_policy.controller_head.sample(
          rngs=rngs,
          inputs=sample_policy_outputs.outputs,
          prev_controller_state=prev_action).controller_state

    policy_samples = sample(rngs.fork(split=self.num_samples))

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
      policy_samples: embed.Action,
  ) -> tuple[Loss, dict, RecurrentState, embed.Action, jax.Array, RecurrentState]:
    frames = jax.tree.map(jax_utils.swap_axes, bm_frames)
    frames = self._get_delayed_frames(frames)

    q_outputs, final_state = q_function.loss(frames, initial_states, self.discount)

    q_bias = q_outputs.q_values - q_outputs.values

    def q_fn(policy_samples: embed.Action) -> jax.Array:
      return self.q_function.q_values_from_hidden_states(
          values=q_outputs.values,
          hidden_states=q_outputs.hidden_states,
          actions=policy_samples,
      )

    assert _SAMPLE_AXIS == 0
    sample_q_values = jax.lax.map(
        q_fn, policy_samples,
        batch_size=self.config.sample_batch_size,
    )

    sample_policy_expected_return = jnp.mean(
        sample_q_values, axis=_SAMPLE_AXIS)
    sample_policy_advantages = sample_policy_expected_return - q_outputs.values

    actions = policy_samples
    q_values = sample_q_values

    if self.include_action_taken_in_samples:
      actions = jax.tree.map(
        lambda samples, action_taken: jnp.concatenate(
          [samples, jnp.expand_dims(action_taken[1:], axis=_SAMPLE_AXIS)], axis=_SAMPLE_AXIS),
        policy_samples, frames.state_action.action)
      q_values = jnp.concatenate(
        [sample_q_values, jnp.expand_dims(q_outputs.q_values, axis=_SAMPLE_AXIS)], axis=_SAMPLE_AXIS)

    best_action_index = jnp.argmax(q_values, axis=_SAMPLE_AXIS, keepdims=True)
    best_action = jax.tree.map(
        lambda x: jnp.squeeze(
            jnp.take_along_axis(x, best_action_index, axis=_SAMPLE_AXIS),
            axis=_SAMPLE_AXIS),
        actions)

    optimal_expected_return = jnp.max(q_values, axis=_SAMPLE_AXIS)
    optimal_advantages = optimal_expected_return - q_outputs.values

    action_taken_is_optimal = optimal_expected_return <= q_outputs.q_values

    bm_loss = jnp.mean(q_outputs.loss, axis=0)
    metrics = dict(
        q_outputs.metrics,
        sample_policy_advantages=sample_policy_advantages,
        optimal_advantages=optimal_advantages,
        action_taken_is_optimal=action_taken_is_optimal,
        q_bias=q_bias,
    )

    bm_metrics = jax.tree.map(jax_utils.swap_axes, metrics)

    return bm_loss, bm_metrics, final_state, best_action, q_outputs.values, q_outputs.hidden_states

  def _unroll_q_policy(
      self,
      q_policy: Policy[embed.Action],
      bm_frames: Frames[Rank2, embed.Action],
      initial_states: RecurrentState,
      rngs: nnx.Rngs,
      best_action: embed.Action,
      values: jax.Array,
      q_hidden_states: RecurrentState,
  ) -> tuple[Loss, dict, RecurrentState]:
    frames = jax.tree.map(jax_utils.swap_axes, bm_frames)
    frames = self._get_delayed_frames(frames)

    action = frames.state_action.action
    prev_action = jax.tree.map(lambda t: t[:-1], action)

    num_samples = self.num_samples
    if self.include_action_taken_in_samples:
      num_samples += 1

    # Train the q_policy by argmaxing the q_function over the sample_policy
    q_policy_outputs = q_policy.unroll_with_outputs(
        frames, initial_states)
    q_policy_imitation_loss = q_policy_outputs.imitation_loss

    q_policy_distance = q_policy.controller_head.distance(
          inputs=q_policy_outputs.outputs,
          prev_controller_state=prev_action,
          target_controller_state=best_action,
      ).distance

    q_policy_argmax_loss = jax_utils.add_n(
        q_policy.controller_head.controller_embedding.flatten(
            q_policy_distance))

    # Estimate q_policy returns
    @nnx.vmap(in_axes=0, out_axes=_SAMPLE_AXIS)
    def sample_q_policy(rngs: nnx.Rngs):
      return q_policy.controller_head.sample(
          rngs=rngs,
          inputs=q_policy_outputs.outputs,
          prev_controller_state=prev_action).controller_state

    q_policy_samples = sample_q_policy(rngs.fork(split=num_samples))

    def compute_q(policy_sample: embed.Action) -> jax.Array:
      return self.q_function.q_values_from_hidden_states(
          values=values,
          hidden_states=q_hidden_states,
          actions=policy_sample,
      )

    assert _SAMPLE_AXIS == 0
    q_policy_sample_q_values = jax.lax.map(
        compute_q, q_policy_samples,
        batch_size=self.config.sample_batch_size,
    )
    q_policy_expected_return = jnp.mean(q_policy_sample_q_values, axis=_SAMPLE_AXIS)
    q_policy_advantages = q_policy_expected_return - values

    losses = [
        self.q_policy_argmax_weight * q_policy_argmax_loss,
        self.q_policy_imitation_weight * q_policy_imitation_loss,
    ]
    q_policy_total_loss = jax_utils.add_n(losses)

    q_policy_metrics = dict(
        q_loss=q_policy_argmax_loss,
        imitation_loss=q_policy_imitation_loss,
        q_policy_advantages=q_policy_advantages,
    )

    bm_loss = jnp.mean(q_policy_total_loss, axis=0)
    bm_metrics = jax.tree.map(jax_utils.swap_axes, q_policy_metrics)

    return bm_loss, bm_metrics, q_policy_outputs.final_state

  def step_sample_policy(
      self,
      batch: Batch,
      initial_state: RecurrentState,
      train: bool = True,
  ):
    state_action = StateAction(
        batch.game, batch.game.p0.controller, batch.name)
    frames = Frames(
        state_action=self.sample_policy.network.encode(state_action),
        is_resetting=batch.is_resetting,
        reward=batch.reward,
    )
    frames = self._shard_frames(frames)

    # TODO: properly fork rngs for shard_map
    if train:
      return self.train_sample_policy(frames, initial_state)
    else:
      return self.run_sample_policy(frames, initial_state)

  def step_q_function(
      self,
      batch: Batch[Rank2],  # batch-major
      initial_state: RecurrentState,
      policy_samples: embed.Action,  # time-major
      train: bool = True,
  ):
    state_action = StateAction(
        batch.game, batch.game.p0.controller, batch.name)
    frames = Frames(
        state_action=self.q_function.core_net.encode(state_action),
        is_resetting=batch.is_resetting,
        reward=batch.reward,
    )
    frames = self._shard_frames(frames)

    if train:
      return self.train_q_function(frames, initial_state, policy_samples)
    else:
      return self.run_q_function(frames, initial_state, policy_samples)

  def step_q_policy(
      self,
      batch: Batch[Rank2], # batch-major
      initial_state: RecurrentState,
      best_action: embed.Action,
      values: jax.Array,
      q_hidden_states: RecurrentState,
      train: bool = True,
  ):
    state_action = StateAction(
        batch.game, batch.game.p0.controller, batch.name)
    frames = Frames(
        state_action=self.q_policy.network.encode(state_action),
        is_resetting=batch.is_resetting,
        reward=batch.reward,
    )
    frames = self._shard_frames(frames)

    if train:
      return self.train_q_policy(
          frames, initial_state,
          best_action, values, q_hidden_states)
    else:
      return self.run_q_policy(
          frames, initial_state,
          best_action, values, q_hidden_states)

  def step(
      self,
      batch: Batch,
      initial_states: RecurrentState,
      train: bool = True,
  ) -> tuple[dict, RecurrentState]:
    # TODO: take into account delay
    final_states = initial_states  # GC initial states as they are replaced
    metrics = {}

    (
      metrics[SAMPLE_POLICY],
      final_states[SAMPLE_POLICY],
      policy_samples,
    ) = self.step_sample_policy(
        batch, initial_states[SAMPLE_POLICY],
        train=train and self.config.train_sample_policy)

    (
      metrics[Q_FUNCTION],
      final_states[Q_FUNCTION],
      best_action,
      values,
      q_hidden_states,
    ) = self.step_q_function(
        batch, initial_states[Q_FUNCTION], policy_samples,
        train=train)
    del policy_samples

    (
      metrics[Q_POLICY],
      final_states[Q_POLICY],
    ) = self.step_q_policy(
        batch, initial_states[Q_POLICY],
        best_action, values, q_hidden_states,
        train=train)

    # satisfy train_q_lib._get_loss
    metrics['total_loss'] = metrics[Q_POLICY]['q_loss']

    return metrics, final_states
