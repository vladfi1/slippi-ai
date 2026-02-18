import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from slippi_ai import utils
from slippi_ai.data import Batch, Frames, StateAction
from slippi_ai.jax.policies import Policy, RecurrentState
from slippi_ai.jax import q_function as q_lib
from slippi_ai.jax import embed, rl_lib, jax_utils
from slippi_ai.jax.jax_utils import PS, DATA_AXIS

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  reward_halflife: float = 4

  train_sample_policy: bool = False

  num_samples: int = 1
  q_policy_imitation_weight: float = 0
  q_policy_expected_return_weight: float = 0

_SAMPLE_AXIS = 2

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
      learning_rate: float,
      reward_halflife: float,
      q_function: q_lib.QFunction[embed.Action],
      sample_policy: Policy[embed.Action],  # trained via imitation
      q_policy: Policy[embed.Action],  # trained to maximize q_function outputs
      rngs: nnx.Rngs,  # used for sampling
      num_samples: int,
      mesh: jax.sharding.Mesh,
      data_sharding: jax.sharding.NamedSharding,
      explicit_pmean: bool = False,
      smap_optimizer: bool = True,
      train_sample_policy: bool = True,
      q_policy_imitation_weight: float = 0,
      q_policy_expected_return_weight: float = 0,
  ):
    self.q_function = q_function
    self.sample_policy = sample_policy
    self.q_policy = q_policy

    self.q_function_optimizer = nnx.Optimizer(
        q_function, optax.adam(learning_rate), wrt=nnx.Param)

    self.sample_policy_optimizer = nnx.Optimizer(
        sample_policy, optax.adam(learning_rate), wrt=nnx.Param)
    self.q_policy_optimizer = nnx.Optimizer(
        q_policy, optax.adam(learning_rate), wrt=nnx.Param)

    self.discount = rl_lib.discount_from_halflife(reward_halflife)

    self.should_train_sample_policy = train_sample_policy

    self.num_samples = num_samples
    self.q_policy_imitation_weight = q_policy_imitation_weight
    self.q_policy_expected_return_weight = q_policy_expected_return_weight

    self.delay = q_policy.delay
    assert sample_policy.delay == self.delay

    jax_utils.replicate_module(self, mesh)

    self.data_sharding = data_sharding
    sharding_kwargs = ShardingKwargs(
        mesh=mesh,
        explicit_pmean=explicit_pmean,
        smap_optimizer=smap_optimizer,
    )

    time_major = PS(None, DATA_AXIS)

    sample_policy_specs = ShardingSpecs(
        extra_in_specs=None,
        extra_out_specs=(time_major,),  # policy samples
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
        extra_in_specs=(time_major,),  # policy samples
        extra_out_specs=(time_major,),  # q_function outputs
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
        extra_in_specs=(time_major, time_major),  # policy samples, q_function outputs
        extra_out_specs=None,
    )

    self.train_q_policy = jax_utils.data_parallel_train(
        module=self.q_policy,
        optimizer=self.q_policy_optimizer,
        loss_fn=self._unroll_q_policy,
        **sharding_kwargs,
        **q_policy_specs,
    )

    self.run_q_policy = jax_utils.shard_map_loss_fn(
        module=self.q_policy,
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

    replicate_samples = replicate(self.num_samples, axis=_SAMPLE_AXIS)

    # Because the action space is too large, we compute a finite subsample
    # using the sample_policy.
    replicated_sample_policy_outputs = replicate_samples(sample_policy_outputs.outputs)
    replicated_prev_action = jax.tree.map(replicate_samples, prev_action)
    policy_samples = self.sample_policy.controller_head.sample(
        rngs=rngs,
        inputs=replicated_sample_policy_outputs,
        prev_controller_state=replicated_prev_action)

    # Include the actual action taken among the samples.
    policy_samples = jax.tree.map(
        lambda samples, na: jnp.concatenate([
            samples, jnp.expand_dims(na, _SAMPLE_AXIS)], _SAMPLE_AXIS),
        policy_samples.controller_state, next_action,
    )

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
      *,
      optimize: bool = True,  # Don't recompute q-values of taken actions
  ) -> tuple[Loss, dict, RecurrentState, QFunctionOutputs]:
    frames = jax.tree.map(jax_utils.swap_axes, bm_frames)
    frames = self._get_delayed_frames(frames)

    q_outputs, final_state = q_function.loss(frames, initial_states, self.discount)

    if optimize:
      num_samples = self.num_samples
      # Slice out last action since that was the one taken and its q-value is
      # already computed in q_outputs.q_values.
      policy_samples = jax.tree.map(
          lambda x: jax.lax.slice_in_dim(x, 0, -1, axis=_SAMPLE_AXIS),
          policy_samples)
    else:
      num_samples = self.num_samples + 1

    replicate_samples = lambda nest: jax.tree.map(
        replicate(num_samples, axis=_SAMPLE_AXIS), nest)

    # Compute the q-values of the sampled actions
    replicated_hidden_states = replicate_samples(q_outputs.hidden_states)
    sample_q_values = self.q_function.q_values_from_hidden_states(
        hidden_states=replicated_hidden_states,
        actions=policy_samples,
    )

    if optimize:
      sample_q_values = jnp.concatenate(
          [sample_q_values, jnp.expand_dims(q_outputs.q_values, _SAMPLE_AXIS)], axis=_SAMPLE_AXIS)

    outputs = QFunctionOutputs(
        values=q_outputs.values,
        sample_q_values=sample_q_values,
    )

    bm_loss = jnp.mean(q_outputs.loss, axis=0)
    bm_metrics = jax.tree.map(jax_utils.swap_axes, q_outputs.metrics)

    return bm_loss, bm_metrics, final_state, outputs

  def _unroll_q_policy(
      self,
      q_policy: Policy[embed.Action],
      bm_frames: Frames[Rank2, embed.Action],
      initial_states: RecurrentState,
      policy_samples: embed.Action,
      q_function_outputs: QFunctionOutputs,
  ) -> tuple[Loss, dict, RecurrentState]:
    frames = jax.tree.map(jax_utils.swap_axes, bm_frames)
    frames = self._get_delayed_frames(frames)

    sample_q_values = q_function_outputs.sample_q_values
    action = frames.state_action.action
    prev_action = jax.tree.map(lambda t: t[:-1], action)

    num_samples = self.num_samples + 1
    assert sample_q_values.shape[_SAMPLE_AXIS] == num_samples
    replicate_samples = lambda nest: jax.tree.map(
        replicate(num_samples, axis=_SAMPLE_AXIS), nest)

    # Train the q_policy by argmaxing the q_function over the sample_policy
    q_policy_outputs = q_policy.unroll_with_outputs(
        frames, initial_states)

    # Construct a target distribution over the subsample and regress the
    # q_policy to this target.
    replicated_q_policy_outputs = replicate_samples(q_policy_outputs.outputs)
    q_policy_distances = q_policy.controller_head.distance(
        inputs=replicated_q_policy_outputs,
        prev_controller_state=replicate_samples(prev_action),
        target_controller_state=policy_samples,
    )
    q_policy_log_probs = -jax_utils.add_n(
        q_policy.controller_head.controller_embedding.flatten(
            q_policy_distances.distance))
    q_policy_imitation_loss = -jax.lax.index_in_dim(
        q_policy_log_probs, -1, axis=_SAMPLE_AXIS, keepdims=False)

    # Normalize log-probs for the finite sample
    q_policy_log_probs -= jax.nn.logsumexp(
        q_policy_log_probs, axis=_SAMPLE_AXIS, keepdims=True)

    best_action = jnp.argmax(sample_q_values, axis=_SAMPLE_AXIS)
    target_distribution = jax.nn.one_hot(best_action, num_samples, axis=_SAMPLE_AXIS)
    q_policy_argmax_loss = -jnp.sum(
        target_distribution * q_policy_log_probs, axis=_SAMPLE_AXIS)

    q_policy_probs = jnp.exp(q_policy_log_probs)
    q_policy_expected_return = jnp.sum(
        q_policy_probs * sample_q_values, axis=_SAMPLE_AXIS)

    # We could also use the returns (value_targets) from the q_function, but
    # it's a bit weird because they are correlated with the action taken.
    q_policy_advantages = q_policy_expected_return - q_function_outputs.values
    optimal_expected_return = jnp.max(sample_q_values, axis=_SAMPLE_AXIS)
    optimal_advantages = optimal_expected_return - q_function_outputs.values
    regret = q_policy_expected_return - optimal_expected_return

    losses = [
        q_policy_argmax_loss,
        self.q_policy_imitation_weight * q_policy_imitation_loss,
        -self.q_policy_expected_return_weight * q_policy_expected_return,
    ]
    q_policy_total_loss = jax_utils.add_n(losses)

    action_taken_is_optimal = jnp.equal(best_action, num_samples - 1)

    q_policy_metrics = dict(
        q_loss=q_policy_argmax_loss,
        imitation_loss=q_policy_imitation_loss,
        expected_return=q_policy_expected_return,
        advantages=q_policy_advantages,
        optimal_advantages=optimal_advantages,
        regret=regret,
        action_taken_is_optimal=action_taken_is_optimal,
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
      policy_samples: embed.Action,  # time-major
      q_function_outputs: QFunctionOutputs,  # time-major
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
          policy_samples, q_function_outputs)
    else:
      return self.run_q_policy(
          frames, initial_state,
          policy_samples, q_function_outputs)

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
        train=train and self.should_train_sample_policy)

    (
      metrics[Q_FUNCTION],
      final_states[Q_FUNCTION],
      q_function_outputs,
    ) = self.step_q_function(
        batch, initial_states[Q_FUNCTION], policy_samples,
        train=train)

    (
      metrics[Q_POLICY],
      final_states[Q_POLICY],
    ) = self.step_q_policy(
        batch, initial_states[Q_POLICY],
        policy_samples, q_function_outputs, train)

    # satisfy train_q_lib._get_loss
    metrics['total_loss'] = metrics[Q_POLICY]['q_loss']

    return metrics, final_states
