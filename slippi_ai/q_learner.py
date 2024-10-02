import dataclasses
import typing as tp
from typing import Optional

import numpy as np
import sonnet as snt
import tensorflow as tf

from slippi_ai.data import Batch, Frames
from slippi_ai.policies import Policy, RecurrentState
from slippi_ai import q_function as q_lib
from slippi_ai import tf_utils, embed

def swap_axes(t, axis1=0, axis2=1):
  permutation = list(range(len(t.shape)))
  permutation[axis2] = axis1
  permutation[axis1] = axis2
  return tf.transpose(t, permutation)

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  compile: bool = True
  jit_compile: bool = True
  reward_halflife: float = 8

  num_samples: int = 1
  q_policy_imitation_weight: float = 0
  q_policy_expected_return_weight: float = 0

# TODO: use tf.tile instead?
replicate = lambda n: lambda t: tf.stack([t] * n)

T = tp.TypeVar('T')

def identity(x: T) -> T:
  return x

class QFunctionOutputs(tp.NamedTuple):
  values: tf.Tensor  # [T, B]
  sample_q_values: tf.Tensor  # [num_samples, T, B]

# TODO: should this be a snt.Module?
class Learner:

  def __init__(self,
      learning_rate: float,
      compile: bool,
      reward_halflife: float,
      q_function: q_lib.QFunction,
      sample_policy: Policy,  # trained via imitation
      q_policy: Policy,
      num_samples: int,
      q_policy_imitation_weight: float = 0,
      q_policy_expected_return_weight: float = 0,
      jit_compile: Optional[bool] = None,
  ):
    self.q_function = q_function
    self.sample_policy = sample_policy
    self.q_policy = q_policy
    self.q_function_optimizer = snt.optimizers.Adam(learning_rate)
    self.sample_policy_optimizer = snt.optimizers.Adam(learning_rate)
    self.q_policy_optimizer = snt.optimizers.Adam(learning_rate)
    self.discount = 0.5 ** (1 / (reward_halflife * 60))

    if compile:
      maybe_compile = tf.function(jit_compile=jit_compile, autograph=False)
    else:
      maybe_compile = identity

    self.train_sample_policy = maybe_compile(self._train_sample_policy)
    self.train_q_function = maybe_compile(self._train_q_function)
    self.train_q_policy = maybe_compile(self._train_q_policy)
    self.compiled_step = self.step  # This isn't entirely accurate.

    self.num_samples = num_samples
    self.q_policy_imitation_weight = q_policy_imitation_weight
    self.q_policy_expected_return_weight = q_policy_expected_return_weight

    assert q_policy.delay == 0
    assert sample_policy.delay == 0

  def initial_state(self, batch_size: int) -> RecurrentState:
    return dict(
        q_function=self.q_function.initial_state(batch_size),
        q_policy=self.q_policy.initial_state(batch_size),
        sample_policy=self.sample_policy.initial_state(batch_size),
    )

  def initialize_variables(self):
    self.sample_policy.initialize_variables()
    self.sample_policy_optimizer._initialize(self.sample_policy.trainable_variables)

    self.q_policy.initialize_variables()
    self.q_policy_optimizer._initialize(self.q_policy.trainable_variables)

    self.q_function.initialize_variables()
    self.q_function_optimizer._initialize(self.q_function.trainable_variables)

  def _train_sample_policy(
      self,
      frames: Frames,
      # batch: Batch,
      initial_states: RecurrentState,
      train: bool = True,
  ) -> tuple[embed.Action, RecurrentState, dict]:
    # frames = batch.frames
    # restarting = batch.needs_reset
    # batch_size = restarting.shape[0]

    # # reset initial_states where necessary
    # restarting = tf.expand_dims(restarting, -1)
    # initial_states = tf.nest.map_structure(
    #     lambda x, y: tf.where(restarting, x, y),
    #     self.sample_policy.initial_state(batch_size),
    #     initial_states)

    action = frames.state_action.action
    prev_action = tf.nest.map_structure(lambda t: t[:-1], action)
    next_action = tf.nest.map_structure(lambda t: t[1:], action)

    # Train sample policy with imitation
    with tf.GradientTape() as tape:
      sample_policy_outputs = self.sample_policy.unroll_with_outputs(
          frames, initial_states)
      sample_policy_loss = sample_policy_outputs.imitation_loss

    replicate_samples = replicate(self.num_samples)

    # Because the action space is too large, we compute a finite subsample
    # using the sample_policy.
    replicated_sample_policy_outputs = replicate_samples(sample_policy_outputs.outputs)
    replicated_prev_action = tf.nest.map_structure(replicate_samples, prev_action)
    policy_samples = self.sample_policy.controller_head.sample(
        replicated_sample_policy_outputs, replicated_prev_action)

    # Include the actual action taken among the samples.
    policy_samples = tf.nest.map_structure(
        lambda samples, na: tf.concat([samples, tf.expand_dims(na, 0)], 0),
        policy_samples.controller_state, next_action,
    )

    # Defer updating the policy until after we sample from it.
    if train:
      sample_policy_vars = self.sample_policy.trainable_variables
      tf_utils.assert_same_variables(tape.watched_variables(), sample_policy_vars)
      sample_policy_grads = tape.gradient(sample_policy_loss, sample_policy_vars)
      self.sample_policy_optimizer.apply(sample_policy_grads, sample_policy_vars)

    return (
        policy_samples,
        sample_policy_outputs.final_state,
        sample_policy_outputs.metrics
    )

  def _train_q_function(
      self,
      # batch: Batch,
      frames: Frames,
      initial_states: RecurrentState,
      policy_samples: embed.Action,
      train: bool = True,
  ) -> tuple[QFunctionOutputs, RecurrentState, dict]:
    # frames = batch.frames
    # restarting = batch.needs_reset
    # batch_size = restarting.shape[0]

    # # reset initial_states where necessary
    # restarting = tf.expand_dims(restarting, -1)
    # initial_states = tf.nest.map_structure(
    #     lambda x, y: tf.where(restarting, x, y),
    #     self.q_function.initial_state(batch_size),
    #     initial_states)

    # Train q function with regression
    with tf.GradientTape() as tape:
      # TODO: take into account delay
      q_outputs, q_final_states = self.q_function.loss(
          frames, initial_states, self.discount)

    optimize = True  # Don't recompute q-values of taken actions

    if optimize:
      num_samples = self.num_samples
      policy_samples = tf.nest.map_structure(
          lambda t: t[:-1], policy_samples,
      )
    else:
      num_samples = self.num_samples + 1

    replicate_samples = lambda nest: tf.nest.map_structure(
        replicate(num_samples), nest)

    # Compute the q-values of the sampled actions
    replicated_hidden_states = replicate_samples(q_outputs.hidden_states)
    sample_q_values = snt.BatchApply(
        self.q_function.q_values_from_hidden_states, num_dims=3)(
            hidden_states=replicated_hidden_states,
            actions=policy_samples,
        )

    if optimize:
      sample_q_values = tf.concat(
          [sample_q_values, tf.expand_dims(q_outputs.q_values, 0)], axis=0)

    # Defer updating q_function until after we apply it to the policy_samples
    if train:
      q_function_vars = self.q_function.trainable_variables
      tf_utils.assert_same_variables(tape.watched_variables(), q_function_vars)
      q_function_grads = tape.gradient(q_outputs.loss, q_function_vars)
      self.q_function_optimizer.apply(q_function_grads, q_function_vars)

    outputs = QFunctionOutputs(
        values=q_outputs.values,
        sample_q_values=sample_q_values,
    )

    return outputs, q_final_states, q_outputs.metrics

  def _train_q_policy(
      self,
      frames: Frames,
      # batch: Batch,
      initial_states: RecurrentState,
      policy_samples: embed.Action,
      q_function_outputs: QFunctionOutputs,
      train: bool = True,
  ) -> tuple[RecurrentState, dict]:
    # frames = batch.frames
    # restarting = batch.needs_reset
    # batch_size = restarting.shape[0]

    # # reset initial_states where necessary
    # restarting = tf.expand_dims(restarting, -1)
    # initial_states = tf.nest.map_structure(
    #     lambda x, y: tf.where(restarting, x, y),
    #     self.q_policy.initial_state(batch_size),
    #     initial_states)

    sample_q_values = q_function_outputs.sample_q_values
    action = frames.state_action.action
    prev_action = tf.nest.map_structure(lambda t: t[:-1], action)

    num_samples = self.num_samples + 1
    assert sample_q_values.shape.as_list()[0] == num_samples
    replicate_samples = lambda nest: tf.nest.map_structure(
        replicate(num_samples), nest)

    # Train the q_policy by argmaxing the q_function over the sample_policy
    with tf.GradientTape() as tape:
      q_policy_outputs = self.q_policy.unroll_with_outputs(
          frames, initial_states)

      # Construct a target distribution over the subsample and regress the
      # q_policy to this target.
      replicated_q_policy_outputs = replicate_samples(q_policy_outputs.outputs)
      q_policy_distances = self.q_policy.controller_head.distance(
          inputs=replicated_q_policy_outputs,
          prev_controller_state=replicate_samples(prev_action),
          target_controller_state=policy_samples,
      )
      q_policy_log_probs = -tf.add_n(list(
          self.q_policy.controller_embedding.flatten(
              q_policy_distances.distance)))
      q_policy_imitation_loss = -q_policy_log_probs[-1]

      # Normalize log-probs for the finite sample
      q_policy_log_probs -= tf.math.reduce_logsumexp(
          q_policy_log_probs, axis=0, keepdims=True)

      target_distribution = tf.one_hot(
          tf.argmax(sample_q_values, axis=0), num_samples, axis=0)
      q_policy_q_loss = -tf.reduce_sum(
          target_distribution * q_policy_log_probs, axis=0)

      q_policy_probs = tf.exp(q_policy_log_probs)
      q_policy_expected_return = tf.reduce_sum(
          q_policy_probs * sample_q_values, axis=0)

      # We could also use the returns (value_targets) from the q_function, but
      # it's a bit weird because they are correlated with the action taken.
      q_policy_advantages = q_policy_expected_return - q_function_outputs.values
      optimal_expected_return = tf.reduce_max(sample_q_values, axis=0)
      optimal_advantages = optimal_expected_return - q_function_outputs.values
      regret = q_policy_expected_return - optimal_expected_return

      losses = [
          q_policy_q_loss,
          self.q_policy_imitation_weight * q_policy_imitation_loss,
          self.q_policy_expected_return_weight * q_policy_expected_return,
      ]
      q_policy_total_loss = tf.add_n(losses)

    q_policy_metrics = dict(
        q_loss=q_policy_q_loss,
        imitation_loss=q_policy_imitation_loss,
        expected_return=q_policy_expected_return,
        advantages=q_policy_advantages,
        optimal_advantages=optimal_advantages,
        regret=regret,
    )

    if train:
      q_policy_vars = self.q_policy.trainable_variables
      q_policy_grads = tape.gradient(q_policy_total_loss, q_policy_vars)
      self.q_policy_optimizer.apply(q_policy_grads, q_policy_vars)

    return q_policy_outputs.final_state, q_policy_metrics

  # @tf.function(jit_compile=True)
  def reset_initial_states(
      self,
      initial_states: RecurrentState,
      restarting: np.ndarray,
  ) -> RecurrentState:
    batch_size = restarting.shape[0]
    restarting = tf.expand_dims(restarting, -1)
    return tf.nest.map_structure(
        lambda x, y: tf.where(restarting, x, y),
        self.initial_state(batch_size),
        initial_states)

  def step(
      self,
      batch: Batch,
      initial_states: RecurrentState,
      train: bool = True,
  ) -> tuple[dict, RecurrentState]:
    # switch axes to time-major
    tm_frames: Frames = tf.nest.map_structure(
        lambda a: np.swapaxes(a, 0, 1), batch.frames)
    # Put on device memory once.
    tm_frames: Frames = tf.nest.map_structure(tf.convert_to_tensor, tm_frames)

    # tm_batch = batch._replace(frames=tm_frames)
    # tm_batch: Batch = tf.nest.map_structure(tf.convert_to_tensor, tm_batch)

    # Ideally this should be pushed into the compiled train_* functions, but
    # very strangely if we _don't_ do the reset here it results in an OOM in
    # train_q_function. Hopefully we don't take much of a performance hit
    # from this? (Compiling this also results in the same OOM.)
    initial_states = self.reset_initial_states(
        initial_states, batch.needs_reset)

    # TODO: take into account delay
    final_states = initial_states  # GC initial states as they are replaced
    metrics = {}

    (
      policy_samples,
      final_states['sample_policy'],
      metrics['sample_policy'],
    ) = self.train_sample_policy(
        tm_frames, initial_states['sample_policy'], train)

    (
      sample_q_values,
      final_states['q_function'],
      metrics['q_function'],
    ) = self.train_q_function(
        tm_frames, initial_states['q_function'], policy_samples, train)

    (
      final_states['q_policy'],
      metrics['q_policy'],
    ) = self.train_q_policy(
        tm_frames, initial_states['q_policy'], policy_samples, sample_q_values, train)

    # satisfy train_q_lib._get_loss
    metrics['total_loss'] = metrics['q_policy']['q_loss']

    return metrics, final_states
