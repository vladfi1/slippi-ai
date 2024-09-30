import dataclasses

import sonnet as snt
import tensorflow as tf

from slippi_ai.data import Batch, Frames
from slippi_ai.policies import Policy, RecurrentState
from slippi_ai import q_function as q_lib
from slippi_ai import tf_utils

def swap_axes(t, axis1=0, axis2=1):
  permutation = list(range(len(t.shape)))
  permutation[axis2] = axis1
  permutation[axis1] = axis2
  return tf.transpose(t, permutation)

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  compile: bool = True
  reward_halflife: float = 8

  num_samples: int = 1
  q_policy_imitation_weight: float = 0

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
  ):
    self.q_function = q_function
    self.sample_policy = sample_policy
    self.q_policy = q_policy
    self.q_function_optimizer = snt.optimizers.Adam(learning_rate)
    self.sample_policy_optimizer = snt.optimizers.Adam(learning_rate)
    self.q_policy_optimizer = snt.optimizers.Adam(learning_rate)
    self.discount = 0.5 ** (1 / (reward_halflife * 60))
    self.compiled_step = tf.function(self.step) if compile else self.step
    self.num_samples = num_samples
    self.q_policy_imitation_weight = q_policy_imitation_weight

    assert q_policy.delay == 0
    assert sample_policy.delay == 0

  def initial_state(self, batch_size: int) -> RecurrentState:
    return dict(
        q_function=self.q_function.initial_state(batch_size),
        q_policy=self.q_policy.initial_state(batch_size),
        sample_policy=self.sample_policy.initial_state(batch_size),
    )

  def step(
      self,
      batch: Batch,
      initial_states: RecurrentState,
      train: bool = True,
  ):
    bm_frames = batch.frames
    restarting = batch.needs_reset
    batch_size = restarting.shape[0]

    # reset initial_states where necessary
    restarting = tf.expand_dims(restarting, -1)
    initial_states = tf.nest.map_structure(
        lambda x, y: tf.where(restarting, x, y),
        self.initial_state(batch_size),
        initial_states)

    # switch axes to time-major
    tm_frames: Frames = tf.nest.map_structure(
        swap_axes, bm_frames)

    # TODO: take into account delay

    action = tm_frames.state_action.action
    prev_action = tf.nest.map_structure(lambda t: t[:-1], action)
    next_action = tf.nest.map_structure(lambda t: t[1:], action)

    to_train: list[tuple[list[tf.Tensor], list[tf.Variable], snt.Optimizer]] = []

    # Train sample policy with imitation
    with tf.GradientTape() as tape:
      sample_policy_outputs = self.sample_policy.unroll_with_outputs(
          tm_frames, initial_states['sample_policy'])
      sample_policy_loss = sample_policy_outputs.imitation_loss

      sample_policy_vars = self.sample_policy.trainable_variables
      tf_utils.assert_same_variables(tape.watched_variables(), sample_policy_vars)
      self.sample_policy_optimizer._initialize(sample_policy_vars)

      if train:
        sample_policy_grads = tape.gradient(sample_policy_loss, sample_policy_vars)
        to_train.append((
            sample_policy_grads,
            sample_policy_vars,
            self.sample_policy_optimizer,
        ))

    # TODO: use tf.tile instead?
    replicate = lambda n: lambda t: tf.stack([t] * n)
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

    num_samples = self.num_samples + 1
    replicate_samples = lambda nest: tf.nest.map_structure(
        replicate(num_samples), nest)

    # At this point tensorflow should be able to GC the large
    # sample_policy_outputs.outputs tensor.

    # Train q function with regression
    with tf.GradientTape() as tape:
      # TODO: take into account delay
      q_outputs, q_final_states = self.q_function.loss(
          tm_frames, initial_states['q_function'], self.discount)

      q_function_vars = self.q_function.trainable_variables
      tf_utils.assert_same_variables(tape.watched_variables(), q_function_vars)
      self.q_function_optimizer._initialize(q_function_vars)

      if train:
        q_function_grads = tape.gradient(q_outputs.loss, q_function_vars)
        to_train.append((
            q_function_grads,
            q_function_vars,
            self.q_function_optimizer,
        ))

    # Compute the q-values of the sampled actions
    replicated_hidden_states = replicate_samples(q_outputs.hidden_states)
    sample_q_values = snt.BatchApply(
        self.q_function.q_values_from_hidden_states, num_dims=3)(
            hidden_states=replicated_hidden_states,
            actions=policy_samples,
        )

    # Train the q_policy by argmaxing the q_function over the sample_policy
    with tf.control_dependencies([sample_q_values]), tf.GradientTape() as tape:
      q_policy_outputs = self.q_policy.unroll_with_outputs(
          tm_frames, initial_states['q_policy'])

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

      q_policy_total_loss = q_policy_q_loss
      q_policy_total_loss += self.q_policy_imitation_weight * q_policy_imitation_loss

      q_policy_metrics = dict(
          q_loss=q_policy_q_loss,
          imitation_loss=q_policy_imitation_loss,
      )

      q_policy_vars = self.q_policy.trainable_variables
      q_policy_grads = tape.gradient(q_policy_total_loss, q_policy_vars)
      self.q_policy_optimizer._initialize(q_policy_vars)

      if train:
        to_train.append((
            q_policy_grads,
            q_policy_vars,
            self.q_policy_optimizer,
        ))

    for grads, vars, optimizer in to_train:
      optimizer.apply(grads, vars)

    final_states = dict(
        q_function=q_final_states,
        q_policy=q_policy_outputs.final_state,
        sample_policy=sample_policy_outputs.final_state,
    )

    metrics = dict(
        total_loss=q_policy_q_loss,  # satisfy train_q_lib._get_loss
        q_function=q_outputs.metrics,
        q_policy=q_policy_metrics,
        sample_policy=sample_policy_outputs.metrics,
    )

    return metrics, final_states
