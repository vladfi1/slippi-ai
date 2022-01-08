from typing import List, Optional
import sonnet as snt
import tensorflow as tf
import slippi_ai.vtrace as vtrace
from slippi_ai.data import Batch

from slippi_ai.policies import Policy

def to_time_major(t):
  permutation = list(range(len(t.shape)))
  permutation[0] = 1
  permutation[1] = 0
  return tf.transpose(t, permutation)

def compute_baseline_loss(advantages):
  # Loss for the baseline, summed over the time dimension.
  # Multiply by 0.5 to match the standard update rule:
  # d(loss) / d(baseline) = advantage
  return .5 * tf.reduce_mean(tf.square(advantages))

def compute_policy_gradient_loss(action_logprobs, advantages):
  advantages = tf.stop_gradient(advantages)
  policy_gradient_loss_per_timestep = -action_logprobs * advantages
  return tf.reduce_mean(policy_gradient_loss_per_timestep)

class Learner:

  DEFAULT_CONFIG = dict(
      learning_rate=1e-4,
      compile=True,
      decay_rate=0.,
  )

  def __init__(self,
      learning_rate: float,
      compile: bool,
      policy: Policy,
      optimizer: Optional[snt.Optimizer] = None,
      decay_rate: Optional[float] = None,
  ):
    self.policy = policy
    self.optimizer = optimizer or snt.optimizers.Adam(learning_rate)
    self.decay_rate = decay_rate
    self.compiled_step = tf.function(self.step) if compile else self.step
    self.initial_state = policy.initial_state

  def step(self, batch: Batch, initial_states, train=True):
    bm_gamestate, restarting = batch

    # reset initial_states where necessary
    restarting = tf.expand_dims(restarting, -1)
    initial_states = tf.nest.map_structure(
        lambda x, y: tf.where(restarting, x, y),
        self.policy.initial_state(restarting.shape[0]),
        initial_states)

    # switch axes to time-major
    tm_gamestate = tf.nest.map_structure(to_time_major, bm_gamestate)

    with tf.GradientTape() as tape:
      loss, final_states, distances = self.policy.loss(
          tm_gamestate, initial_states)
      mean_loss = tf.reduce_mean(loss)

      # maybe do this in the Policy?
      counts = tf.cast(tm_gamestate.counts[1:] + 1, tf.float32)
      weighted_loss = tf.reduce_sum(loss) / tf.reduce_sum(counts)

    stats = dict(
        loss=mean_loss,
        weighted_loss=weighted_loss,
        distances=distances,
    )

    if train:
      params: List[tf.Variable] = tape.watched_variables()
      watched_names = [p.name for p in params]
      trainable_names = [v.name for v in self.policy.trainable_variables]
      assert set(watched_names) == set(trainable_names)
      grads = tape.gradient(mean_loss, params)
      self.optimizer.apply(grads, params)

      if self.decay_rate:
        for param in params:
          param.assign((1 - self.decay_rate) * param)

    return stats, final_states

# TODO: should this be a snt.Module?
class OfflineVTraceLearner:

  DEFAULT_CONFIG = dict(
      learning_rate=1e-4,
      compile=True,
      decay_rate=0.,
      value_cost=0.5,
      reward_halflife=2,  # measured in seconds
      teacher_cost=0.00025,
      train_behavior_policy=True,
  )

  def __init__(self,
      learning_rate: float,
      compile: bool,
      value_cost: float,
      reward_halflife: float,
      teacher_cost: float,
      target_policy: Policy,
      behavior_policy: Policy,
      train_behavior_policy: bool,
      optimizer: Optional[snt.Optimizer] = None,
      decay_rate: Optional[float] = None,
  ):
    self.policy = target_policy
    self.behavior_policy = behavior_policy
    self.optimizer = optimizer or snt.optimizers.Adam(learning_rate)
    self.decay_rate = decay_rate
    self.value_cost = value_cost
    self.discount = 0.5 ** (1 / reward_halflife * 60)
    self.teacher_cost = teacher_cost
    self.train_behavior_policy = train_behavior_policy
    self.compiled_step = tf.function(self.step) if compile else self.step

  def initial_state(self, batch_size: int):
    return (
      self.policy.initial_state(batch_size),
      self.behavior_policy.initial_state(batch_size),
    )

  def step(self, batch: Batch, initial_states, train=True):
    bm_gamestate, restarting = batch

    # reset initial_states where necessary
    restarting = tf.expand_dims(restarting, -1)

    initial_states = tf.nest.map_structure(
        lambda x, y: tf.where(restarting, x, y),
        self.initial_state(restarting.shape[0]),
        initial_states)
    target_initial, behavior_initial = initial_states

    # switch axes to time-major
    tm_gamestate = tf.nest.map_structure(to_time_major, bm_gamestate)

    rewards = tm_gamestate.rewards[1:]
    num_frames = tf.cast(tm_gamestate.counts[1:] + 1, tf.float32)
    discounts = tf.pow(tf.cast(self.discount, tf.float32), num_frames)

    with tf.GradientTape() as tape:

      target_logprobs, baseline, target_final = self.policy.run(
          tm_gamestate, target_initial)
      behavior_logprobs, _, behavior_final = self.behavior_policy.run(
          tm_gamestate, behavior_initial)

      log_rhos = target_logprobs - tf.stop_gradient(behavior_logprobs)
      values = baseline[:-1]
      bootstrap_value = baseline[-1]

      # with tf.device('/cpu'):
      vtrace_returns = vtrace.from_importance_weights(
          log_rhos=log_rhos,
          discounts=discounts,
          rewards=rewards,
          values=values,
          bootstrap_value=bootstrap_value,
      )

      total_loss = compute_policy_gradient_loss(
          target_logprobs,
          vtrace_returns.pg_advantages)

      value_loss = compute_baseline_loss(vtrace_returns.vs - values)
      value_stddev = tf.sqrt(tf.reduce_mean(value_loss))
      total_loss += self.value_cost * value_loss

      teacher_loss = -tf.reduce_mean(log_rhos)
      total_loss += self.teacher_cost * teacher_loss

      behavior_loss = -behavior_logprobs
      if self.train_behavior_policy:
        total_loss += behavior_loss

    final_states = (target_final, behavior_final)

    stats = dict(
        total_loss=total_loss,
        value_loss=value_loss,
        value_stddev=value_stddev,
        teacher_loss=teacher_loss,
        behavior_loss=behavior_loss,
    )
    stats = tf.nest.map_structure(tf.reduce_mean, stats)

    if train:
      params: List[tf.Variable] = tape.watched_variables()
      watched_names = [p.name for p in params]
      trainable_names = [v.name for v in self.policy.trainable_variables]
      # print(set(watched_names).difference(set(trainable_names)))
      assert set(watched_names) == set(trainable_names)
      grads = tape.gradient(total_loss, params)
      self.optimizer.apply(grads, params)

      if self.decay_rate:
        for param in params:
          param.assign((1 - self.decay_rate) * param)

    return stats, final_states
