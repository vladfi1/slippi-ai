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
  return .5 * tf.reduce_sum(tf.square(advantages))


def compute_entropy_loss(logits):
  policy = tf.nn.softmax(logits)
  log_policy = tf.nn.log_softmax(logits)
  entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
  return -tf.reduce_sum(entropy_per_timestep)


def compute_policy_gradient_loss(logits, actions, advantages):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=actions, logits=logits)
  advantages = tf.stop_gradient(advantages)
  policy_gradient_loss_per_timestep = cross_entropy * advantages
  return tf.reduce_sum(policy_gradient_loss_per_timestep)


# TODO: should this be a snt.Module?
class Learner:

  DEFAULT_CONFIG = dict(
      learning_rate=1e-4,
      compile=True,
      decay_rate=0.,
      reward_clipping='abs_one',
      discounting=0.99,
      baseline_cost=.5,
      entropy_cost=0.00025,
  )

  def __init__(self,
      learning_rate: float,
      compile: bool,
      reward_clipping: str,
      discounting: float,
      baseline_cost: float,
      entropy_cost: float,
      target_policy: Policy,
      behavior_policy:Policy,
      optimizer: Optional[snt.Optimizer] = None,
      decay_rate: Optional[float] = None,
  ):
    self.policy = target_policy
    self.behavior_policy=behavior_policy
    self.optimizer = optimizer or snt.optimizers.Adam(learning_rate)
    self.decay_rate = decay_rate
    self.compiled_step = tf.function(self.step) if compile else self.step
    self.reward_clipping = reward_clipping
    self.discounting=discounting
    self.baseline_cost=baseline_cost
    self.entropy_cost=entropy_cost


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

    rewards = tm_gamestate.rewards[1:]

    if self.reward_clipping == 'abs_one':
      clipped_rewards = tf.clip_by_value(rewards, -1, 1)
    elif self.reward_clipping == 'soft_asymmetric':
      squeezed = tf.tanh(rewards / 5.0)
      # Negative rewards are given less weight than positive rewards.
      clipped_rewards = tf.where(rewards < 0, .3 * squeezed, squeezed) * 5.

    discounts = tf.ones(tf.shape(clipped_rewards),dtype=tf.dtypes.float32) * self.discounting

    with tf.GradientTape() as tape:

      target_logits, baseline, actions = self.policy.run(tm_gamestate, initial_states)
      behavior_logits, _,_ = self.behavior_policy.run(tm_gamestate, initial_states)
      bootstrap_value = baseline[-1]

      with tf.device('/cpu'):
        vtrace_returns = vtrace.from_logits(
          behaviour_policy_logits=behavior_logits,
          target_policy_logits=target_logits,
          actions=actions,
          discounts=discounts,
          rewards=clipped_rewards,
          values=baseline,
          bootstrap_value=bootstrap_value)


      total_loss = compute_policy_gradient_loss(
        target_logits, actions,
        vtrace_returns.pg_advantages)
      total_loss += self.baseline_cost * compute_baseline_loss(
        vtrace_returns.vs - baseline)
      total_loss += self.entropy_cost * compute_entropy_loss(
        target_logits)

    stats = dict(
        loss=total_loss,
    )

    if train:
      params: List[tf.Variable] = tape.watched_variables()
      watched_names = [p.name for p in params]
      trainable_names = [v.name for v in self.policy.trainable_variables]
      print(set(watched_names).difference(set(trainable_names)))
      assert set(watched_names) == set(trainable_names)
      grads = tape.gradient(total_loss, params)
      self.optimizer.apply(grads, params)

      if self.decay_rate:
        for param in params:
          param.assign((1 - self.decay_rate) * param)

    return stats, final_states
