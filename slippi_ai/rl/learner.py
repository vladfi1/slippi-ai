import dataclasses
import typing as tp

import sonnet as snt
import tensorflow as tf

from slippi_ai import (
    embed,
    rl_lib,
    saving,
    utils,
)
from slippi_ai.rl.actor import Trajectory
from slippi_ai.networks import RecurrentState

@dataclasses.dataclass
class LearnerConfig:
  # TODO: unify this with the imitation config?
  learning_rate: float = 1e-4
  compile: bool = True
  kl_teacher_weight: float = 1e-1
  value_cost: float = 0.5
  reward_halflife: float = 2  # measured in seconds

class Learner(snt.Module):
  """Implements A2C."""

  def __init__(
      self,
      config: LearnerConfig,
      teacher_state: dict,
      rl_state: dict,
      batch_size: int,
      name='RlLearner',
  ) -> None:
    super().__init__(name)
    self._config = config
    self._teacher = saving.load_policy_from_state(teacher_state)
    self._policy = saving.load_policy_from_state(rl_state)
    self.optimizer = snt.optimizers.Adam(config.learning_rate)
    self._value_constant = tf.Variable(0, dtype=tf.float32, name='value_constant')

    self.discount = 0.5 ** (1 / config.reward_halflife * 60)

    self.compiled_step = tf.function(self.step) if config.compile else self.step

    self._teacher_state = self._teacher.initial_state(batch_size)

  def policy_variables(self) -> tp.Sequence[tf.Variable]:
    return self._policy.variables

  def unroll(
      self,
      bm_trajectories: Trajectory,
      initial_teacher_states: RecurrentState,
  ):
    # TODO: handle game resets?

    tm_gamestate: embed.StateActionReward = tf.nest.map_structure(
        utils.to_time_major, bm_trajectories.observations)

    policy_outputs = self._policy.unroll(
        state_action=tm_gamestate,
        initial_state=bm_trajectories.initial_state,
        discount=self.discount,
    )

    teacher_outputs = self._teacher.unroll(
        state_action=tm_gamestate,
        initial_state=initial_teacher_states,
        discount=self.discount,
    )

    # TODO: this is a high-variance estimator; we can do better by
    # including the logits from policy and teacher.
    kl = policy_outputs.log_probs - teacher_outputs.log_probs
    entropy = -policy_outputs.log_probs

    rewards = tf.cast(tm_gamestate.reward[1:], tf.float32)

    # Include the KL in the rewards so that the value function acts as
    # variance reduction. TODO: check that backing up across states is correct.
    rewards = rewards - self._config.kl_teacher_weight * kl

    discounts = tf.cast(self.discount, tf.float32) * tf.ones_like(rewards)
    returns = rl_lib.discounted_returns(
        rewards=rewards,
        discounts=discounts,
        bootstrap=policy_outputs.values[-1])
    returns = tf.stop_gradient(returns)

    advantages = returns - policy_outputs.values[:-1]
    pg_loss = - policy_outputs.log_probs * tf.stop_gradient(advantages)

    # Train the constant value.
    constant_value_loss = tf.square(returns - self._value_constant)
    return_variance = tf.reduce_mean(constant_value_loss)

    # Here we could instead take the ratio per-step (pre-reduce_mean).
    # This would up-weight states with low return variance.
    value_loss = tf.square(advantages)
    uev = tf.reduce_mean(value_loss) / (return_variance + 1e-8)

    losses = [
        pg_loss,
        self._config.value_cost * value_loss,
        constant_value_loss,  # train self._constant_value
    ]
    total_loss = tf.add_n(losses)

    metrics = dict(
        total_loss=total_loss,
        kl=kl,
        entropy=tf.reduce_mean(entropy),
        value=dict(
            loss=value_loss,
            return_mean=tf.reduce_mean(returns),
            return_variance=return_variance,
            uev=uev,
        ),
    )

    return total_loss, teacher_outputs.final_state, metrics

  def step(
      self,
      bm_trajectories: Trajectory,
      initial_teacher_states: RecurrentState,
  ) -> tp.Tuple[RecurrentState, dict]:
    with tf.GradientTape() as tape:
      loss, final_teacher_states, metrics = self.unroll(
          bm_trajectories,
          initial_teacher_states=initial_teacher_states,
      )

    params: tp.Sequence[tf.Variable] = tape.watched_variables()
    watched_names = [p.name for p in params]
    trainable_names = [v.name for v in self.variables if v.trainable]
    assert set(watched_names) == set(trainable_names)
    grads = tape.gradient(loss, params)
    self.optimizer.apply(grads, params)

    return final_teacher_states, metrics

  def train(
      self,
      bm_trajectories: Trajectory,
  ) -> dict:
    self._teacher_state, metrics = self.compiled_step(
        bm_trajectories, self._teacher_state)
    return metrics
