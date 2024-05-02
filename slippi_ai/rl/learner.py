import dataclasses
import typing as tp

import sonnet as snt
import tensorflow as tf

from slippi_ai.policies import Policy
from slippi_ai.evaluators import Trajectory
from slippi_ai.networks import RecurrentState
from slippi_ai import value_function as vf_lib

@dataclasses.dataclass
class LearnerConfig:
  # TODO: unify this with the imitation config?
  learning_rate: float = 1e-4
  compile: bool = True
  kl_teacher_weight: float = 1e-1
  value_cost: float = 0.5
  reward_halflife: float = 2  # measured in seconds

class Learner:
  """Implements A2C."""

  def __init__(
      self,
      config: LearnerConfig,
      policy: Policy,
      teacher: Policy,
      batch_size: int,
      value_function: tp.Optional[vf_lib.ValueFunction] = None,
  ) -> None:
    self._config = config
    self._policy = policy
    self._teacher = teacher
    self._value_function = value_function or vf_lib.FakeValueFunction()

    # TODO: init from the imitation optimizer
    self.optimizer = snt.optimizers.Adam(config.learning_rate)

    self.discount = 0.5 ** (1 / config.reward_halflife * 60)

    self.compiled_step = tf.function(self.step) if config.compile else self.step

    self._teacher_state = self._teacher.initial_state(batch_size)

  def policy_variables(self) -> tp.Sequence[tf.Variable]:
    return self._policy.variables

  def unroll(
      self,
      trajectory: Trajectory,
      initial_teacher_states: RecurrentState,
  ):
    policy_outputs = self._policy.unroll(
        frames=trajectory.frames,
        initial_state=trajectory.initial_state,
        discount=self.discount,
    )

    teacher_outputs = self._teacher.unroll(
        frames=trajectory.frames,
        initial_state=initial_teacher_states,
        discount=self.discount,
    )

    returns = policy_outputs.metrics['value']['return']
    advantages = returns - policy_outputs.values
    pg_loss = - policy_outputs.log_probs * tf.stop_gradient(advantages)

    # TODO: this is a high-variance estimator; we can do better by
    # including the logits from policy and teacher.
    kl = policy_outputs.log_probs - teacher_outputs.log_probs

    # grad-exp trick
    kl_teacher_loss = policy_outputs.log_probs * tf.stop_gradient(kl)

    metrics = policy_outputs.metrics

    losses = [
        pg_loss,
        self._config.kl_teacher_weight * kl_teacher_loss,
        self._config.value_cost * metrics['value']['loss'],
    ]

    total_loss = tf.add_n(losses)

    metrics.update(
        total_loss=total_loss,
        kl=kl,
    )

    return total_loss, teacher_outputs.final_state, metrics

  def step(
      self,
      bm_trajectories: Trajectory,
  ) -> dict:
    with tf.GradientTape() as tape:
      loss, self._teacher_state, metrics = self.unroll(
          bm_trajectories,
          initial_teacher_states=self._teacher_state,
      )

    params: tp.Sequence[tf.Variable] = tape.watched_variables()
    watched_names = [p.name for p in params]
    trainable_names = [v.name for v in self._policy.trainable_variables]
    assert set(watched_names) == set(trainable_names)
    grads = tape.gradient(loss, params)
    self.optimizer.apply(grads, params)

    return metrics
