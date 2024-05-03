import dataclasses
import typing as tp

import sonnet as snt
import tensorflow as tf

from slippi_ai.data import Frames
from slippi_ai.embed import StateAction
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
    if trajectory.delayed_actions:
      raise NotImplementedError('Delayed actions not supported yet.')

    # TODO: take into account game resetting. This isn't necessary right
    # now because we set the actors to infinite time, meaning there are
    # never any resets except for the very first frame.

    state_action = StateAction(
        state=trajectory.states,
        action=trajectory.actions.controller_state,
        name=trajectory.name,
    )
    frames = Frames(state_action, trajectory.rewards)

    policy_outputs = self._policy.unroll(
        frames=frames,
        initial_state=trajectory.initial_state,
        discount=self.discount,
    )

    teacher_outputs = self._teacher.unroll(
        # TODO: use the teacher's name instead?
        frames=frames,
        initial_state=initial_teacher_states,
        discount=self.discount,
    )

    controller_embedding = self._policy.controller_embedding

    def get_distribution(logits):
      """Returns a Controller-shaped structure of distributions."""
      # TODO: return an actual JointDistribution instead?
      return controller_embedding.map(lambda e, t: e.distribution(t), logits)

    # Drop the first action which precedes the first frame.
    actor_logits = tf.nest.map_structure(lambda t: t[1:], trajectory.actions.logits)
    actor_distribution = get_distribution(actor_logits)
    policy_distribution = get_distribution(policy_outputs.distances.logits)
    # No stop_gradient needed as the teacher's variables aren't trainable.
    teacher_distribution = get_distribution(teacher_outputs.distances.logits)

    def compute_kl(dist1, dist2):
      kls = controller_embedding.map(
          lambda _, d1, d2: d1.kl_divergence(d2),
          dist1, dist2)
      return tf.add_n(list(controller_embedding.flatten(kls)))

    # Actor KL measures how
    actor_kl = compute_kl(actor_distribution, policy_distribution)

    # We take the "forward" KL to the teacher, which a) is more correct as the
    # trajectory and autoregressive actions are samples according to the
    # learned policy and b) incentivizes the agent to refine what humans do as
    # opposed to the usual "reverse" KL from supervised learning which forces
    # the policy to imitate all behaviors of the teacher, including mistakes.
    teacher_kl = compute_kl(policy_distribution, teacher_distribution)

    returns = policy_outputs.metrics['value']['return']
    advantages = returns - policy_outputs.values
    pg_loss = - policy_outputs.log_probs * tf.stop_gradient(advantages)

    metrics = policy_outputs.metrics

    losses = [
        pg_loss,
        self._config.kl_teacher_weight * teacher_kl,
        self._config.value_cost * metrics['value']['loss'],
    ]

    total_loss = tf.add_n(losses)

    metrics.update(
        total_loss=total_loss,
        teacher_kl=teacher_kl,
        actor_kl=actor_kl,
    )

    return total_loss, teacher_outputs.final_state, metrics

  def step(
      self,
      tm_trajectories: Trajectory,
  ) -> dict:
    with tf.GradientTape() as tape:
      loss, self._teacher_state, metrics = self.unroll(
          tm_trajectories,
          initial_teacher_states=self._teacher_state,
      )

    params: tp.Sequence[tf.Variable] = tape.watched_variables()
    watched_names = [p.name for p in params]
    trainable_names = [v.name for v in self._policy.trainable_variables]
    assert set(watched_names) == set(trainable_names)
    grads = tape.gradient(loss, params)
    self.optimizer.apply(grads, params)

    return metrics
