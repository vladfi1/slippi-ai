import dataclasses
import typing as tp

import sonnet as snt
import tensorflow as tf

from slippi_ai.data import Frames
from slippi_ai.embed import StateAction
from slippi_ai.policies import Policy, UnrollOutputs
from slippi_ai.evaluators import Trajectory
from slippi_ai.networks import RecurrentState
from slippi_ai.controller_heads import ControllerType
from slippi_ai import value_function as vf_lib
from slippi_ai import tf_utils

@dataclasses.dataclass
class LearnerConfig:
  # TODO: unify this with the imitation config?
  learning_rate: float = 1e-4
  compile: bool = True
  policy_gradient_weight: float = 1
  kl_teacher_weight: float = 1e-1
  entropy_weight: float = 0
  value_cost: float = 0.5
  reward_halflife: float = 2  # measured in seconds

class LearnerState(tp.NamedTuple):
  teacher: RecurrentState
  value_function: RecurrentState

class LearnerOutputs(tp.NamedTuple):
  total_loss: tf.Tensor
  policy: UnrollOutputs
  metrics: dict

class Learner:
  """Implements A2C."""

  def __init__(
      self,
      config: LearnerConfig,
      policy: Policy,
      teacher: Policy,
      value_function: tp.Optional[vf_lib.ValueFunction] = None,
  ) -> None:
    self._config = config
    self._policy = policy
    self._teacher = teacher
    self._use_separate_vf = value_function is not None
    self._value_function = value_function or vf_lib.FakeValueFunction()

    self.policy_optimizer = snt.optimizers.Adam(config.learning_rate)
    self.value_optimizer = snt.optimizers.Adam(config.learning_rate)

    self.discount = 0.5 ** (1 / config.reward_halflife * 60)

    self.compiled_step = tf.function(self.step) if config.compile else self.step

  def initial_state(self, batch_size: int) -> LearnerState:
    return LearnerState(
        teacher=self._teacher.initial_state(batch_size),
        value_function=self._value_function.initial_state(batch_size),
    )

  def policy_variables(self) -> tp.Sequence[tf.Variable]:
    return self._policy.variables

  def _get_distribution(self, logits):
    """Returns a Controller-shaped structure of distributions."""
    # TODO: return an actual JointDistribution instead?
    return self._policy.controller_embedding.map(
        lambda e, t: e.distribution(t), logits)

  def _compute_kl(self, dist1, dist2):
    kls = self._policy.controller_embedding.map(
        lambda _, d1, d2: d1.kl_divergence(d2),
        dist1, dist2)
    return tf.add_n(list(self._policy.controller_embedding.flatten(kls)))

  def unroll(
      self,
      trajectory: Trajectory,
      initial_state: LearnerState,
  ) -> tp.Tuple[LearnerOutputs, LearnerState]:
    if trajectory.delayed_actions:
      raise NotImplementedError('Delayed actions not supported yet.')

    # Currently, resetting states can only occur on the first frame, which
    # conveniently means we don't have to deal with resets inside `unroll`.
    # Note that we also have to reset the policy state; this isn't visible to
    # the actor as it happens inside the agent (eval_lib.BasicAgent).
    is_resetting = trajectory.is_resetting[0]  # [B]
    batch_size = is_resetting.shape[0]
    initial_state, initial_policy_state = tf.nest.map_structure(
        lambda x, y: tf_utils.where(is_resetting, x, y),
        (self.initial_state(batch_size),
         self._policy.initial_state(batch_size)),
        (initial_state, trajectory.initial_state),
    )

    state_action = StateAction(
        state=trajectory.states,
        action=trajectory.actions.controller_state,
        name=trajectory.name,
    )
    frames = Frames(state_action, trajectory.rewards)

    policy_outputs = self._policy.unroll(
        frames=frames,
        initial_state=initial_policy_state,
        discount=self.discount,
    )

    teacher_outputs = self._teacher.unroll(
        # TODO: use the teacher's name instead?
        frames=frames,
        initial_state=initial_state.teacher,
        discount=self.discount,
    )

    controller_embedding = self._policy.controller_embedding

    # Drop the first action which precedes the first frame.
    actor_logits = tf.nest.map_structure(lambda t: t[1:], trajectory.actions.logits)
    actor_distribution = self._get_distribution(actor_logits)
    policy_distribution = self._get_distribution(policy_outputs.distances.logits)
    # No stop_gradient needed as the teacher's variables aren't trainable.
    teacher_distribution = self._get_distribution(teacher_outputs.distances.logits)

    def compute_entropy(dist):
      entropies = controller_embedding.map(lambda _, d: d.entropy(), dist)
      return tf.add_n(list(controller_embedding.flatten(entropies)))

    # Actor KL measures how off-policy the data is.
    actor_kl = self._compute_kl(actor_distribution, policy_distribution)

    # We take the "forward" KL to the teacher, which a) is more correct as the
    # trajectory and autoregressive actions are samples according to the
    # learned policy and b) incentivizes the agent to refine what humans do as
    # opposed to the usual "reverse" KL from supervised learning which forces
    # the policy to imitate all behaviors of the teacher, including mistakes.
    teacher_kl = self._compute_kl(policy_distribution, teacher_distribution)
    # Also compute reverse KL for logging.
    reverse_teacher_kl = self._compute_kl(teacher_distribution, policy_distribution)
    entropy = compute_entropy(actor_distribution)

    if self._use_separate_vf:
      value_ouputs, final_value_state = self._value_function.loss(
          frames=frames,
          initial_state=initial_state.value_function,
          discount=self.discount,
      )
    else:
      value_ouputs = policy_outputs.value_outputs
      final_value_state = initial_state.value_function

    advantages = tf.stop_gradient(value_ouputs.advantages)
    pg_loss = - policy_outputs.log_probs * advantages

    losses = [
        self._config.policy_gradient_weight * pg_loss,
        self._config.kl_teacher_weight * teacher_kl,
        self._config.value_cost * value_ouputs.loss,
        -self._config.entropy_weight * entropy,
    ]

    total_loss = tf.add_n(losses)

    metrics = dict(
        teacher_kl=teacher_kl,
        reverse_teacher_kl=reverse_teacher_kl,
        entropy=entropy,
        actor_kl=actor_kl,
        value=value_ouputs.metrics,
    )

    final_state = LearnerState(
        teacher=teacher_outputs.final_state,
        value_function=final_value_state,
    )

    outputs = LearnerOutputs(
        total_loss=total_loss,
        policy=policy_outputs,
        metrics=metrics,
    )

    return outputs, final_state

  def step(
      self,
      tm_trajectories: Trajectory,
      initial_state: LearnerState,
      train: bool = True,
      # Computing post-update metrics adds some overhead.
      with_post_update_metrics: bool = True,
  ) -> tuple[LearnerState, dict]:
    with tf.GradientTape() as tape:
      outputs, final_state = self.unroll(
          tm_trajectories,
          initial_state=initial_state,
      )

    metrics = outputs.metrics

    if train:
      params: tp.Sequence[tf.Variable] = tape.watched_variables()
      watched_names = [p.name for p in params]
      trainable_names = [v.name for v in self.trainable_variables]
      assert set(watched_names) == set(trainable_names)
      grads = tape.gradient(outputs.total_loss, params)
      self.optimizer.apply(grads, params)

      if with_post_update_metrics:
        post_update_outputs, _ = self.unroll(tm_trajectories, initial_state)
        pre_update_dist = self._get_distribution(
            outputs.policy.distances.logits)
        post_update_dist = self._get_distribution(
            post_update_outputs.policy.distances.logits)
        post_update_kl = self._compute_kl(pre_update_dist, post_update_dist)
        metrics.update(post_update_kl=dict(
            mean=tf.reduce_mean(post_update_kl),
            max=tf.reduce_max(post_update_kl),
        ))

    return final_state, metrics

  @property
  def trainable_variables(self) -> tp.Sequence[tf.Variable]:
    return (self._policy.trainable_variables +
            self._value_function.trainable_variables)

  def initialize(self, trajectory: Trajectory, pretraining_state: dict):
    """Initialize model and optimizer variables."""
    # Note: restore optimizer state, we need the optimizer to be initialized
    # in the same way as imitation learning. Imitation uses the variables in
    # the order returned by the gradient tape, which is construction order.
    batch_size = trajectory.is_resetting.shape[1]
    with tf.GradientTape() as value_tape:
      self.unroll(trajectory, self.initial_state(batch_size))
    self._value_vars = value_tape.watched_variables()
    tf_utils.assert_same_variables(
        self._value_vars, self._value_function.trainable_variables)
    self.value_optimizer._initialize(self._value_vars)

    with tf.GradientTape() as policy_tape:
      frames = get_frames(trajectory)
      self._policy.unroll(frames, trajectory.initial_state)
    self._policy_vars = policy_tape.watched_variables()
    tf_utils.assert_same_variables(
        self._policy_vars, self._policy.trainable_variables)
    self.policy_optimizer._initialize(self._policy_vars)

    # Imitation uses a single optimizer for both policy and value, constructing
    # the policy variables first. So to restore, we first create a dummy
    # optimizer which will mirror the imitation optimizer, and then copy over
    # the values into our real optimizers. Imitation creates the policy vars
    # first, so those will be the first to appear in the optimizer's m/v lists.
    # We need a Variable learning rate because imitation uses one and it
    # appears in the optimizer's variables.
    imitation_optimizer = snt.optimizers.Adam(
        tf.Variable(0, trainable=False, dtype=tf.float32))
    imitation_optimizer._initialize(self._policy_vars + self._value_vars)

    # This whole structure needs to conform imitation learning.
    tf_state = dict(
        policy=self._policy.variables,
        value_function=self._value_function.variables,
        optimizer=imitation_optimizer.variables,
    )

    # Restore variables from pretraining state into fake optimizer.
    pretraining_state = {
        k: v for k, v in pretraining_state.items() if k in tf_state}
    tf.nest.map_structure(
        lambda var, val: var.assign(val),
        tf_state, pretraining_state)

    # Now set the actual optimizer variables
    n = len(self._policy_vars)
    optimizer_state = dict(
        policy=dict(
            step=self.policy_optimizer.step,
            m=self.policy_optimizer.m,
            v=self.policy_optimizer.v,
        ),
        value=dict(
            step=self.value_optimizer.step,
            m=self.value_optimizer.m,
            v=self.value_optimizer.v,
        ),
    )
    to_restore = dict(
        policy=dict(
            step=imitation_optimizer.step,
            m=imitation_optimizer.m[:n],
            v=imitation_optimizer.v[:n],
        ),
        value=dict(
            step=imitation_optimizer.step,
            m=imitation_optimizer.m[n:],
            v=imitation_optimizer.v[n:],
        ),
    )
    tf.nest.map_structure(
        lambda var, val: var.assign(val),
        optimizer_state, to_restore)
