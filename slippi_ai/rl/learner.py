import dataclasses
import typing as tp

import numpy as np
import sonnet as snt
import tensorflow as tf

from slippi_ai.data import Frames
from slippi_ai.embed import StateAction
from slippi_ai.policies import Policy, UnrollOutputs, SampleOutputs
from slippi_ai.evaluators import Trajectory
from slippi_ai.networks import RecurrentState
from slippi_ai.controller_heads import ControllerType
from slippi_ai import value_function as vf_lib
from slippi_ai import tf_utils, utils

field = lambda f: dataclasses.field(default_factory=f)

@dataclasses.dataclass
class PPOConfig:
  num_epochs: int = 1
  num_batches: int = 1
  epsilon: float = 1e-2
  beta: float = 0
  minibatched: bool = False
  # target_kl: float = 1e-3

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
  ppo: PPOConfig = field(PPOConfig)

class LearnerState(tp.NamedTuple):
  teacher: RecurrentState
  value_function: RecurrentState

class LearnerOutputs(tp.NamedTuple):
  teacher: UnrollOutputs
  value: vf_lib.ValueOutputs

def get_frames(trajectory: Trajectory) -> Frames:
  state_action = StateAction(
      state=trajectory.states,
      action=trajectory.actions.controller_state,
      name=trajectory.name,
  )
  return Frames(state_action, trajectory.rewards)

def get_delayed_frames(trajectory: Trajectory) -> Frames:
  """Delay frames to prepare trajectory for policy unroll."""
  delay = len(trajectory.delayed_actions)

  delayed_actions = [
      sample_output.controller_state
      for sample_output in trajectory.delayed_actions
  ]
  # Add time dimension
  delayed_actions = tf.nest.map_structure(
      lambda t: tf.expand_dims(t, 0), delayed_actions)

  # Concatenate everything together.
  actions = tf.nest.map_structure(
      lambda *ts: tf.concat(ts, 0),
      trajectory.actions.controller_state, *delayed_actions)
  # Chop off the beginning _after_ concatenation to handle the case where
  # trajectory length < delay, which happens during initialization.
  actions = tf.nest.map_structure(lambda t: t[delay:], actions)

  state_action = StateAction(
      state=trajectory.states,
      action=actions,
      name=trajectory.name,
  )
  # Trajectory.rewards is technically wrong, but it's fine because
  # we don't use the policy's builtin value function anyways.
  return Frames(state_action, trajectory.rewards)

def combine_grads(x: tp.Optional[tf.Tensor], y: tp.Optional[tf.Tensor]):
  if x is None or y is None:
    return None
  return x + y

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

    self.discount = 0.5 ** (1 / (config.reward_halflife * 60))

    maybe_compile = tf.function if config.compile else lambda f: f

    self.compiled_unroll = maybe_compile(self.unroll)
    self.compiled_ppo_grads = maybe_compile(self.ppo_grads)
    self.compiled_ppo_grads_acc = maybe_compile(self.ppo_grads_acc)
    self.compiled_ppo = maybe_compile(self.ppo)

  def initial_state(self, batch_size: int) -> LearnerState:
    return LearnerState(
        teacher=self._teacher.initial_state(batch_size),
        value_function=self._value_function.initial_state(batch_size),
    )

  def policy_variables(self) -> tp.Sequence[tf.Variable]:
    return self._policy.variables

  def _get_distribution(self, logits: ControllerType):
    """Returns a Controller-shaped structure of distributions."""
    # TODO: return an actual JointDistribution instead?
    return self._policy.controller_embedding.map(
        lambda e, t: e.distribution(t), logits)

  def _compute_kl(self, dist1: ControllerType, dist2: ControllerType):
    kls = self._policy.controller_embedding.map(
        lambda _, d1, d2: d1.kl_divergence(d2),
        dist1, dist2)
    return tf.add_n(list(self._policy.controller_embedding.flatten(kls)))

  def _get_log_prob(self, logits: ControllerType, action: ControllerType):
    controller_embedding = self._policy.controller_embedding
    distances = controller_embedding.map(
        lambda e, t, a: e.distance(t, a), logits, action)
    return - tf.add_n(list(controller_embedding.flatten(distances)))

  def _compute_entropy(self, dist: ControllerType):
    controller_embedding = self._policy.controller_embedding
    entropies = controller_embedding.map(lambda _, d: d.entropy(), dist)
    return tf.add_n(list(controller_embedding.flatten(entropies)))

  def unroll(
      self,
      trajectory: Trajectory,
      initial_state: LearnerState,
      train_value_function: bool = False,
  ) -> tp.Tuple[LearnerOutputs, LearnerState]:
    assert len(trajectory.delayed_actions) == self._policy.delay

    # Currently, resetting states can only occur on the first frame, which
    # conveniently means we don't have to deal with resets inside `unroll`.
    is_resetting = trajectory.is_resetting[0]  # [B]
    batch_size = is_resetting.shape[0]
    initial_state = tf.nest.map_structure(
        lambda x, y: tf_utils.where(is_resetting, x, y),
        self.initial_state(batch_size), initial_state)

    teacher_outputs = self._teacher.unroll(
        # TODO: use the teacher's name instead?
        frames=get_delayed_frames(trajectory),
        initial_state=initial_state.teacher,
        discount=self.discount,
    )

    with tf.GradientTape() as tape:
      value_ouputs, final_value_state = self._value_function.loss(
          frames=get_frames(trajectory),
          initial_state=initial_state.value_function,
          discount=self.discount,
      )
      if train_value_function:
        grads = tape.gradient(value_ouputs.loss, self._value_vars)
        self.value_optimizer.apply(grads, self._value_vars)

    final_state = LearnerState(
        teacher=teacher_outputs.final_state,
        value_function=final_value_state,
    )

    outputs = LearnerOutputs(
        teacher=teacher_outputs,
        value=value_ouputs,
    )

    return outputs, final_state

  def ppo_grads(self, outputs: LearnerOutputs, trajectory: Trajectory):
    # Value function outputs are for [0, U] while policy outputs are for
    # [D, U+D]. This means we can only train on steps [D, U].

    delay = self._policy.delay  # "D"
    remove_first = lambda t: t[delay:]
    remove_last = lambda t: t[:t.shape[0] - delay]

    advantages = outputs.value.advantages[delay:]  # [0, U] -> [D, U]

    # Teacher logits are between [D, U+D]; truncate to [D, U]
    # Note: no stop_gradient needed as the teacher's variables aren't trainable.
    teacher_logits = tf.nest.map_structure(
        remove_last, outputs.teacher.distances.logits)
    teacher_distribution = self._get_distribution(teacher_logits)

    del outputs

    # We also have to reset the policy state; this isn't visible to
    # the actor as it happens inside the agent (eval_lib.BasicAgent).
    is_resetting = trajectory.is_resetting[0]  # [B]
    batch_size = is_resetting.shape[0]
    initial_policy_state = tf.nest.map_structure(
        lambda x, y: tf_utils.where(is_resetting, x, y),
        self._policy.initial_state(batch_size), trajectory.initial_state)

    policy_frames = Frames(
        state_action=StateAction(
            state=tf.nest.map_structure(remove_last, trajectory.states),
            action=tf.nest.map_structure(
                remove_first, trajectory.actions.controller_state),
            name=remove_last(trajectory.name),
        ),
        reward=remove_first(trajectory.rewards),
    )

    # For the actor, also drop the first action which precedes the first frame.
    actions: SampleOutputs = tf.nest.map_structure(
        lambda t: t[1+delay:], trajectory.actions)
    actor_distribution = self._get_distribution(actions.logits)
    actor_log_probs = self._get_log_prob(
        actions.logits, actions.controller_state)
    del trajectory

    with tf.GradientTape() as tape:
      policy_outputs = self._policy.unroll(
          frames=policy_frames,
          initial_state=initial_policy_state,
          discount=self.discount,
      )
      policy_distribution = self._get_distribution(policy_outputs.distances.logits)
      entropy = self._compute_entropy(policy_distribution)

      # We take the "forward" KL to the teacher, which a) is more correct as the
      # trajectory and autoregressive actions are samples according to the
      # learned policy and b) incentivizes the agent to refine what humans do as
      # opposed to the usual "reverse" KL from supervised learning which forces
      # the policy to imitate all behaviors of the teacher, including mistakes.
      teacher_kl = self._compute_kl(policy_distribution, teacher_distribution)
      actor_kl = self._compute_kl(actor_distribution, policy_distribution)

      log_rhos = policy_outputs.log_probs - actor_log_probs
      rhos = tf.exp(log_rhos)

      eps = self._config.ppo.epsilon
      clipped_log_rhos = tf.clip_by_value(log_rhos, -eps, eps)
      clipped_rhos = tf.exp(clipped_log_rhos)

      ppo_objective = tf.minimum(rhos * advantages, clipped_rhos * advantages)

      weighted_losses = [
          - self._config.policy_gradient_weight * ppo_objective,
          self._config.ppo.beta * actor_kl,
          self._config.kl_teacher_weight * teacher_kl,
          -self._config.entropy_weight * entropy,
      ]
      loss = tf.add_n(weighted_losses)
      grads = tape.gradient(loss, self._policy_vars)
      # tf.while_loop doesn't like None's in the loop vars
      grads = [
          tf.zeros_like(v) if g is None else g
          for g, v in zip(grads, self._policy_vars)]

    metrics = dict(
        total_loss=loss,
        ppo_objective=ppo_objective,
        teacher_kl=teacher_kl,
        entropy=entropy,
        actor_kl=actor_kl,
    )

    return grads, metrics

  def ppo_grads_acc(self, outputs: LearnerOutputs, trajectory: Trajectory, grads_acc: list):
    grads, metrics = self.compiled_ppo_grads(outputs, trajectory)
    grads_acc = [a + g for a, g in zip(grads_acc, grads)]
    return metrics, grads_acc

  @tf.function
  def apply_grads(self, grads, scale: float = 1):
    grads = [g * scale for g in grads]
    self.policy_optimizer.apply(grads, self._policy_vars)

  def ppo_epoch_full(
      self,
      learner_outputs: list[LearnerOutputs],
      trajectories: list[Trajectory],
      train: bool,
  ):
    # Could cache this?
    grads_acc = [np.zeros(v.shape, dtype=v.dtype.as_numpy_dtype()) for v in self._policy_vars]
    metrics_acc = []

    metrics_acc = []
    for outputs, trajectory in zip(learner_outputs, trajectories):
      metrics, grads_acc = self.compiled_ppo_grads_acc(outputs, trajectory, grads_acc)
      metrics_acc.append(metrics)

    if train:
      self.apply_grads(grads_acc, scale=1 / len(learner_outputs))

    metrics_acc = tf.nest.map_structure(lambda t: t.numpy(), metrics_acc)
    metrics = utils.batch_nest(metrics_acc)

    # Make sure to take max over whole epoch, not just over the minibatch.
    actor_kl = metrics['actor_kl']
    metrics['actor_kl'] = dict(
        mean=np.mean(actor_kl),
        max=np.amax(actor_kl),
    )
    return metrics

  @tf.function(autograph=False)
  def ppo_epoch_full_tf(
      self,
      learner_outputs: list[LearnerOutputs],
      trajectories: list[Trajectory],
      train: bool,
  ):
    # Accumulate gradients across the entire batch.
    grads_acc = [tf.zeros_like(v) for v in self._policy_vars]

    # def body(inputs, grads_acc: list):
    #   learner_output, trajectory = inputs
    #   grads, metrics = self.ppo_grads(learner_output, trajectory)
    #   grads_acc = [combine_grads(a, g) for a, g in zip(grads_acc, grads)]
    #   return metrics, grads_acc

    # metrics, grads = tf_utils.dynamic_rnn(
    #     body, (learner_outputs, trajectories), grads_acc)

    metrics_acc = []
    for outputs, trajectory in zip(learner_outputs, trajectories):
      with tf.control_dependencies(grads_acc):
        metrics, grads_acc = self.compiled_ppo_grads_acc(outputs, trajectory, grads_acc)
        metrics_acc.append(metrics)

    metrics = tf.nest.map_structure(lambda *xs: tf.stack(xs), *metrics_acc)

    if train:
      self.policy_optimizer.apply(grads_acc, self._policy_vars)

    # Make sure to take max over whole epoch, not just over the minibatch.
    actor_kl = metrics['actor_kl']
    metrics['actor_kl'] = dict(
        mean=tf.reduce_mean(actor_kl),
        max=tf.reduce_max(actor_kl),
    )
    return metrics

  @tf.function
  def ppo_batch(self, outputs: LearnerOutputs, trajectory: Trajectory, train: bool):
    grads, metrics = self.compiled_ppo_grads(outputs, trajectory)
    if train:
      self.policy_optimizer.apply(grads, self._policy_vars)
    return metrics

  def ppo_epoch_batched(
      self,
      learner_outputs: list[LearnerOutputs],
      trajectories: list[Trajectory],
      train: bool,
  ):
    """Per-minibatch gradients."""
    metrics = []
    for outputs, trajectory in zip(learner_outputs, trajectories):
      metrics.append(self.ppo_batch(outputs, trajectory, train))

    metrics = tf.nest.map_structure(lambda t: t.numpy(), metrics)
    metrics = utils.batch_nest(metrics)

    # Make sure to take max over whole epoch, not just over the minibatch.
    actor_kl = metrics['actor_kl']
    metrics['actor_kl'] = dict(
        mean=np.mean(actor_kl),
        max=np.amax(actor_kl),
    )
    return metrics

  def ppo(
      self,
      trajectories: list[Trajectory],
      initial_state: LearnerState,
      num_epochs: int = None,
  ) -> tuple[LearnerState, dict]:
    assert self._use_separate_vf

    learner_outputs: list[LearnerOutputs] = []

    hidden_state = initial_state
    for trajectory in trajectories:
      outputs, hidden_state = self.compiled_unroll(
          trajectory, hidden_state, train_value_function=True)
      learner_outputs.append(outputs)

    value_metrics = [outputs.value.metrics for outputs in learner_outputs]
    value_metrics = utils.map_single_structure(
        lambda t: t.numpy(), value_metrics)
    value_metrics = utils.batch_nest(value_metrics)

    if num_epochs is None:
      num_epochs = self._config.ppo.num_epochs
    # learner_outputs = utils.batch_nest(learner_outputs)
    # trajectories = utils.batch_nest(trajectories)

    if self._config.ppo.minibatched:
      ppo_epoch = self.ppo_epoch_batched
    else:
      ppo_epoch = self.ppo_epoch_full

    per_epoch_metrics = []
    for _ in range(num_epochs):
      per_epoch_metrics.append(ppo_epoch(learner_outputs, trajectories, train=True))
    # Just for logging.
    per_epoch_metrics.append(ppo_epoch(learner_outputs, trajectories, train=False))

    metrics = dict(
        ppo_step={str(i): d for i, d in enumerate(per_epoch_metrics)},
        post_update=per_epoch_metrics[-1],
        value=value_metrics,
    )

    return hidden_state, metrics

  @property
  def trainable_variables(self) -> tp.Sequence[tf.Variable]:
    return (self._policy.trainable_variables +
            self._value_function.trainable_variables)

  def initialize(self, trajectory: Trajectory):
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

  def restore_from_imitation(self, imitation_state: dict):
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
    state = {k: v for k, v in imitation_state.items() if k in tf_state}
    tf.nest.map_structure(
        lambda var, val: var.assign(val),
        tf_state, state)

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

  def get_vars(self) -> dict:
    return dict(
        policy=self._policy.variables,
        value_function=self._value_function.variables,
        optimizers=dict(
            policy=self.policy_optimizer.variables,
            value=self.value_optimizer.variables,
        ),
    )

  def get_state(self) -> dict:
    return tf.nest.map_structure(lambda t: t.numpy(), self.get_vars())
