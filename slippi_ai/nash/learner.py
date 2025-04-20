import dataclasses
import typing as tp
from typing import Optional

import numpy as np
import sonnet as snt
import tensorflow as tf

from slippi_ai.data import Batch, Frames
from slippi_ai.policies import Policy, RecurrentState
from slippi_ai.nash import q_function as q_lib
from slippi_ai import tf_utils, embed
from slippi_ai.controller_heads import ControllerType
from slippi_ai.nash import data as nash_data
from slippi_ai.nash import nash as nash_lib
from slippi_ai.nash import optimization as opt_lib

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
  reward_halflife: float = 4

  train_sample_policy: bool = True

  num_samples: int = 1
  q_policy_imitation_weight: float = 0

  # Only train the q_policy if the q function is sufficiently good.
  min_uev_delta: Optional[float] = None

replicate = lambda n: lambda t: tf_utils.expand_tile(t, axis=0, multiple=n)

T = tp.TypeVar('T')

def identity(x: T) -> T:
  return x

def solve_nash(payoff_matrices: tf.Tensor):
  nash_variables, stats = nash_lib.solve_zero_sum_nash_tf(
      payoff_matrices=tf.cast(payoff_matrices, tf.float64),
      optimization_solver=opt_lib.solve_optimization_interior_point_primal_dual,
      error=1e-5,
      is_linear=True,
  )
  nash_variables = tf.nest.map_structure(
      lambda x: tf.cast(x, payoff_matrices.dtype), nash_variables)
  return nash_variables, stats

def entropy(probs: tf.Tensor, axis=-1) -> tf.Tensor:
  safe_log_probs = tf.where(probs > 0, tf.math.log(probs), 0)
  return -tf.reduce_sum(probs * safe_log_probs, axis=axis)

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
      train_sample_policy: bool = True,
      q_policy_imitation_weight: float = 0,
      min_uev_delta: Optional[float] = None,
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

    self.should_train_sample_policy = train_sample_policy
    self.train_sample_policy = maybe_compile(self._train_sample_policy)
    self.train_q_function = maybe_compile(self._train_q_function)
    # self.compute_nash = maybe_compile(self._compute_nash)
    self.train_q_policy = maybe_compile(self._train_q_policy)

    self.num_samples = num_samples
    self.q_policy_imitation_weight = q_policy_imitation_weight
    self.min_uev_delta = min_uev_delta

    self.delay = q_policy.delay
    assert sample_policy.delay == self.delay

  def initial_state(self, batch_size: int) -> RecurrentState:
    return dict(
        q_function=self.q_function.initial_state(batch_size),
        q_policy=[self.q_policy.initial_state(batch_size)] * 2,
        sample_policy=[self.sample_policy.initial_state(batch_size)] * 2,
    )

  def initialize_variables(self):
    self.sample_policy.initialize_variables()
    self.sample_policy_optimizer._initialize(self.sample_policy.trainable_variables)

    self.q_policy.initialize_variables()
    self.q_policy_optimizer._initialize(self.q_policy.trainable_variables)

    self.q_function.initialize_variables()
    self.q_function_optimizer._initialize(self.q_function.trainable_variables)

  def _get_delayed_frames(self, frames: Frames) -> Frames:
    state_action = frames.state_action
    # Includes "overlap" frame.
    unroll_length = state_action.state.stage.shape[0] - self.delay

    return Frames(
        state_action=embed.StateAction(
            state=tf.nest.map_structure(
                lambda t: t[:unroll_length], state_action.state),
            action=tf.nest.map_structure(
                lambda t: t[self.delay:], state_action.action),
            name=state_action.name[:unroll_length],
        ),
        is_resetting=frames.is_resetting[:unroll_length],
        # Only use rewards that follow actions.
        reward=frames.reward[self.delay:],
    )

  def _get_distribution(self, logits: ControllerType):
    """Returns a Controller-shaped structure of distributions."""
    # TODO: return an actual JointDistribution instead?
    return self.q_policy.controller_embedding.map(
        lambda e, t: e.distribution(t), logits)

  def _compute_entropy(self, dist: ControllerType):
    controller_embedding = self.q_policy.controller_embedding
    entropies = controller_embedding.map(lambda _, d: d.entropy(), dist)
    return tf.add_n(list(controller_embedding.flatten(entropies)))

  def _compute_kl(self, dist1: ControllerType, dist2: ControllerType):
    controller_embedding = self.q_policy.controller_embedding
    kls = controller_embedding.map(
        lambda _, d1, d2: d1.kl_divergence(d2),
        dist1, dist2)
    return tf.add_n(list(controller_embedding.flatten(kls)))

  def _train_sample_policy(
      self,
      frames: Frames, # merged
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

    # merged_initial_states = nash_data.merge(initial_states, axis=0)

    frames = self._get_delayed_frames(frames)

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
    policy_sample_outputs = self.sample_policy.controller_head.sample(
        replicated_sample_policy_outputs, replicated_prev_action)

    # Include the actual action taken among the samples.
    policy_samples = tf.nest.map_structure(
        lambda samples, na: tf.concat([samples, tf.expand_dims(na, 0)], 0),
        policy_sample_outputs.controller_state, next_action,
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
        sample_policy_outputs.metrics,
        sample_policy_outputs.distances.logits,  # Only for action taken.
    )

  def _train_q_function(
      self,
      # batch: Batch,
      frames: Frames,  # [T, 2B ...]
      initial_states: RecurrentState,  # [2B, ...]
      policy_samples: embed.Action,  # [S, T, 2B, ...]
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

    frames = self._get_delayed_frames(frames)

    # Train q function with regression
    with tf.GradientTape() as tape:
      # TODO: take into account delay
      q_outputs, q_final_states = self.q_function.loss(
          frames, initial_states, self.discount)

    # Compute the q-values of the sampled actions
    sample_q_values = self.q_function.multi_q_values_from_hidden_states(
        hidden_states=q_outputs.hidden_states,
        actions=policy_samples,
    )

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

  @tf.function(jit_compile=False)  # the solver is faster without jit
  def _compute_nash(self, q_values: tf.Tensor) -> nash_lib.NashVariables:
    # q_values: [2, S, S, T, B]
    N, S1, S2, T, B = q_values.shape
    assert N == 2

    # Average the estimates from each player's point of view.
    q_values = (q_values[0] - q_values[1]) / 2  # [S, S, T, B]

    # q_values = tf.reshape(q_values, [S1, S2, T * B])
    q_values = tf.transpose(q_values, [2, 3, 0, 1])  # [T, B, S1, S2]

    nash_variables: nash_lib.NashVariables
    nash_variables, stats = snt.BatchApply(solve_nash)(q_values)

    # Convert [T, B, S] -> [S, T, B]
    return nash_variables._replace(
        p1=tf.transpose(nash_variables.p1, [2, 0, 1]),
        p2=tf.transpose(nash_variables.p2, [2, 0, 1]),
    )

  def _train_q_policy(
      self,
      frames: Frames,
      # batch: Batch,
      initial_states: RecurrentState,
      policy_samples: embed.Action,  # [S, T, 2B]
      sample_policy_logits: ControllerType,  # [T, 2B] (only on action taken)
      sample_q_values: tf.Tensor,  # [2, S, S, T, B]
      nash_policy: nash_lib.NashVariables,  # [S, T, B]
      # q_function_outputs: QFunctionOutputs,
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

    frames = self._get_delayed_frames(frames)

    action = frames.state_action.action
    prev_action = tf.nest.map_structure(lambda t: t[:-1], action)

    num_samples = self.num_samples + 1
    assert nash_policy.p1.shape[0] == num_samples
    assert nash_policy.p2.shape[0] == num_samples
    replicate_samples = lambda nest: tf.nest.map_structure(
        replicate(num_samples), nest)

    # Train the q_policy by regressing to the compute nash policy
    with tf.GradientTape() as tape:
      q_policy_outputs = self.q_policy.unroll_with_outputs(
          frames, initial_states)

      # Find the Nash policy over the action subsample and regress the
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

      # The action taken is the last one along the first (sample) axis.
      # Note that this measures the distance to the actual replay data,
      # not to the sample policy which only estimates the data distribution.
      q_policy_imitation_loss = -q_policy_log_probs[-1]

      # Normalize log-probs for the finite sample
      q_policy_log_probs -= tf.math.reduce_logsumexp(
          q_policy_log_probs, axis=0, keepdims=True)

      # nash_distribution: [S, T, 2B]
      nash_distribution = tf.concat([nash_policy.p1, nash_policy.p2], axis=2)
      q_policy_cross_entropy = -tf.reduce_sum(
          nash_distribution * q_policy_log_probs, axis=0)
      nash_entropy = entropy(nash_distribution, axis=0)
      q_policy_nash_kl = q_policy_cross_entropy - nash_entropy

      sample_q_values = (sample_q_values[0] - sample_q_values[1]) / 2
      payoff_matrices = tf.transpose(sample_q_values, [2, 3, 0, 1])  # [T, B, S, S]
      p1_nash_policy = tf.transpose(nash_policy.p1, [1, 2, 0])  # [T, B, S]
      p2_nash_policy = tf.transpose(nash_policy.p2, [1, 2, 0])  # [T, B, S]

      # Compute payoffs against the Nash policy.
      p1_vs_nash_payoffs = tf.linalg.matvec(
          payoff_matrices, p2_nash_policy)  # [T, B, S]
      p2_vs_nash_payoffs = -tf.linalg.matvec(
          payoff_matrices, p1_nash_policy, transpose_a=True)  # [T, B, S]
      vs_nash_payoffs = tf.concat(
          [p1_vs_nash_payoffs, p2_vs_nash_payoffs], axis=1)  # [T, 2B, S]

      q_policy_probs = tf.exp(q_policy_log_probs)  # [S, T, 2B]
      q_policy_probs = tf.transpose(q_policy_probs, [1, 2, 0])  # [T, 2B, S]
      q_policy_vs_nash = tf.reduce_sum(vs_nash_payoffs * q_policy_probs, axis=2)

      nash_values = tf.concat([
          nash_policy.p1_nash_value,
          -nash_policy.p1_nash_value,
      ], axis=1)  # [T, 2B]

      regret = q_policy_vs_nash - nash_values

      losses = [
          q_policy_cross_entropy,
          self.q_policy_imitation_weight * q_policy_imitation_loss,
      ]
      q_policy_total_loss = tf.add_n(losses)

    action_taken_nash_prob = nash_distribution[-1]

    # Estimate the entropy of the q_policy.
    q_policy_samples = self.q_policy.controller_head.sample(
        inputs=q_policy_outputs.outputs,
        prev_controller_state=prev_action,
    )
    q_policy_entropy = self._compute_entropy(
        self._get_distribution(q_policy_samples.logits))

    # Compute KL to sample policy (on action taken)
    q_policy_logits = tf.nest.map_structure(
        lambda t: t[-1], q_policy_distances.logits)
    q_policy_sample_kl = self._compute_kl(
        self._get_distribution(sample_policy_logits),
        self._get_distribution(q_policy_logits),
    )

    q_policy_metrics = dict(
        nash_entropy=nash_entropy,
        nash_loss=q_policy_cross_entropy,
        nash_kl=q_policy_nash_kl,
        imitation_loss=q_policy_imitation_loss,
        regret=regret,
        action_taken_nash_prob=action_taken_nash_prob,
        entropy=q_policy_entropy,
        sample_policy_kl=q_policy_sample_kl,
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
    return tf.nest.map_structure(
        lambda x, y: tf_utils.where(restarting, x, y),
        self.initial_state(batch_size),
        initial_states)

  def step(
      self,
      batch: nash_data.TwoPlayerBatch,  # [B, T]
      initial_states: RecurrentState,
      train: bool = True,
      compile: bool = True,
  ) -> tuple[dict, RecurrentState]:
    del compile  # TODO: use this

    # merged_frames: [2B, T]
    merged_frames = tf.nest.map_structure(
        lambda *xs: np.concatenate(xs, axis=0),
        batch.p0_frames, batch.p1_frames)

    # switch axes to time-major: [T, 2B]
    tm_frames: Frames = tf.nest.map_structure(
        lambda a: np.swapaxes(a, 0, 1), merged_frames)

    # Here we assume that the sample policy, q function, and q policy all have
    # the same state and action embeddings.
    tm_frames = tm_frames._replace(
        state_action=self.sample_policy.embed_state_action.from_state(
            tm_frames.state_action))

    # Put on device memory once.
    tm_frames: Frames = tf.nest.map_structure(tf.convert_to_tensor, tm_frames)

    # tm_batch = batch._replace(frames=tm_frames)
    # tm_batch: Batch = tf.nest.map_structure(tf.convert_to_tensor, tm_batch)

    # Ideally this should be pushed into the compiled train_* functions, but
    # very strangely if we _don't_ do the reset here it results in an OOM in
    # train_q_function. Hopefully we don't take much of a performance hit
    # from this? (Compiling this also results in the same OOM.)
    initial_states = self.reset_initial_states(
        initial_states, batch.needs_reset[:, 0])

    # inital_states: [2B, ...]
    initial_states = {
        key: nash_data.merge(initial_state, axis=0)
        for key, initial_state in initial_states.items()
    }

    # TODO: take into account delay
    final_states = initial_states  # GC initial states as they are replaced
    metrics = {}

    (
      policy_samples,  # [S, T, 2B]
      final_states['sample_policy'],
      metrics['sample_policy'],
      sample_policy_logits,  # [T, 2B]
    ) = self.train_sample_policy(
        tm_frames, initial_states['sample_policy'],
        train=train and self.should_train_sample_policy)

    (
      q_function_outputs,  # [2, S, S, T, B]
      final_states['q_function'],
      metrics['q_function'],
    ) = self.train_q_function(
        tm_frames, initial_states['q_function'], policy_samples, train)

    nash_policy = self._compute_nash(q_function_outputs.sample_q_values)

    # Only start training the q_policy when the q function is decent.
    # Empirically the q function is worse than the value function at the start,
    # likely because the targets are bootstrapped from the value function.
    train_q_policy = train
    if train_q_policy and self.min_uev_delta is not None:
      uev_delta = metrics['q_function']['q']['uev_delta'].numpy().mean()
      if uev_delta < self.min_uev_delta:
        train_q_policy = False

    (
      final_states['q_policy'],
      metrics['q_policy'],
    ) = self.train_q_policy(
        tm_frames,
        initial_states=initial_states['q_policy'],
        policy_samples=policy_samples,
        sample_policy_logits=sample_policy_logits,
        sample_q_values=q_function_outputs.sample_q_values,
        nash_policy=nash_policy,
        train=train_q_policy,
    )

    final_states = {
        key: nash_data.split(final_state, axis=0)
        for key, final_state in final_states.items()
    }

    # satisfy train_q_lib._get_loss
    metrics['total_loss'] = metrics['q_policy']['nash_kl']

    return metrics, final_states
