"""Mixture learner that approximates "fictitious play".

Conceptually we alternate between two steps:
1. Train the exploiter policy against the old mixture policy.
2. Train the new mixture policy to imitate a mixture of the old mixture policy
  and the new exploiter policy.

In practice, we do both of these at the same time.
"""

import dataclasses
import typing as tp

import sonnet as snt
import tensorflow as tf

from slippi_ai.policies import Policy, UnrollOutputs
from slippi_ai.evaluators import Trajectory
from slippi_ai.networks import RecurrentState
from slippi_ai import value_function as vf_lib
from slippi_ai import tf_utils, utils
from slippi_ai.rl import learner

@dataclasses.dataclass
class LearnerConfig(learner.LearnerConfig):
  exploiter_weight: float = 0.1

class MixtureState(tp.NamedTuple):
  exploiter: RecurrentState  # Mixture state on exploiter trajectory
  mixture: RecurrentState  # Mixture state on old-mixture trajectory

class LearnerState(tp.NamedTuple):
  # We get the exploiter state from the actor in the Trajectory.
  # The rest only live on the learner and we keep them here.
  mixture: MixtureState
  rl: learner.LearnerState  # Teacher/VF state on exploiter trajectory

class LearnerOutputs(tp.NamedTuple):
  rl: learner.LearnerOutputs  # unused
  stats: dict

class Learner:
  """Implements A2C."""

  def __init__(
      self,
      config: LearnerConfig,
      exploiter_policy: Policy,
      mixture_policy: Policy,
      teacher: Policy,
      value_function: tp.Optional[vf_lib.ValueFunction] = None,
  ) -> None:
    self._learner = learner.Learner(
        config=config,
        policy=exploiter_policy,
        teacher=teacher,
        value_function=value_function,
    )
    self._config = config
    self.mixture_policy = mixture_policy
    self.mixture_optimizer = snt.optimizers.Adam(config.learning_rate)

    if config.compile:
      maybe_compile = tf.function(
          jit_compile=config.jit_compile, autograph=False)
    else:
      maybe_compile = lambda f: f

    self.compiled_unroll_mixture = maybe_compile(self.unroll_mixture)
    self.compiled_unroll = maybe_compile(self.unroll)

  def exploiter_variables(self):
    return self._learner.policy_variables()

  def mixture_variables(self):
    return self.mixture_policy.variables

  def initial_mixture_state(self, batch_size: int) -> MixtureState:
    return MixtureState(
        exploiter=self.mixture_policy.initial_state(batch_size),
        mixture=self.mixture_policy.initial_state(batch_size),
    )

  def initial_state(self, batch_size: int) -> LearnerState:
    return LearnerState(
        mixture=self.initial_mixture_state(batch_size),
        rl=self._learner.initial_state(batch_size),
    )

  def _unroll_mixture_policy(
      self,
      trajectory: Trajectory,
      initial_state: RecurrentState,
  ) -> tp.Tuple[UnrollOutputs, tf.Tensor]:
    # Currently, resetting states can only occur on the first frame, which
    # conveniently means we don't have to deal with resets inside `unroll`.
    is_resetting = trajectory.is_resetting[0]  # [B]
    batch_size = is_resetting.shape[0]
    initial_state = tf.nest.map_structure(
        lambda x, y: tf_utils.where(is_resetting, x, y),
        self.mixture_policy.initial_state(batch_size), initial_state)

    outputs = self.mixture_policy.unroll(
        frames=learner.get_delayed_frames(trajectory),
        initial_state=initial_state,
        discount=self._learner.discount,
    )

    get_distribution = self._learner._get_distribution

    # For the actor, also drop the first action which precedes the first frame.
    actor_sample_outputs = learner.get_delayed_sample_outputs(trajectory)
    actor_logits = tf.nest.map_structure(
        lambda t: t[1:], actor_sample_outputs.logits)
    actor_distribution = get_distribution(actor_logits)

    mixture_distribution = get_distribution(outputs.distances.logits)
    kl = self._learner._compute_kl(actor_distribution, mixture_distribution)

    return outputs, kl

  def unroll_mixture(
      self,
      exploiter_trajectory: Trajectory,
      mixture_trajectory: Trajectory,
      initial_state: MixtureState,
      train_mixture_policy: bool = False,
      exploiter_weight: tp.Optional[float] = None,
  ) -> tp.Tuple[MixtureState, dict]:

    with tf.GradientTape() as tape:
      if exploiter_weight is None:
        exploiter_weight = self._config.exploiter_weight

      # We could batch these together.
      mixture_exploiter_outputs, exploiter_kl = self._unroll_mixture_policy(
          exploiter_trajectory, initial_state.exploiter)

      mixture_mixture_outputs, mixture_kl = self._unroll_mixture_policy(
          mixture_trajectory, initial_state.mixture)

      mixture_loss = (
          exploiter_kl * exploiter_weight
          + mixture_kl * (1 - exploiter_weight))

      if train_mixture_policy:
        mixture_params = self.mixture_policy.trainable_variables
        loss = tf.reduce_mean(mixture_loss)
        grads = tape.gradient(loss, mixture_params)
        self.mixture_optimizer.apply(grads, mixture_params)

    final_state = MixtureState(
        exploiter=mixture_exploiter_outputs.final_state,
        mixture=mixture_mixture_outputs.final_state,
    )

    stats = dict(
        exploiter_kl=exploiter_kl,
        mixture_kl=mixture_kl,
        mixture_loss=mixture_loss,
    )

    return final_state, stats

  def unroll(
      self,
      exploiter_trajectory: Trajectory,
      mixture_trajectory: Trajectory,
      initial_state: LearnerState,
      train_mixture_policy: bool = False,
      exploiter_weight: tp.Optional[float] = None,
      train_value_function: bool = False,
  ) -> tp.Tuple[LearnerOutputs, LearnerState]:
    stats = {}

    rl_outputs, rl_final_state = self._learner.unroll(
        trajectory=exploiter_trajectory,
        initial_state=initial_state.rl,
        train_value_function=train_value_function)

    mixture_final_state, stats['mixture'] = self.unroll_mixture(
        exploiter_trajectory=exploiter_trajectory,
        mixture_trajectory=mixture_trajectory,
        initial_state=initial_state.mixture,
        train_mixture_policy=train_mixture_policy,
        exploiter_weight=exploiter_weight)

    final_state = LearnerState(
        mixture=mixture_final_state,
        rl=rl_final_state,
    )

    outputs = LearnerOutputs(
        rl=rl_outputs,
        stats=stats,
    )

    return outputs, final_state

  def step(
      self,
      exploiter_trajectories: tp.Sequence[Trajectory],
      mixture_trajectories: tp.Sequence[Trajectory],
      initial_state: LearnerState,
      num_ppo_epochs: int = None,
      exploiter_weight: tp.Optional[float] = None,
  ) -> tp.Tuple[LearnerState, dict]:
    mixture_stats = []
    mixture_hidden_state = initial_state.mixture
    for exploiter_trajectory, mixture_trajectory in zip(
        exploiter_trajectories, mixture_trajectories):
      mixture_hidden_state, stats = self.compiled_unroll_mixture(
          exploiter_trajectory=exploiter_trajectory,
          mixture_trajectory=mixture_trajectory,
          initial_state=mixture_hidden_state,
          train_mixture_policy=True,
          exploiter_weight=exploiter_weight,
      )
      mixture_stats.append(stats)
    mixture_stats = utils.map_single_structure(
        lambda t: t.numpy(), mixture_stats)
    mixture_stats = utils.batch_nest(mixture_stats)

    rl_final_state, rl_stats = self._learner.ppo(
        trajectories=exploiter_trajectories,
        initial_state=initial_state.rl,
        num_epochs=num_ppo_epochs)

    stats = dict(
        mixture=mixture_stats,
        rl=rl_stats,
    )

    final_state = LearnerState(
        mixture=mixture_hidden_state,
        rl=rl_final_state,
    )

    return final_state, stats

  def initialize(self, trajectory: Trajectory):
    """Initialize model and optimizer variables."""
    self._learner.initialize(trajectory)

    # Initialize mixture policy.
    frames = learner.get_frames(trajectory)
    self.mixture_policy.unroll(frames, trajectory.initial_state)
    self._mixture_vars = self.mixture_policy.variables
    self.mixture_optimizer._initialize(self._mixture_vars)

  def get_vars(self) -> dict:
    return dict(
        # The main policy is the mixture policy.
        policy=self.mixture_policy.variables,
        mixture_optimizer=self.mixture_optimizer.variables,
        rl=self._learner.get_vars(),
    )

  def restore_from_imitation(self, imitation_state: dict):
    # The rl-learner restores the exploiter_policy and value_function.
    self._learner.restore_from_imitation(imitation_state)

    vars = dict(
        policy=self.mixture_policy.variables,
        optimizer=self.mixture_optimizer.variables,
    )
    to_restore = dict(
        policy=imitation_state['policy'],
        optimizer=imitation_state['optimizers']['policy'],
    )
    tf.nest.map_structure(
        lambda var, val: var.assign(val),
        vars, to_restore)

  def reset_exploiter_policy(self):
    tf.nest.map_structure(
        lambda var, val: var.assign(val),
        self.exploiter_variables(), self.mixture_variables())
