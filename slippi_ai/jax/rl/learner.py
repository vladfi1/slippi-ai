"""JAX RL learner implementing PPO."""

import dataclasses
import functools
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from slippi_ai import data, reward as reward_lib, utils
from slippi_ai.evaluators import Trajectory
from slippi_ai.jax import jax_utils, embed
from slippi_ai.jax import value_function as vf_lib
from slippi_ai.jax.policies import Policy, UnrollOutputs
from slippi_ai.jax.networks import RecurrentState

Array = jax.Array
field = lambda f: dataclasses.field(default_factory=f)


@dataclasses.dataclass
class PPOConfig:
  num_epochs: int = 1
  num_batches: int = 1
  epsilon: float = 1e-2
  beta: float = 0
  max_mean_actor_kl: float = 1e-4


@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  policy_gradient_weight: float = 1
  kl_teacher_weight: float = 1e-1
  reverse_kl_teacher_weight: float = 0
  entropy_weight: float = 0
  value_cost: float = 0.5
  reward_halflife: float = 4  # measured in seconds
  reward: reward_lib.RewardConfig = field(reward_lib.RewardConfig)
  ppo: PPOConfig = field(PPOConfig)

  optimizer_burnin_steps: int = 0
  value_burnin_steps: int = 0

class LearnerState(tp.NamedTuple):
  teacher: RecurrentState
  value_function: RecurrentState


class LearnerOutputs(tp.NamedTuple):
  teacher: UnrollOutputs
  value: vf_lib.ValueOutputs


def get_frames(trajectory: Trajectory) -> data.Frames:
  """Gives time-major frames with actions taken."""
  state_action = data.StateAction(
      state=trajectory.states,
      action=trajectory.actions.controller_state,
      name=trajectory.name,
  )
  return data.Frames(state_action, trajectory.is_resetting, trajectory.rewards)


def get_delayed_frames(trajectory: Trajectory) -> data.Frames:
  """Gives time-major frames with delayed actions, for teacher/policy unroll."""
  delay = len(trajectory.delayed_actions)

  if delay == 0:
    return get_frames(trajectory)

  # Extract controller states from delayed actions, each is [B, ...]
  delayed_cs = [sa.controller_state for sa in trajectory.delayed_actions]

  # Add time dimension: [B, ...] -> [1, B, ...]
  delayed_cs_with_time = [
      jax.tree.map(lambda t: t[np.newaxis], cs) for cs in delayed_cs
  ]

  # Concatenate: [T+1, B, ...] + D * [1, B, ...] -> [T+1+D, B, ...]
  # Then take [delay:] to align -> [T+1, B, ...]
  actions = jax.tree.map(
      lambda *ts: jnp.concatenate(ts, axis=0),
      trajectory.actions.controller_state,
      *delayed_cs_with_time,
  )
  actions = jax.tree.map(lambda t: t[delay:], actions)

  state_action = data.StateAction(
      state=trajectory.states,
      action=actions,
      name=trajectory.name,
  )
  return data.Frames(state_action, trajectory.is_resetting, trajectory.rewards)


def update_rewards(
    trajectory: Trajectory,
    reward_config: reward_lib.RewardConfig,
) -> Trajectory:
  rewards = reward_lib.compute_rewards(
      trajectory.states, **dataclasses.asdict(reward_config))
  rewards = np.where(trajectory.is_resetting[1:], 0.0, rewards)
  return trajectory._replace(rewards=rewards)

def warmup_schedule(burnin_steps: int, base_value: float):
  burnin = optax.constant_schedule(0)
  normal = optax.constant_schedule(base_value)
  return optax.join_schedules([burnin, normal], [burnin_steps])


@nnx.jit(
    donate_argnums=(0, 1, 2, 4),
    static_argnames=['train_value_function'],
)
def _unroll_teacher_and_vf(
    teacher: Policy,
    value_function: vf_lib.ValueFunction,
    value_optimizer: nnx.Optimizer[vf_lib.ValueFunction],
    trajectory: Trajectory,
    initial_state: LearnerState,
    discount: float,
    *,
    train_value_function: bool,
):
  teacher_frames = get_delayed_frames(trajectory)
  teacher_outputs = teacher.unroll(teacher_frames, initial_state.teacher)

  # Run value function (with or without gradient update).
  value_frames = get_frames(trajectory)

  if train_value_function:
    def value_loss_fn(vf: vf_lib.ValueFunction):
      outputs, final_state = vf.loss(
          value_frames, initial_state.value_function,
          discount)
      return jnp.mean(outputs.loss), (outputs, final_state)

    grads, (value_outputs, value_final_state) = jax_utils.grad_with_aux(
        value_loss_fn)(value_function)
    value_optimizer.update(value_function, grads)
  else:
    value_outputs, value_final_state = value_function.loss(
        value_frames, initial_state.value_function, discount)

  final_state = LearnerState(
      teacher=teacher_outputs.final_state,
      value_function=value_final_state,
  )
  outputs = LearnerOutputs(
      teacher=teacher_outputs,
      value=value_outputs,
  )
  return outputs, final_state

class Learner(nnx.Module):
  """Implements PPO for RL fine-tuning."""

  def __init__(
      self,
      config: LearnerConfig,
      policy: Policy,
      teacher: Policy,
      value_function: vf_lib.ValueFunction,
  ) -> None:
    self._config = config
    self.policy = policy
    self.teacher = teacher
    self.value_function = value_function

    self._controller_embedding = policy.controller_head.controller_embedding

    value_lr_zero_steps = config.optimizer_burnin_steps
    policy_lr_zero_steps = config.optimizer_burnin_steps + config.optimizer_burnin_steps

    def schedule(steps: int):
      # Each epoch the optimizer will be called num_batches times
      return warmup_schedule(steps * config.ppo.num_batches, config.learning_rate)

    self.policy_optimizer = nnx.Optimizer(
        policy,
        optax.adam(schedule(policy_lr_zero_steps)),
        wrt=nnx.Param)
    self.value_optimizer = nnx.Optimizer(
        self.value_function,
        optax.adam(schedule(value_lr_zero_steps)),
        wrt=nnx.Param)

    self.discount = 0.5 ** (1 / (config.reward_halflife * 60))

    jit_unroll = nnx.jit(
        Learner._unroll_teacher_and_vf,
        donate_argnums=(0, 2),
        static_argnames=['train_value_function'],
    )
    self.unroll = nnx.cached_partial(jit_unroll, self)

    jit_ppo_epoch = nnx.jit(
        Learner.ppo_epoch,
        donate_argnums=0,
        static_argnames=['train'])
    self.jit_ppo_epoch = nnx.cached_partial(jit_ppo_epoch, self)

  def initial_state(
      self, batch_size: int, rngs: tp.Optional[nnx.Rngs] = None,
  ) -> LearnerState:
    if rngs is None:
      rngs = nnx.Rngs(0)
    return LearnerState(
        teacher=self.teacher.initial_state(batch_size, rngs),
        value_function=self.value_function.initial_state(batch_size, rngs),
    )

  def policy_variables(self):
    """Returns policy state for actor update via evaluators.update_variables."""
    return self.policy.get_state()

  def _sum_leaves(self, embedding: embed.Embedding, struct) -> Array:
    return functools.reduce(jnp.add, embedding.flatten(struct))

  def _compute_kl(self, logits_p, logits_q) -> Array:
    """Computes total KL(P||Q) summed over all controller components."""
    kls = self._controller_embedding.map(
        lambda e, lp, lq: e.kl_divergence(lp, lq),
        logits_p, logits_q)
    return self._sum_leaves(self._controller_embedding, kls)

  def _compute_entropy(self, logits) -> Array:
    """Computes total entropy H(P) summed over all controller components."""
    entropies = self._controller_embedding.map(
        lambda e, l: e.entropy(l), logits)
    return self._sum_leaves(self._controller_embedding, entropies)

  def _get_log_prob(self, logits, action) -> Array:
    """Computes log P(action | logits) summed over all controller components."""
    distances = self._controller_embedding.map(
        lambda e, l, a: e.distance(l, a), logits, action)
    return -self._sum_leaves(self._controller_embedding, distances)

  def _unroll_teacher_and_vf(
      self,
      trajectory: Trajectory,
      initial_state: LearnerState,
      *,
      train_value_function: bool = False,
  ):
    teacher_frames = get_delayed_frames(trajectory)
    teacher_outputs = self.teacher.unroll(teacher_frames, initial_state.teacher)

    # Run value function (with or without gradient update).
    value_frames = get_frames(trajectory)

    if train_value_function:
      def value_loss_fn(vf: vf_lib.ValueFunction):
        outputs, final_state = vf.loss(
            value_frames, initial_state.value_function, self.discount)
        return jnp.mean(outputs.loss), (outputs, final_state)

      grads, (value_outputs, value_final_state) = jax_utils.grad_with_aux(
          value_loss_fn)(self.value_function)
      self.value_optimizer.update(self.value_function, grads)
    else:
      value_outputs, value_final_state = self.value_function.loss(
          value_frames, initial_state.value_function, self.discount)

    final_state = LearnerState(
        teacher=teacher_outputs.final_state,
        value_function=value_final_state,
    )
    outputs = LearnerOutputs(
        teacher=teacher_outputs,
        value=value_outputs,
    )
    return outputs, final_state

  def ppo_grads(
      self,
      outputs: LearnerOutputs,
      trajectory: Trajectory,
  ) -> tp.Tuple[tp.Any, dict]:
    """Computes policy gradients for one PPO step.

    Value function outputs are for [0, U] while policy outputs are for
    [D, U+D]. This means we can only train on steps [D, U].

    Args:
      outputs: Pre-computed teacher and value function outputs.
      trajectory: The collected trajectory.

    Returns:
      Tuple of (gradients, metrics dict).
    """
    delay = self.policy.delay  # D
    remove_first = lambda t: t[delay:] if delay > 0 else t
    remove_last = lambda t: t[:t.shape[0] - delay] if delay > 0 else t

    # Advantages from [0, U]: take [D, U] -> U-D steps.
    advantages = jax.lax.stop_gradient(outputs.value.advantages[delay:])

    # Teacher logits: [D, U+D] -> truncate last D -> [D, U].
    # Note: no stop_gradient needed since teacher has no trainable variables.
    teacher_logits = jax.tree.map(remove_last, outputs.teacher.distances.logits)

    # Policy frames: states [0, U-D+1], actions [D, U+1].
    policy_frames = data.Frames(
        state_action=data.StateAction(
            state=jax.tree.map(remove_last, trajectory.states),
            action=jax.tree.map(remove_first, trajectory.actions.controller_state),
            name=remove_last(trajectory.name),
        ),
        is_resetting=remove_last(trajectory.is_resetting),
        reward=remove_first(trajectory.rewards),
    )

    # Actor (old policy) logits and log probs for steps [D+1, U+1].
    actor_outputs = utils.map_single_structure(
        lambda t: t[1 + delay:], trajectory.actions)
    actor_logits = actor_outputs.logits
    actor_log_probs = jax.lax.stop_gradient(
        self._get_log_prob(actor_logits, actor_outputs.controller_state))

    def policy_loss_fn(policy: Policy):
      policy_outputs = policy.unroll(policy_frames, trajectory.initial_state)
      new_logits = policy_outputs.distances.logits
      new_log_probs = policy_outputs.log_probs

      # KL divergences (computed over full output distribution, not just sampled action).
      # Forward KL to teacher: incentivizes refining human actions over covering all of them.
      teacher_kl = self._compute_kl(new_logits, teacher_logits)
      # KL of old actor from new policy: used for monitoring / reverting bad updates.
      actor_kl = self._compute_kl(actor_logits, new_logits)
      reverse_teacher_kl = self._compute_kl(teacher_logits, new_logits)
      entropy = self._compute_entropy(new_logits)

      # PPO clipped objective.
      log_rhos = new_log_probs - actor_log_probs
      rhos = jnp.exp(log_rhos)

      eps = self._config.ppo.epsilon
      clipped_log_rhos = jnp.clip(log_rhos, -eps, eps)
      clipped_rhos = jnp.exp(clipped_log_rhos)

      ppo_objective = jnp.minimum(rhos * advantages, clipped_rhos * advantages)

      loss = jnp.mean(
          - self._config.policy_gradient_weight * ppo_objective
          + self._config.ppo.beta * actor_kl
          + self._config.kl_teacher_weight * teacher_kl
          + self._config.reverse_kl_teacher_weight * reverse_teacher_kl
          - self._config.entropy_weight * entropy
      )

      metrics = dict(
          total_loss=loss,
          ppo_objective=ppo_objective,
          teacher_kl=teacher_kl,
          entropy=entropy,
          actor_kl=actor_kl,
          reverse_teacher_kl=reverse_teacher_kl,
      )
      return loss, metrics

    grads, metrics = jax_utils.grad_with_aux(policy_loss_fn)(self.policy)
    return grads, metrics

  def ppo_epoch(
      self,
      learner_outputs: list[LearnerOutputs],
      trajectories: list[Trajectory],
      train: bool = True,
  ) -> dict:
    """One epoch of PPO: accumulate gradients over all trajectories."""
    grad_shapes, _ = nnx.eval_shape(
        Learner.ppo_grads, self, learner_outputs[0], trajectories[0])
    grads_acc = jax.tree.map(jnp.zeros_like, grad_shapes)

    batched_learner_outputs = jax.tree.map(
      lambda *xs: jnp.stack(xs), *learner_outputs)
    batched_trajectories = jax.tree.map(
      lambda *xs: jnp.stack(xs), *trajectories)

    @nnx.scan(
        in_axes=(None, 0, 0, nnx.Carry),
        out_axes=(0, nnx.Carry),
    )
    def scan_fn(
        learner: Learner,
        learner_outputs: LearnerOutputs,
        trajectory: Trajectory,
        grads_acc: jax_utils.Grads,
    ) -> tuple[dict, jax_utils.Grads]:
      grads, metrics = learner.ppo_grads(learner_outputs, trajectory)
      new_grads_acc = jax.tree.map(jnp.add, grads_acc, grads)
      return metrics, new_grads_acc

    metrics, grads_acc = scan_fn(
        self, batched_learner_outputs, batched_trajectories, grads_acc)

    if train:
      n = len(trajectories)
      assert n > 0
      grads_acc = jax.tree.map(lambda g: g / n, grads_acc)
      self.policy_optimizer.update(self.policy, grads_acc)

    actor_kl = metrics['actor_kl']
    metrics['actor_kl'] = dict(
        mean=jnp.mean(actor_kl),
        max=jnp.max(actor_kl),
    )
    return metrics

  def ppo(
      self,
      trajectories: list[Trajectory],
      initial_state: LearnerState,
      num_epochs: tp.Optional[int] = None,
  ) -> tp.Tuple[LearnerState, dict]:
    """Multi-epoch PPO update.

    Args:
      trajectories: List of trajectories (one per PPO batch).
      initial_state: Initial recurrent states for teacher and value function.
      num_epochs: Number of PPO epochs. Defaults to config value.

    Returns:
      Tuple of (new hidden state, metrics dict).
    """
    # Compute rewards from game states.
    trajectories = [
        update_rewards(t, self._config.reward) for t in trajectories
    ]

    # Checkpoint policy state for potential reverting.
    # checkpoint = jax_utils.get_module_state(self.policy)

    # Unroll teacher + value function, training value function.
    learner_outputs = []
    hidden_state = initial_state
    for trajectory in trajectories:
      outputs, hidden_state = self.unroll(
          trajectory, hidden_state, train_value_function=True)
      learner_outputs.append(outputs)

    # Collect value function metrics.
    value_metrics_list = [o.value.metrics for o in learner_outputs]
    value_metrics = utils.batch_nest(
        utils.map_single_structure(np.asarray, value_metrics_list))
    value_metrics = utils.map_single_structure(np.mean, value_metrics)

    if num_epochs is None:
      num_epochs = self._config.ppo.num_epochs

    # PPO epochs with gradient updates.
    per_epoch_metrics = []
    for _ in range(num_epochs):
      epoch_metrics = self.jit_ppo_epoch(learner_outputs, trajectories, train=True)
      per_epoch_metrics.append(epoch_metrics)

    # Final eval epoch (no gradient update) to measure post-update KL.
    final_metrics = self.jit_ppo_epoch(learner_outputs, trajectories, train=False)
    per_epoch_metrics.append(final_metrics)

    # Revert if the policy moved too far from the actor.
    reverted = False
    if final_metrics['actor_kl']['mean'] > self._config.ppo.max_mean_actor_kl:
      # jax_utils.set_module_state(self.policy, checkpoint)
      # reverted = True
      raise ValueError(f"Mean actor KL after PPO update is too high: {final_metrics['actor_kl']['mean']}")

    metrics = dict(
        ppo_step={str(i): m for i, m in enumerate(per_epoch_metrics)},
        post_update=final_metrics,
        value=value_metrics,
        reverted=reverted,
    )

    return hidden_state, metrics

  def get_state(self) -> dict:
    return jax_utils.get_module_state(self)

  def restore_from_imitation(self, state_dict: dict):
    # Assumes policy/value function + optimizers have the same keys as in
    # imitation checkpoints.
    jax_utils.set_module_state(self, state_dict)
