"""JAX RL learner implementing PPO."""

import dataclasses
import functools
import logging
import time
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from slippi_ai import data, reward as reward_lib, utils
from slippi_ai.evaluators import Trajectory
from slippi_ai.jax import jax_utils, embed, rl_lib
from slippi_ai.jax import value_function as vf_lib
from slippi_ai.jax.policies import Policy, UnrollOutputs, ControllerType
from slippi_ai.jax.networks import RecurrentState
from slippi_ai.jax.agents import DType

Array = jax.Array
field = lambda f: dataclasses.field(default_factory=f)


@dataclasses.dataclass
class PPOConfig:
  num_epochs: int = 1
  num_batches: int = 1
  epsilon: float = 1e-2
  beta: float = 0
  max_mean_actor_kl: float = 1e-4

_DEFAULT_FP16_POLICY_LOSS_SCALE = 1024

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  policy_gradient_weight: float = 1
  kl_teacher_weight: float = 1e-1
  reverse_kl_teacher_weight: float = 0
  entropy_weight: float = 0
  value_cost: float = 0.5
  reward_halflife: float = 4  # measured in seconds
  # TD-lambda for the value function's training targets.
  gae_lambda: float = 0
  # TD-lambda for the advantages used by PPO; None falls back to gae_lambda.
  # The default of 1 gives unbiased advantages that are robust to a poorly
  # fit value function, while gae_lambda=0 keeps the value targets themselves
  # low-variance.
  advantage_lambda: tp.Optional[float] = 1.0
  reward: reward_lib.RewardConfig = field(reward_lib.RewardConfig.default)
  ppo: PPOConfig = field(PPOConfig)
  fused: bool = False

  optimizer_burnin_epochs: int = 1
  value_burnin_epochs: int = 1

  # 0 means no microbatching
  microbatch_size: int = 0
  teacher_mbs: int = 0
  value_mbs: int = 0

  teacher_dtype: DType = DType.FP32
  value_dtype: DType = DType.FP32
  policy_dtype: DType = DType.FP32

  policy_loss_scale: tp.Optional[float] = None
  value_loss_scale: tp.Optional[float] = None

  def __post_init__(self):
    if self.policy_dtype is DType.FP16 and self.policy_loss_scale is None:
      self.policy_loss_scale = _DEFAULT_FP16_POLICY_LOSS_SCALE
      logging.warning(f'Set default policy loss scale for FP16 to {self.policy_loss_scale} to stabilize training.')

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
      rating=trajectory.rating,
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
      rating=trajectory.rating,
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

class MBKwargs(tp.TypedDict):
  microbatch_size: int
  input_batch_dims: int | tuple
  output_batch_dims: int | tuple

class PPOInputs(tp.NamedTuple, tp.Generic[ControllerType]):
  teacher_logits: ControllerType
  advantages: jax.Array
  trajectory: Trajectory

Metrics = dict[str, tp.Any]
Advantage = jax.Array

# batch axes
_STATE_AXIS = 0
_METRICS_AXIS = 1
_TEACHER_LOGITS_AXIS = 1
_ADVANTAGES_AXIS = 1
_TRAJECTORY_AXES = Trajectory.batch_dims()

class Learner(nnx.Module, tp.Generic[ControllerType]):
  """Implements PPO for RL fine-tuning."""

  def __init__(
      self,
      config: LearnerConfig,
      policy: Policy[ControllerType],
      teacher: Policy[ControllerType],
      value_function: vf_lib.ValueFunction,
      device: tp.Optional[jax.Device] = None,
  ) -> None:
    self._config = config
    self._device = device
    self.policy = policy
    self.teacher = teacher
    self.value_function = value_function

    self._controller_embedding = policy.controller_head.controller_embedding

    # The policy update is accumulated across the trajectories in one epoch.
    # For the first value_burnin_epochs we don't train the policy at all.
    # Then for the next optimizer_burnin_epochs we train the policy for a
    # single epoch with learning_rate=0 to initialize the optimizer state.

    self.policy_schedule = warmup_schedule(
        config.optimizer_burnin_epochs,
        config.learning_rate,
    )

    self.policy_optimizer = nnx.Optimizer(
        policy,
        optax.adam(self.policy_schedule),
        wrt=nnx.Param)
    self.value_optimizer = nnx.Optimizer(
        self.value_function,
        optax.adam(config.learning_rate),
        wrt=nnx.Param)

    self.discount = rl_lib.discount_from_halflife(config.reward_halflife)

    mbs = self._config.microbatch_size

    teacher_mbkwargs = MBKwargs(
        microbatch_size=self._config.teacher_mbs,
        input_batch_dims=(_TRAJECTORY_AXES, _STATE_AXIS),
        output_batch_dims=(_TEACHER_LOGITS_AXIS, _STATE_AXIS),
    )
    jax_utils.cast_module_state_to_dtype(self.teacher, config.teacher_dtype.dtype)
    self._unroll_teacher_mb = jax_utils.microbatch_fn(
        jax_utils.no_loss(self._unroll_teacher),
        **teacher_mbkwargs
    )
    self.unroll_teacher = jax_utils.run_loss_fn(
        self.teacher, self._unroll_teacher,
        **teacher_mbkwargs,
    )

    value_mbkwargs = MBKwargs(
        microbatch_size=self._config.value_mbs,
        input_batch_dims=(_TRAJECTORY_AXES, _STATE_AXIS),
        output_batch_dims=(_METRICS_AXIS, _STATE_AXIS, _ADVANTAGES_AXIS),
    )
    # Train with gae_lambda targets while computing PPO advantages with
    # advantage_lambda; evaluate with lambda=1 for unbiased value targets.
    unroll_vf = jax_utils.with_compute_dtype(
        self._unroll_vf, config.value_dtype.dtype)
    train_unroll_vf = jax_utils.with_compute_dtype(
        functools.partial(
            self._unroll_vf,
            lambda_=config.gae_lambda,
            advantage_lambda=config.advantage_lambda),
        config.value_dtype.dtype)
    self._unroll_vf_mb = jax_utils.microbatch_fn(
        self._unroll_vf, **value_mbkwargs)
    self.unroll_vf = jax_utils.run_loss_fn(
        self.value_function,
        unroll_vf,
        **value_mbkwargs,
    )
    self._train_vf_mb = jax_utils.train_fn(
        train_unroll_vf, loss_scale=self._config.value_loss_scale,
        **value_mbkwargs)
    self.train_vf = jax_utils.cached_train_fn(
        self.value_function,
        self.value_optimizer,
        train_unroll_vf,
        **value_mbkwargs,
    )

    jit_ppo_epoch = jax_utils.nnx_jit(
        Learner[ControllerType].ppo_epoch,
        donate_argnums=0,
        static_argnames=['train'])
    self.jit_ppo_epoch = jax_utils.cached_partial(jit_ppo_epoch, self)

    fused_ppo = jax_utils.nnx_jit(
        Learner[ControllerType]._ppo,
        donate_argnums=0,
        static_argnames=['num_epochs'],
    )
    self.fused_ppo = jax_utils.cached_partial(fused_ppo, self)

    if device is not None:
      # Commits all parameters and optimizer state, so subsequent computations
      # (which follow committed inputs) run on this device.
      jax_utils.put_module_on_device(self, device)

  def initial_state(
      self, batch_size: int, rngs: tp.Optional[nnx.Rngs] = None,
  ) -> LearnerState:
    if rngs is None:
      rngs = nnx.Rngs(0)

    teacher_state = jax_utils.cast_floats_to_dtype(
        self.teacher.initial_state(batch_size, rngs),
        dtype=self._config.teacher_dtype.dtype)

    value_state = jax_utils.cast_floats_to_dtype(
        self.value_function.initial_state(batch_size, rngs),
        dtype=self._config.value_dtype.dtype)

    state = LearnerState(
        teacher=teacher_state,
        value_function=value_state,
    )
    if self._device is not None:
      state = jax.device_put(state, self._device)
    return state

  def policy_variables(self, to_numpy: bool = True):
    """Returns policy state for actor update via evaluators.update_variables."""
    return self.policy.get_state(to_numpy=to_numpy)

  def _sum_leaves(self, embedding: embed.Embedding, struct) -> Array:
    return functools.reduce(jnp.add, embedding.flatten(struct))

  def _compute_kl(self, logits_p: ControllerType, logits_q: ControllerType) -> Array:
    """Computes total KL(P||Q) summed over all controller components."""
    kls = self._controller_embedding.map(
        lambda e, lp, lq: e.kl_divergence(lp, lq),
        logits_p, logits_q)
    return self._sum_leaves(self._controller_embedding, kls)

  def _compute_entropy(self, logits: ControllerType) -> Array:
    """Computes total entropy H(P) summed over all controller components."""
    entropies = self._controller_embedding.map(
        lambda e, l: e.entropy(l), logits)
    return self._sum_leaves(self._controller_embedding, entropies)

  def _get_log_prob(self, logits: ControllerType, action: ControllerType) -> Array:
    """Computes log P(action | logits) summed over all controller components."""
    distances = self._controller_embedding.map(
        lambda e, l, a: e.distance(l, a), logits, action)
    return -self._sum_leaves(self._controller_embedding, distances)

  def _unroll_teacher(
      self,
      teacher: Policy[ControllerType],
      trajectory: Trajectory,
      initial_state: RecurrentState,
  ) -> tuple[jax_utils.Loss, ControllerType, RecurrentState]:
    teacher_frames = get_delayed_frames(trajectory)
    outputs = teacher.unroll(teacher_frames, initial_state)
    loss = -outputs.log_probs  # unused
    return loss, outputs.distances.logits, outputs.final_state

  def _unroll_vf(
      self,
      value_function: vf_lib.ValueFunction,
      trajectory: Trajectory,
      initial_state: RecurrentState,
      lambda_: float = 1.0,
      advantage_lambda: tp.Optional[float] = None,
  ) -> tuple[jax_utils.Loss, Metrics, RecurrentState, Advantage]:
    value_frames = get_frames(trajectory)
    outputs, final_state = value_function.loss(
        value_frames, initial_state, self.discount, lambda_=lambda_,
        advantage_lambda=advantage_lambda)
    # b16 metrics are annoying to log and print
    metrics = utils.map_single_structure(
        lambda x: jnp.astype(x, jnp.float32), outputs.metrics)
    return outputs.loss, metrics, final_state, outputs.advantages

  def unroll(
      self,
      trajectory: Trajectory[ControllerType, RecurrentState],
      initial_state: LearnerState,
  ):
    _, teacher_state = self.unroll_teacher(trajectory, initial_state.teacher)
    metrics, vf_state, _ = self.unroll_vf(trajectory, initial_state.value_function)
    return metrics, LearnerState(teacher=teacher_state, value_function=vf_state)

  def ppo_loss(
      self,
      policy: Policy,
      advantages: jax.Array,
      teacher_logits: ControllerType,
      trajectory: Trajectory,
  ) -> tp.Tuple[jax_utils.Loss, dict]:
    """Computes policy gradients for one PPO step.

    Value function outputs are for [0, U] while policy outputs are for
    [D, U+D]. This means we can only train on steps [D, U].

    Args:
      outputs: Pre-computed teacher and value function outputs.
      trajectory: The collected trajectory.

    Returns:
      Tuple of (gradients, metrics dict).
    """
    delay = policy.delay  # D
    remove_first = lambda t: t[delay:] if delay > 0 else t
    remove_last = lambda t: t[:t.shape[0] - delay] if delay > 0 else t

    # Advantages from [0, U]: take [D, U] -> U-D steps.
    advantages = jax.lax.stop_gradient(advantages[delay:])

    # Teacher logits: [D, U+D] -> truncate last D -> [D, U].
    # Note: no stop_gradient needed since teacher has no trainable variables.
    teacher_logits = utils.map_nt(remove_last, teacher_logits)

    # Policy frames: states [0, U-D+1], actions [D, U+1].
    policy_frames = data.Frames(
        state_action=data.StateAction(
            state=jax.tree.map(remove_last, trajectory.states),
            action=jax.tree.map(remove_first, trajectory.actions.controller_state),
            name=remove_last(trajectory.name),
            rating=remove_last(trajectory.rating),
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

    policy_dtype = jax_utils.module_dtype(policy)
    initial_state = jax_utils.cast_floats_to_dtype(
        trajectory.initial_state, policy_dtype)
    policy_outputs = policy.unroll(policy_frames, initial_state)
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

    loss = (
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

  def ppo_epoch(
      self,
      advantages: list[jax.Array],
      teacher_logits: list[ControllerType],
      trajectories: list[Trajectory],
      train: bool = True,
  ) -> dict:
    """One epoch of PPO: accumulate gradients over all trajectories."""

    # Instead of scanning over the trajectories we simply batch them and let
    # microbatching handle the scan.

    batch_size = advantages[0].shape[1]
    mbs = self._config.microbatch_size or batch_size

    batched_advantages = utils.map_nt(
        lambda *xs: jnp.concatenate(xs, _ADVANTAGES_AXIS),
        *advantages)
    batched_logits = utils.map_nt(
        lambda *xs: jnp.concatenate(xs, _TEACHER_LOGITS_AXIS),
        *teacher_logits)
    batched_trajectories = utils.map_nt(
        lambda axis, *substructs: jax.tree.map(
            lambda *xs: jnp.concatenate(xs, axis), *substructs),
        _TRAJECTORY_AXES, *trajectories)

    ppo_loss = jax_utils.with_compute_dtype(
        self.ppo_loss, self._config.policy_dtype.dtype)

    mbkwargs = MBKwargs(
        microbatch_size=mbs,
        input_batch_dims=(_ADVANTAGES_AXIS, _TEACHER_LOGITS_AXIS, _TRAJECTORY_AXES),
        output_batch_dims=1,  # metrics are time-major
    )

    if train:
      grad_fn = jax_utils.microbatched_grads(
          ppo_loss, loss_scale=self._config.policy_loss_scale, **mbkwargs)

      grads, metrics = grad_fn(
          self.policy,
          batched_advantages,
          batched_logits,
          batched_trajectories)

      grads_dict = nnx.to_pure_dict(grads)
      grad_norms = jax.tree.map(jnp.linalg.norm, grads_dict)
      max_grad_norm = jax.tree.reduce(jnp.maximum, grad_norms)

      grad_abs = jax.tree.map(jnp.abs, grads_dict)
      ps = np.linspace(10, 100, num=10)
      grad_percentiles = jax.tree.map(
        lambda x: jnp.percentile(x, ps), grad_abs)
      grad_percentiles = jax.tree.leaves(grad_percentiles)
      grad_percentiles = jnp.stack(grad_percentiles, axis=1)
      gmean_grad_percentiles = jnp.exp(jnp.log(grad_percentiles + 1e-9).mean(axis=1))
      grad_percentiles = jnp.unstack(gmean_grad_percentiles)

      metrics['grads'] = dict(
          norms=grad_norms,
          percentiles=dict(zip(ps, grad_percentiles)),
          max_norm=max_grad_norm,
      )

      self.policy_optimizer.update(self.policy, grads)
    else:
      run_ppo = jax_utils.microbatch_fn(
          jax_utils.no_loss(ppo_loss),
          **mbkwargs)
      (metrics,) = run_ppo(self.policy, batched_advantages, batched_logits, batched_trajectories)

    actor_kl = metrics['actor_kl']
    metrics['actor_kl'] = dict(
        mean=jnp.mean(actor_kl),
        max=jnp.max(actor_kl),
    )
    return metrics

  def _ppo(
      self,
      trajectories: list[Trajectory],
      initial_state: LearnerState,
      num_epochs: int,
      # set jit to false to jit-trace the whole thing
      jit: bool = False,
  ) -> tp.Tuple[LearnerState, dict]:
    assert len(trajectories) == self._config.ppo.num_batches

    if jit:
      unroll_teacher = self.unroll_teacher
      train_vf = self.train_vf
      ppo_epoch = self.jit_ppo_epoch
    else:
      unroll_teacher = jax_utils.partial(self._unroll_teacher_mb, self.teacher)
      train_vf = jax_utils.partial(
        self._train_vf_mb, self.value_function, self.value_optimizer)
      ppo_epoch = self.ppo_epoch

    timings: dict[str, float] = {}

    # Unroll teacher + value function, training value function.
    # NOTE: when using sim-envs with large batch sizes, we typically set
    # num_batches = 1, which means that there isn't any benefit from fusing
    # the loop over trajectories with a scan.

    start = time.perf_counter()
    teacher_state = initial_state.teacher
    teacher_logits: list[ControllerType] = []
    for trajectory in trajectories:
      logits, teacher_state = unroll_teacher(trajectory, teacher_state)
      teacher_logits.append(logits)
    timings['teacher'] = time.perf_counter() - start

    start = time.perf_counter()
    value_state = initial_state.value_function
    advantages: list[Advantage] = []
    value_metrics_list: list[Metrics] = []
    for trajectory in trajectories:
      metrics, value_state, advantage = train_vf(trajectory, value_state)
      advantages.append(advantage)
      value_metrics_list.append(metrics)

    # Collect value function metrics. Keep them on device so we don't block
    # dispatch of the PPO epochs below.
    value_metrics = utils.map_nt(
        lambda *xs: jnp.stack(xs).mean(), *value_metrics_list)

    timings['vf'] = time.perf_counter() - start

    # Checkpoint policy state for potential reverting.
    # checkpoint = jax_utils.get_module_state(self.policy)

    ppo_epoch = self.jit_ppo_epoch if jit else self.ppo_epoch

    # PPO epochs with gradient updates.
    per_epoch_metrics = []
    for i in range(num_epochs):
      start = time.perf_counter()
      epoch_metrics = ppo_epoch(advantages, teacher_logits, trajectories, train=True)
      per_epoch_metrics.append(epoch_metrics)
      timings[f'ppo_{i}'] = time.perf_counter() - start

    # Final eval epoch (no gradient update) to measure post-update KL.
    start = time.perf_counter()
    final_metrics = ppo_epoch(advantages, teacher_logits, trajectories, train=False)
    per_epoch_metrics.append(final_metrics)
    timings[f'ppo_eval'] = time.perf_counter() - start

    metrics = dict(
        ppo_step={str(i): m for i, m in enumerate(per_epoch_metrics)},
        post_update=final_metrics,
        value=value_metrics,
    )

    if jit:
      metrics['timing'] = timings

    hidden_state = LearnerState(
        teacher=teacher_state,
        value_function=value_state,
    )

    return hidden_state, metrics

  def ppo(
      self,
      trajectories: list[Trajectory],
      initial_state: LearnerState,
      step: int,
      jit: bool = True,
  ) -> tp.Tuple[LearnerState, dict]:
    """Multi-epoch PPO update.

    Note: this runs in python, not under jit.

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

    if step < self._config.value_burnin_epochs:
      num_epochs = 0
    elif step < self._config.value_burnin_epochs + self._config.optimizer_burnin_epochs:
      num_epochs = 1
    else:
      num_epochs = self._config.ppo.num_epochs

    if self._config.fused:
      final_state, metrics = self.fused_ppo(
          trajectories, initial_state, num_epochs)
    else:
      final_state, metrics = self._ppo(
          trajectories, initial_state, num_epochs, jit=jit)

    metrics['reverted'] = False

    return final_state, metrics

  def check_actor_kl(self, metrics: dict):
    """Raise if the post-update actor KL is too high.

    Blocks on this learner's device; when running multiple learners, call
    this only after all of them have dispatched their updates via `ppo`.
    """
    # TODO: implement state reversion instead of raising
    final_actor_kl = np.asarray(metrics['post_update']['actor_kl']['mean'])
    if final_actor_kl > self._config.ppo.max_mean_actor_kl:
      raise ValueError(
          f"Mean actor KL after PPO update is too high: {final_actor_kl}")

  def get_state(self) -> dict:
    state = jax_utils.get_module_state(self)
    del state['teacher']
    return state

  def restore_from_imitation(self, state_dict: dict):
    # Assumes policy/value function + optimizers have the same keys as in
    # imitation checkpoints. There is no teacher state but that's ok since
    # nn.update only needs the updates to be a subset of the state.
    # Checkpoints are unpickled as numpy arrays; convert to jax arrays.

    if 'teacher' in state_dict:
      del state_dict['teacher']
      logging.warning("Found teacher state in imitation checkpoint; ignoring it.")

    # device_put with device=None is equivalent to jnp.asarray.
    state_dict = jax.tree.map(
        lambda x: jax.device_put(x, self._device), state_dict)
    state_dict = jax.tree.map(jnp.asarray, state_dict)
    jax_utils.set_module_state(self, state_dict)
