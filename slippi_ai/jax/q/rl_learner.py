"""Online RL learner that trains a Q-policy by argmax over the Q-function.

This is the online (self-play / vs-opponent) analog of `q_policy_learner.py`,
mirroring `nash/rl_learner.py` but using a single-player Q-function and an argmax
objective instead of solving a Nash equilibrium.

The actor rolls out games using `policy` (the "sample" policy). Each step we:
  1. re-sample candidate actions from `policy`,
  2. score them with the (online-trained) `q_function`,
  3. take the argmax action sequence (`best_action`),
  4. regress `q_policy` toward `best_action` (+ optional teacher KL),
and every `epoch_length` steps copy `q_policy`'s weights into `policy`.
"""

import dataclasses
import functools
import logging
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from slippi_ai import utils
from slippi_ai.types import Action, Frames
from slippi_ai.jax.policies import Policy, RecurrentState, DistanceOutputs
from slippi_ai.jax import embed, rl_lib, jax_utils, saving
from slippi_ai.jax.agents import DType
from slippi_ai.jax.q import q_function as q_lib
from slippi_ai.jax.rl.learner import FrameSkipTrajectory, get_frames, MBKwargs

T = tp.TypeVar('T')
Rank2 = tuple[int, int]
field = lambda f: dataclasses.field(default_factory=f)


@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  q_fn_learning_rate: tp.Optional[float] = None
  reward_halflife: float = 4
  gae_lambda: float = 1.0

  num_samples: int = 1
  sample_batch_size: int = 0  # 0 means full batch size, i.e. vmap
  include_action_taken_in_samples: bool = True
  epoch_length: int = 100

  # Weight on the argmax-distillation loss (regress q_policy toward best action).
  argmax_weight: float = 1
  kl_teacher_weight: float = 0
  reverse_kl_teacher_weight: float = 0
  actor_kl_weight: float = 0
  # Weight the argmax-distillation loss by the advantage of the best action.
  weight_by_advantage: bool = False

  max_actor_kl: tp.Optional[float] = None

  # Delay q_policy updates while the q_function warms up.
  value_burnin_steps: int = 0

  sample_policy_dtype: DType = DType.FP32
  teacher_dtype: DType = DType.FP32
  q_policy_dtype: DType = DType.FP32
  q_fn_dtype: DType = DType.FP32

  microbatch_size: int = 0
  remat: bool = False


_SAMPLE_AXIS = 0

Loss = jax.Array
Metrics = dict

SAMPLE_POLICY = 'sample_policy'
Q_FUNCTION = 'q_function'
Q_POLICY = 'q_policy'
TEACHER = 'teacher'


def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
  masked_sum = jnp.sum(x * mask, keepdims=True)
  count = jnp.sum(mask)
  return masked_sum / (count + 1e-8)


def batch_fs(xs: list[T]) -> T:
  return utils.map_nt(lambda *xs: jnp.stack(xs, axis=1), *xs)


def warmup_schedule(burnin_steps: int, base_value: float):
  burnin = optax.constant_schedule(0)
  normal = optax.constant_schedule(base_value)
  return optax.join_schedules([burnin, normal], [burnin_steps])


@jax.jit
def copy_struct(struct: T) -> T:
  return jax.tree.map(jnp.copy, struct)

class QFunctionOutputs(tp.NamedTuple, tp.Generic[Action]):
  values: jax.Array
  action_init_state: RecurrentState
  best_action: list[Action]
  best_values: jax.Array  # Q-values for the best action

class Learner(nnx.Module, tp.Generic[Action]):

  def __init__(
      self,
      config: LearnerConfig,
      q_function_config: q_lib.QFunctionConfig,
      policy_config: dict,
      state: dict,
      rngs: nnx.Rngs,  # used for sampling
  ):
    self.config = config

    q_function = q_lib.build_q_function(rngs, q_function_config)
    self.q_function = q_function

    # We use `policy` to sample and act, and regress `q_policy` toward the
    # argmax of the q_function. At the end of each epoch, we copy the
    # q_policy's weights to the policy.
    self.policy: Policy[Action] = saving.policy_from_config_dict(policy_config)
    self.q_policy: Policy[Action] = saving.policy_from_config_dict(policy_config)
    self.teacher: Policy[Action] = saving.policy_from_config_dict(policy_config)

    if config.remat:
      self.q_policy.enable_remat()

    self._controller_embedding = self.policy.controller_head.controller_embedding

    self.discount = rl_lib.discount_from_halflife(
      config.reward_halflife, frame_skip=q_function.frame_skip)

    learning_rate = config.learning_rate

    self.policy_schedule = warmup_schedule(
        config.value_burnin_steps, config.learning_rate)

    self.policy_optimizer = nnx.Optimizer(
        self.q_policy, optax.adam(self.policy_schedule), wrt=nnx.Param)

    q_fn_learning_rate = config.q_fn_learning_rate or learning_rate
    self.q_function_optimizer = nnx.Optimizer(
        q_function, optax.adam(q_fn_learning_rate), wrt=nnx.Param)

    # NOTE: some jax_utils functions expect jax arrays inside modules.
    jax_utils.set_module_state(self, utils.map_nt(jnp.asarray, state))

    if not config.include_action_taken_in_samples and config.num_samples < 2:
      raise ValueError('num_samples must be at least 2 if not including action taken in samples')

    if config.sample_batch_size > 0:
      ns = config.num_samples
      if config.include_action_taken_in_samples:
        ns += 1
      if ns % config.sample_batch_size != 0:
        logging.warning(f'sample_batch_size {config.sample_batch_size} does not divide num_samples {ns}')

    self.num_samples = config.num_samples
    self.delay = self.policy.delay
    assert self.delay == 0
    self.frame_skip = self.policy.frame_skip
    assert self.q_function.frame_skip == self.frame_skip

    unroll_sample_policy = jax_utils.with_compute_dtype(
      self._unroll_sample_policy, config.sample_policy_dtype.dtype)

    self.run_sample_policy = jax_utils.cached_partial(
        jax_utils.nnx_jit(
            jax_utils.no_loss(unroll_sample_policy),
            donate_argnums=(0, 1, 3),
        ),
        self.policy, rngs,
    )

    unroll_q_function = jax_utils.with_compute_dtype(
      self._unroll_q_function, config.q_fn_dtype.dtype)

    self.train_q_function = jax_utils.cached_train_fn(
        module=self.q_function,
        optimizer=self.q_function_optimizer,
        loss_fn=unroll_q_function,
    )

    self.run_q_function = jax_utils.cached_partial(
        jax_utils.nnx_jit(
            jax_utils.no_loss(unroll_q_function),
            donate_argnums=(0, 2),
        ),
        self.q_function,
    )

    def unroll_teacher(
        teacher: Policy[Action],
        frames: Frames[Rank2, Action], /,
        initial_states: RecurrentState,  # [B]
    ) -> tuple[list[DistanceOutputs[Action]], RecurrentState]:
      teacher_outputs = teacher.unroll(frames, initial_states)
      return teacher_outputs.distances, teacher_outputs.final_state

    unroll_teacher = jax_utils.with_compute_dtype(
        unroll_teacher, config.teacher_dtype.dtype)

    self.run_teacher = jax_utils.cached_partial(
        jax_utils.nnx_jit(unroll_teacher, donate_argnums=(0, 2)),
        self.teacher,
    )

    TM = 1
    BM = 0
    _Q_FUNCTION_AXIS = None
    _RNG_AXIS = nnx.StateAxes({...: nnx.Carry})
    _FRAMES_AXIS = TM
    _STATE_AXIS = BM
    _Q_FUNCTION_OUTPUTS_AXIS = TM
    _TEACHER_OUTPUTS_AXIS = TM
    _ACTOR_LOGITS_AXIS = TM

    _METRICS_AXIS = BM

    q_policy_input_batch_dims = (
        _Q_FUNCTION_AXIS, _RNG_AXIS, _FRAMES_AXIS, _STATE_AXIS,
        _Q_FUNCTION_OUTPUTS_AXIS, _TEACHER_OUTPUTS_AXIS, _ACTOR_LOGITS_AXIS,
    )
    q_policy_output_batch_dims = (
        _METRICS_AXIS, _STATE_AXIS,
    )
    q_policy_mbkwargs = MBKwargs(
        microbatch_size=config.microbatch_size,
        input_batch_dims=q_policy_input_batch_dims,
        output_batch_dims=q_policy_output_batch_dims,
    )

    unroll_q_policy = jax_utils.with_compute_dtype(
        self._unroll_q_policy, config.q_policy_dtype.dtype)

    train_q_policy_mb = jax_utils.train_fn(
        unroll_q_policy,
        **q_policy_mbkwargs,
    )
    jit_train_q_policy = jax_utils.nnx_jit(
        train_q_policy_mb,
        donate_argnums=(0, 1, 2, 3, 5),
    )
    self.train_q_policy = jax_utils.cached_partial(
        jit_train_q_policy,
        self.q_policy, self.policy_optimizer, self.q_function, rngs,
    )

    run_q_policy_mb = jax_utils.microbatch_module(
        jax_utils.no_loss(unroll_q_policy),
        **q_policy_mbkwargs,
    )
    self.run_q_policy = jax_utils.cached_partial(
        jax_utils.nnx_jit(
            run_q_policy_mb,
            donate_argnums=(0, 1, 2, 4),
        ),
        self.q_policy, self.q_function, rngs,
    )

    def post_update(
        policy: Policy[Action], /,
        frames: Frames[Rank2, Action],
        initial_state: RecurrentState,
        fs_actor_logits: list[Action],  # FS x [T, B]
    ) -> Metrics:
      policy_outputs = policy.unroll(frames, initial_state)
      policy_logits = batch_fs([do.logits for do in policy_outputs.distances])

      actor_logits = batch_fs(fs_actor_logits)
      actor_logits = utils.map_nt(lambda x: x[1:], actor_logits)

      actor_kl = self._compute_kl(actor_logits, policy_logits)

      actor_kl_metrics = dict(
          mean=jnp.mean(actor_kl),
          max=jnp.max(actor_kl),
      )

      return {'post_update_actor_kl': actor_kl_metrics}

    post_update = jax_utils.with_compute_dtype(
        post_update, config.q_policy_dtype.dtype)

    self.post_update = jax_utils.cached_partial(
        jax_utils.nnx_jit(post_update),
        self.q_policy,
    )

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    initial_states = {
        Q_FUNCTION: self.q_function.initial_state(batch_size, rngs),
        TEACHER: self.teacher.initial_state(batch_size, rngs),
        Q_POLICY: self.q_policy.initial_state(batch_size, rngs),
        # (sample) policy state is supplied by the actor (trajectory.initial_state)
    }

    dtypes = {
        Q_FUNCTION: self.config.q_fn_dtype,
        TEACHER: self.config.teacher_dtype,
        Q_POLICY: self.config.q_policy_dtype,
    }
    for key, dtype in dtypes.items():
      initial_states[key] = jax_utils.cast_floats_to_dtype(initial_states[key], dtype.dtype)

    return initial_states

  def policy_variables(self, to_numpy: bool = True):
    """Returns policy state for actor update via evaluators.update_variables."""
    return self.policy.get_state(to_numpy=to_numpy)

  def _sum_leaves(self, embedding: embed.Embedding[tp.Any, T], struct: T) -> jax.Array:
    return functools.reduce(jnp.add, embedding.flatten(struct))

  def _compute_kl(self, logits_p: Action, logits_q: Action) -> jax.Array:
    """Computes total KL(P||Q) summed over all controller components."""
    kls = self._controller_embedding.map(
        lambda e, lp, lq: e.kl_divergence(lp, lq),
        logits_p, logits_q)
    return self._sum_leaves(self._controller_embedding, kls)

  def _compute_entropy(self, logits: Action) -> jax.Array:
    """Computes total entropy H(P) summed over all controller components."""
    entropies = self._controller_embedding.map(
        lambda e, l: e.entropy(l), logits)
    return self._sum_leaves(self._controller_embedding, entropies)

  def _unroll_sample_policy(
      self,
      sample_policy: Policy[Action],
      rngs: nnx.Rngs,
      frames: Frames[Rank2, Action],  # [T, B]
      initial_states: RecurrentState,  # [B]
  ) -> tuple[Loss, Metrics, RecurrentState, list[Action]]:
    action = frames.state_action.action
    prev_action = utils.map_nt(lambda t: t[:-1], action)

    # sample policy initial states come from the actor which might be in a different dtype
    initial_states = jax_utils.cast_floats_to_dtype(
        initial_states, self.config.sample_policy_dtype.dtype)
    sample_policy_outputs = sample_policy.unroll_with_outputs(frames, initial_states)

    # Because the action space is too large, we compute a finite subsample
    # using the sample_policy.
    @nnx.vmap(in_axes=(None, 0), out_axes=_SAMPLE_AXIS)
    def sample(sample_policy: Policy[Action], rngs: nnx.Rngs):
      sample_outputs = sample_policy.controller_head.sample(
          rngs=rngs,
          inputs=sample_policy_outputs.outputs,
          prev_controller_state=prev_action)
      return [so.controller_state for so in sample_outputs]

    policy_samples = sample(sample_policy, rngs.fork(split=self.num_samples))

    bm_loss = jnp.mean(sample_policy_outputs.imitation_loss, axis=0)
    bm_metrics = utils.map_single_structure(
      lambda x: jnp.mean(x, axis=0), sample_policy_outputs.metrics)

    return bm_loss, bm_metrics, sample_policy_outputs.final_state, policy_samples

  def _unroll_q_function(
      self,
      q_function: q_lib.QFunction[Action],
      frames: Frames[Rank2, Action],  # [T, B]
      initial_states: RecurrentState,  # [B]
      policy_samples: list[Action],  # frame_skip x [S, T, B]
      lambda_: float = 1.0,
  ):
    q_outputs, action_init_state, final_state = q_function.loss_and_action_state(
        frames, initial_states, self.discount, lambda_=lambda_)

    q_bias = q_outputs.q_values - q_outputs.values

    assert _SAMPLE_AXIS == 0
    sample_q_values = q_function.multi_q_values_from_action_state(
        values=q_outputs.values,
        action_init_state=action_init_state,
        actions=policy_samples,
        batch_size=self.config.sample_batch_size,
    )  # [S, T, B]

    sample_policy_expected_return = jnp.mean(sample_q_values, axis=_SAMPLE_AXIS)
    sample_policy_advantages = sample_policy_expected_return - q_outputs.values

    actions = policy_samples
    q_values = sample_q_values

    if self.config.include_action_taken_in_samples:
      actions = utils.map_nt(
        lambda samples, action_taken: jnp.concatenate(
          [samples, jnp.expand_dims(action_taken[1:], axis=_SAMPLE_AXIS)], axis=_SAMPLE_AXIS),
        policy_samples, frames.state_action.action)
      q_values = jnp.concatenate(
        [sample_q_values, jnp.expand_dims(q_outputs.q_values, axis=_SAMPLE_AXIS)], axis=_SAMPLE_AXIS)

    best_action_index = jnp.argmax(q_values, axis=_SAMPLE_AXIS, keepdims=True)
    best_action = jax.tree.map(
        lambda x: jnp.squeeze(
            jnp.take_along_axis(x, best_action_index, axis=_SAMPLE_AXIS),
            axis=_SAMPLE_AXIS),
        actions)

    optimal_expected_return = jnp.max(q_values, axis=_SAMPLE_AXIS)
    optimal_advantages = optimal_expected_return - q_outputs.values
    optimal_advantage_std = jnp.std(optimal_advantages, keepdims=True)
    optimal_advantage_log_std = jnp.log(optimal_advantages + 1e-8).std(keepdims=True)

    bm_loss = jnp.mean(q_outputs.loss, axis=0)  # [T, B] -> [B]
    metrics = dict(
        q_outputs.metrics,
        sample_policy_advantages=sample_policy_advantages,
        optimal_advantages=optimal_advantages,
        optimal_advantage_std=optimal_advantage_std,
        optimal_advantage_log_std=optimal_advantage_log_std,
        q_bias=q_bias,
    )
    bm_metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), metrics)

    outputs = QFunctionOutputs(
        values=q_outputs.values,
        action_init_state=action_init_state,
        best_action=best_action,
        best_values=optimal_expected_return,
    )

    return bm_loss, bm_metrics, final_state, outputs

  def _unroll_q_policy(
      self,
      q_policy: Policy[Action],
      q_function: q_lib.QFunction[Action],
      rngs: nnx.Rngs,
      frames: Frames[Rank2, Action],  # [T, B]
      initial_states: RecurrentState,  # [B]
      q_function_outputs: QFunctionOutputs[Action],
      teacher_outputs: list[DistanceOutputs[Action]],  # FS x [T, B]
      fs_actor_logits: list[Action],  # FS x [T, B]
  ) -> tuple[Loss, dict, RecurrentState]:
    action = frames.state_action.action
    prev_action = utils.map_nt(lambda t: t[:-1], action)

    q_policy_outputs = q_policy.unroll_with_outputs(frames, initial_states)
    q_policy_imitation_loss = q_policy_outputs.imitation_loss

    # Argmax-distillation: regress q_policy toward the best (highest-q) action.
    q_policy_distances = q_policy.controller_head.distance(
        inputs=q_policy_outputs.outputs,
        prev_controller_state=prev_action,
        target_controller_state=q_function_outputs.best_action,
    )
    q_policy_argmax_loss = jax_utils.add_n(q_policy_distances) / len(q_policy_distances)

    if self.config.weight_by_advantage:
      optimal_advantages = q_function_outputs.best_values - q_function_outputs.values
      q_policy_argmax_loss *= optimal_advantages / optimal_advantages.mean()

    # Estimate q_policy returns.
    q_policy_samples = [
        so.controller_state
        for so in q_policy.controller_head.sample(
            rngs=rngs,
            inputs=q_policy_outputs.outputs,
            prev_controller_state=prev_action)]

    q_function = jax_utils.cast_params_to_dtype(
        q_function, self.config.q_fn_dtype.dtype)
    q_policy_sample_q_values = q_function.q_values_from_action_state(
        values=q_function_outputs.values,
        action_init_state=q_function_outputs.action_init_state,
        actions=q_policy_samples,
    )
    q_policy_advantages = q_policy_sample_q_values - q_function_outputs.values
    optimality_gap = q_function_outputs.best_values - q_policy_sample_q_values

    q_policy_logits = batch_fs([do.logits for do in q_policy_outputs.distances])
    teacher_logits = batch_fs([do.logits for do in teacher_outputs])
    actor_logits = batch_fs(fs_actor_logits)
    actor_logits = utils.map_nt(lambda x: x[1:], actor_logits)

    # These aren't techincally correct -- the actions should be sampled from
    # the first distribution when computing the KL. Even the actor_kl is not
    # quite right because the trajectory comes from picking the best action
    # sequence from among the S samples for each state.
    teacher_kl = self._compute_kl(q_policy_logits, teacher_logits)  # [T, FS, B]
    reverse_teacher_kl = self._compute_kl(teacher_logits, q_policy_logits)
    actor_kl = self._compute_kl(actor_logits, q_policy_logits)
    entropy = self._compute_entropy(q_policy_logits)

    def fs_mean(x: jax.Array) -> jax.Array:
      assert x.shape[1] == self.frame_skip
      return jnp.mean(x, axis=1)

    losses = [
        self.config.argmax_weight * q_policy_argmax_loss,
        self.config.kl_teacher_weight * fs_mean(teacher_kl),
        self.config.reverse_kl_teacher_weight * fs_mean(reverse_teacher_kl),
        self.config.actor_kl_weight * fs_mean(actor_kl),
    ]
    q_policy_total_loss = jax_utils.add_n(losses)

    metrics = dict(
        q_loss=q_policy_argmax_loss,
        imitation_loss=q_policy_imitation_loss,
        total_loss=q_policy_total_loss,
        q_policy_advantages=q_policy_advantages,
        optimality_gap=optimality_gap,
        teacher_kl=teacher_kl,
        reverse_teacher_kl=reverse_teacher_kl,
        actor_kl=actor_kl,
        entropy=entropy,
    )

    bm_loss = jnp.mean(q_policy_total_loss, axis=0)
    bm_metrics = utils.map_single_structure(
      lambda x: jnp.mean(x, axis=0), metrics)

    return bm_loss, bm_metrics, q_policy_outputs.final_state

  def step_sample_policy(self, tm_frames, initial_state):
    return self.run_sample_policy(tm_frames, initial_state)

  def step_q_function(self, tm_frames, initial_state, policy_samples, train: bool):
    fn = self.train_q_function if train else self.run_q_function
    lambda_ = self.config.gae_lambda if train else 1.0
    return fn(tm_frames, initial_state, policy_samples, lambda_)

  def step(
      self,
      trajectory: FrameSkipTrajectory[Action],  # [T, B]
      initial_states: dict[str, RecurrentState],  # [B]
      step: int,
      train: bool = True,
  ) -> tuple[dict, dict[str, RecurrentState]]:
    frames = get_frames(trajectory)

    final_states = dict(initial_states)
    metrics = {}

    (
      metrics[SAMPLE_POLICY],
      _,
      policy_samples,
    ) = self.step_sample_policy(frames, trajectory.initial_state)

    (
      metrics[Q_FUNCTION],
      final_states[Q_FUNCTION],
      q_function_outputs,
    ) = self.step_q_function(
        frames, initial_states[Q_FUNCTION], policy_samples, train=train)
    del policy_samples

    (
      teacher_outputs,
      final_states[TEACHER],
    ) = self.run_teacher(frames, initial_states[TEACHER])


    step_q_policy = self.train_q_policy if train else self.run_q_policy
    actor_logits = [so.logits for so in trajectory.actions]
    # Need a copy since the original gets donated by the q_policy update.
    initial_q_policy_state = copy_struct(initial_states[Q_POLICY])
    (
      metrics[Q_POLICY],
      final_states[Q_POLICY],
    ) = step_q_policy(
        frames, initial_states[Q_POLICY], q_function_outputs, teacher_outputs,
        actor_logits)

    post_update_metrics = self.post_update(
        frames, initial_q_policy_state, actor_logits)
    metrics[Q_POLICY].update(post_update_metrics)

    mean_actor_kl = jax.device_get(post_update_metrics['post_update_actor_kl']['mean'])
    if self.config.max_actor_kl is not None and mean_actor_kl > self.config.max_actor_kl:
      raise ValueError(f"Mean actor KL after Q-policy update is too high: {mean_actor_kl}")

    if train and step % self.config.epoch_length == 0:
      jax_utils.set_module_state(
          self.policy, jax_utils.get_module_state(self.q_policy))

    return metrics, final_states
