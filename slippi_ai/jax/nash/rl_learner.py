"""Online nash RL learner.

The actor rolls out self-play with `policy` (the sample policy). Each step we
sample chains of actions from it, score every pair of chains with the
(online-trained) `q_function`, solve the nash of the resulting payoff
matrices, and regress `nash_policy` towards the nash distribution (plus KL
terms to the `teacher`). Every `epoch_length` steps `nash_policy`'s weights
are copied into `policy`.

With delay the chain game of docs/plans/nash_policy_delay.md applies: the
frame-skip converter (rl/learner.py) folds the actor's queued actions into
the trajectory and overlaps consecutive rollouts by skip_delay steps, which
gives each rollout the chunk layout of the offline learner
(nash_utils.ChunkLayout): every network unrolls over T = U + Ds steps, the
game and every loss cover the U valid indices [Ds, T), and the learner
carries every network's state (the sample policy's included) from after
index U - 1, where the next rollout starts. The chain machinery itself is
shared with the offline learner (chain_game.py).
"""

import dataclasses
import functools
import typing as tp

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from slippi_ai import utils
from slippi_ai.types import Frames, Action
from slippi_ai.jax.policies import Policy, RecurrentState
from slippi_ai.jax import embed, rl_lib, jax_utils, saving
from slippi_ai.jax.agents import DType
from slippi_ai.nash import data as nash_data
from slippi_ai.jax.nash import (
    chain_game,
    q_function as q_lib,
    utils as nash_utils,
)
from slippi_ai.jax.nash import nash
from slippi_ai.jax.rl.learner import (
    FrameSkipTrajectory, from_so_frames, get_delayed_frames,
)

T = tp.TypeVar('T')
Rank3 = tuple[int, int, int]

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  q_fn_learning_rate: tp.Optional[float] = None
  reward_halflife: float = 4
  gae_lambda: float = 0

  num_samples: int = 1
  sample_batch_size: int = 0  # 0 means full batch size, i.e. vmap
  include_action_taken_in_samples: bool = True
  subsample: tp.Optional[int] = None
  epoch_length: int = 100

  remat: bool = True

  # Number of epistemic indices to sample. The nash is solved once per index,
  # and the nash_policy regresses to the mixture of the per-index nash
  # distributions. Needs to be at least 2 for the epistemic metrics.
  num_index_samples: int = 4

  nash_weight: float = 1
  weight_by_advantage: bool = True

  initial_kl_weight: float = 3e-1
  kl_weight_lr: float = 3e-2
  target_teacher_kl: float = 0.05
  target_reverse_teacher_kl: float = 0.05

  value_burnin_steps: int = 0

  sample_policy_dtype: DType = DType.FP32
  teacher_dtype: DType = DType.FP32
  nash_policy_dtype: DType = DType.FP32
  q_fn_dtype: DType = DType.FP32

  microbatch_size: int = 0
  teacher_mbs: int = 0


Loss = jax.Array
Metrics = dict

QFunctionOutputs = chain_game.QFunctionOutputs

SAMPLE_POLICY = 'sample_policy'
Q_FUNCTION = 'q_function'
NASH = 'nash'
NASH_POLICY = 'nash_policy'
TEACHER = 'teacher'


def batch_fs(xs: list[T]) -> T:
  return utils.map_nt(
      lambda *xs: jnp.stack(xs, axis=1),
      *xs
  )

def warmup_schedule(burnin_steps: int, base_value: float):
  burnin = optax.constant_schedule(0)
  normal = optax.constant_schedule(base_value)
  return optax.join_schedules([burnin, normal], [burnin_steps])

@jax.jit
def copy_struct(struct: T) -> T:
  return jax.tree.map(jnp.copy, struct)

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

    # We use the policy to sample and act, and regress the nash_policy towards
    # the resulting Nash distribution. At the end of each epoch, we copy the
    # nash_policy's weights to the policy.
    self.policy: Policy[Action] = saving.policy_from_config_dict(policy_config)
    self.nash_policy: Policy[Action] = saving.policy_from_config_dict(policy_config)
    self.teacher: Policy[Action] = saving.policy_from_config_dict(policy_config)

    if config.remat:
      # Only the nash_policy is trained.
      self.nash_policy.enable_remat()

    self._controller_embedding = self.policy.controller_head.controller_embedding

    # within-frame discount
    self.discount = rl_lib.discount_from_halflife(config.reward_halflife)
    # across-frame discount
    self.fs_discount = rl_lib.discount_from_halflife(
      config.reward_halflife, frame_skip=q_function.frame_skip)

    learning_rate = config.learning_rate

    self.policy_schedule = warmup_schedule(
        config.value_burnin_steps,
        config.learning_rate,
    )

    self.policy_optimizer = nnx.Optimizer(
        self.policy, optax.adam(self.policy_schedule), wrt=nnx.Param)

    q_fn_learning_rate = config.q_fn_learning_rate or learning_rate

    self.q_function_optimizer = nnx.Optimizer(
        q_function, optax.adam(q_fn_learning_rate), wrt=nnx.Param)

    kl_weight_schedule = warmup_schedule(
        config.value_burnin_steps, config.kl_weight_lr)
    self.kl_teacher_weights = jax_utils.KLTeacherWeights(config.initial_kl_weight)
    self.kl_teacher_weights_optimizer = nnx.Optimizer(
      self.kl_teacher_weights, optax.sgd(kl_weight_schedule), wrt=nnx.Param)

    # NOTE: some jax_utils functions expect jax arrays inside modules.
    jax_utils.set_module_state(self, utils.map_nt(jnp.asarray, state))

    self.delay = self.policy.delay
    self.frame_skip = self.policy.frame_skip
    assert self.q_function.frame_skip == self.frame_skip
    # With delay, the game at each index is over chains of skip_delay + 1
    # actions; see docs/plans/nash_policy_delay.md and chain_game. The
    # frame-skip converter overlaps rollouts by skip_delay steps.
    self.layout = nash_utils.ChunkLayout(self.delay, self.frame_skip)
    self.skip_delay = self.layout.skip_delay
    self.game = chain_game.ChainGame[Action](
        self.layout,
        num_samples=config.num_samples,
        include_action_taken_in_samples=config.include_action_taken_in_samples,
        subsample=config.subsample,
        sample_batch_size=config.sample_batch_size,
    )

    jax_utils.cast_module_state_to_dtype(
      self.policy, config.sample_policy_dtype.dtype)

    self.run_sample_policy = jax_utils.cached_partial(
        jax_utils.nnx_jit(
            jax_utils.no_loss(self.game.unroll_sample_policy),
            donate_argnums=(0, 1, 3),
        ),
        self.policy, rngs,
    )

    unroll_q_function = jax_utils.with_compute_dtype(
      self._unroll_q_function, config.q_fn_dtype.dtype)

    self.train_q_function = jax_utils.train_fn_with_rngs(
        module=self.q_function,
        optimizer=self.q_function_optimizer,
        rngs=rngs.fork(),
        loss_fn=unroll_q_function,
    )

    self.run_q_function = jax_utils.cached_partial(
        jax_utils.nnx_jit(
            jax_utils.no_loss(unroll_q_function),
            donate_argnums=(0, 1, 3),
        ),
        self.q_function, rngs.fork(),
    )

    # We can't shard_map the qpax solver because of vma issues with while_loop.
    # The solution would be to insert a manual pvary inside qpax like we do in
    # our own ippd solver, but we can also just let jit handle running on
    # multiple devices as the solver is completely batch-parallel.
    self.compute_nash = jax_utils.jit(self._compute_nash)
    self.compute_nash = jax.profiler.annotate_function(self.compute_nash)

    def unroll_teacher(
        teacher: Policy[Action],
        frames: Frames[nash_data.Rank3, Action], /,
        initial_states: RecurrentState,  # [B, 2]
    ) -> tuple[jax.Array, RecurrentState]:
      # Only compute the core network outputs at the game indices; the
      # controller head is applied in _unroll_nash_policy where the teacher's
      # logits are evaluated on actions sampled from the teacher and
      # nash_policy.
      inputs = utils.map_nt(lambda t: t[:-1], frames.state_action)
      outputs, hidden_states = teacher.network.scan(
          inputs, frames.is_resetting[:-1], initial_states)
      return (
          self.layout.game_slice(outputs),
          self.layout.carried_state(hidden_states, frames),
      )

    jax_utils.cast_module_state_to_dtype(
      self.teacher, config.teacher_dtype.dtype)

    self.run_teacher = jax_utils.cached_partial(
        jax_utils.nnx_jit(
            unroll_teacher,
            donate_argnums=(0, 2),
        ),
        self.teacher,
    )

    unroll_nash_policy = jax_utils.with_compute_dtype(
        self._unroll_nash_policy, config.nash_policy_dtype.dtype)

    train_nash_policy = jax_utils.train_fn(unroll_nash_policy)

    self.train_nash_policy = jax_utils.cached_partial(
        jax_utils.nnx_jit(train_nash_policy, donate_argnums=(0, 1, 2, 3, 4, 6)),
        self.nash_policy, self.policy_optimizer, rngs,
        self.teacher, self.kl_teacher_weights,
    )

    self.run_nash_policy = jax_utils.cached_partial(
        jax_utils.nnx_jit(
            jax_utils.no_loss(unroll_nash_policy),
            donate_argnums=(0, 1, 2, 3, 5),
        ),
        self.nash_policy, rngs, self.teacher, self.kl_teacher_weights,
    )

    # Scores the nash_policy's sampled chain with the q_function. This runs
    # on the q_function (not closed over by the nash_policy's loss) because
    # its networks may contain nnx transforms, which fail on a closed-over
    # module inside another module's loss.
    nash_policy_qs = jax_utils.with_compute_dtype(
        self._nash_policy_qs, config.q_fn_dtype.dtype)
    self.run_nash_policy_qs = jax_utils.cached_partial(
        jax_utils.nnx_jit(nash_policy_qs, donate_argnums=0),
        self.q_function,
    )

    def post_update(
        policy: Policy[Action], /,
        frames: Frames[nash_data.Rank3, Action],
        initial_state: RecurrentState,
        fs_actor_logits: list[Action],  # FS x [T + 1, B, 2]
    ) -> Metrics:
      policy_outputs = policy.unroll(frames, initial_state)
      policy_logits = batch_fs([
          do.logits for do in policy_outputs.distances])

      actor_logits = batch_fs(fs_actor_logits)
      actor_logits = utils.map_nt(lambda x: x[1:], actor_logits)

      actor_kl = self._compute_kl(actor_logits, policy_logits)  # [T, FS, B, 2]

      metrics = {
          'post_update_actor_kl': self.layout.game_slice(actor_kl)
      }
      return metrics

    post_update = jax_utils.with_compute_dtype(
        post_update, config.nash_policy_dtype.dtype)

    self.post_update = jax_utils.cached_partial(
        jax_utils.nnx_jit(post_update, donate_argnums=0),
        self.nash_policy,
    )

    train_kl_teacher_weights = jax_utils.nnx_jit(
        jax_utils.train_fn(self._unroll_kl_teacher_weights),
        donate_argnums=(0, 1),
    )
    self.train_kl_teacher_weights = jax_utils.cached_partial(
        train_kl_teacher_weights,
        self.kl_teacher_weights, self.kl_teacher_weights_optimizer,
    )

    @nnx.jit(donate_argnums=(0, 1))
    def update_policy(
      policy: Policy[Action],
      nash_policy: Policy[Action],
    ):
      nash_policy_state = nnx.state(nash_policy)
      policy_state = jax_utils.cast_floats_to_dtype(
          nash_policy_state, self.config.sample_policy_dtype.dtype)
      nnx.update(policy, policy_state)

    self._update_policy = jax_utils.cached_partial(
        update_policy, self.policy, self.nash_policy)

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    # The sample policy's state is carried by the learner rather than taken
    # from the actor: with the overlap, each chunk starts skip_delay steps
    # before the rollout the actor's state belongs to. The two agree as the
    # actor runs the same policy (up to its dtype).
    initial_states = {
        SAMPLE_POLICY: self.policy.initial_state((batch_size, 2), rngs),
        Q_FUNCTION: self.q_function.initial_state(batch_size, rngs),
        TEACHER: self.teacher.initial_state((batch_size, 2), rngs),
        NASH_POLICY: self.nash_policy.initial_state((batch_size, 2), rngs),
    }

    dtypes = {
        SAMPLE_POLICY: self.config.sample_policy_dtype,
        Q_FUNCTION: self.config.q_fn_dtype,
        TEACHER: self.config.teacher_dtype,
        NASH_POLICY: self.config.nash_policy_dtype,
    }

    for key, dtype in dtypes.items():
      initial_states[key] = jax_utils.cast_floats_to_dtype(initial_states[key], dtype.dtype)

    return initial_states

  def policy_variables(self):
    """Returns policy state for actor update via evaluators.update_variables."""
    return self.policy.get_state(to_numpy=False)

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

  def _unroll_q_function(
      self,
      q_function: q_lib.QFunction[Action],
      rngs: nnx.Rngs,
      frames: Frames[nash_data.Rank3, Action],  # [T + 1, B, 2]
      initial_states: RecurrentState,  # [B, 2]
      policy_samples: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      lambda_: float | jax.Array = 1.0,
  ) -> tuple[Loss, ...]:  # (loss, *QFunctionOutputs)
    """Trains the q_function on the game indices and scores every pair of
    sampled chains (chain_game.ChainGame.unroll_q_function)."""
    bm_loss, outputs = self.game.unroll_q_function(
        q_function, rngs, frames, initial_states, policy_samples,
        num_index_samples=self.config.num_index_samples,
        discount=self.fs_discount, lambda_=lambda_)
    return (bm_loss, *outputs)

  def _compute_nash(
      self,
      q_values: jax.Array,  # [N, S, S, T, B, 2]
  ) -> tuple[nash.NashVariables, Metrics]:
    num_indices, s1, s2, t, b, n = q_values.shape
    assert n == 2

    payoff_matrices = nash_utils.mixed_payoff_matrices(q_values)  # [N, T, B, S, S]

    # Use separate vmaps over T and B to avoid an XLA SPMD partitioner
    # bug triggered by vmapping over the merged T*B sharded dimension.
    # Only triggered by qpax_fast, probably because it has matrices with
    # some dimensions equal to one.

    solve_vmap = jax_utils.multi_vmap(
        nash._solve_nash_simplex_impl, axes=[0, 1, 2])

    # One nash solution per epistemic index; leaves are [N, T, B, ...].
    nash_variables, nm_metrics = solve_vmap(payoff_matrices)

    nash_variables = utils.map_single_structure(
        lambda x: x.astype(jnp.float32), nash_variables)

    # Batch-major metrics; keep time and index dims so we can take max over num_steps.
    bm_metrics = utils.map_single_structure(
        lambda x: jnp.moveaxis(x, 2, 0), nm_metrics)  # [B, N, T, ...]

    bm_metrics['num_steps_max'] = jnp.max(bm_metrics['num_steps'], keepdims=True)

    return nash_variables, bm_metrics

  def _unroll_nash_policy(
      self,
      nash_policy: Policy[Action],
      rngs: nnx.Rngs,
      teacher: Policy[Action],
      kl_teacher_weights: jax_utils.KLTeacherWeights,
      frames: Frames[nash_data.Rank3, Action],  # [T + 1, B, 2]
      initial_states: RecurrentState,  # [B, 2]
      policy_samples: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      q_values: jax.Array,  # [N, S, S, T', B, 2]
      nash_solution: nash.NashVariables,  # [N, T', B]
      teacher_core_outputs: jax.Array,  # [T', B, 2, O]
      fs_actor_logits: list[Action],  # FS x [T + 1, B, 2]
  ) -> tuple[Loss, dict, RecurrentState, nash_utils.Chain[Action]]:
    """Trains the nash_policy towards the nash distributions over chains,
    regularized towards the teacher.

    The nash term is the cross-entropy from the mixture (over epistemic
    indices) of the nash distributions over the S chains to the
    nash_policy's chain log-probs (chain_game.ChainGame). The KL terms and
    diagnostics are on the one-step distributions at the game indices,
    conditioned on the actions actually taken. Also returns a chain sampled
    from the nash_policy (leaves [T', B, 2]), which _nash_policy_qs scores
    against the nash distribution.
    """
    layout = self.layout

    chains, num_samples = self.game.chains(policy_samples, frames)
    del policy_samples
    targets = self.game.targets(chains, num_samples, q_values, nash_solution)
    metrics = dict(targets.metrics)

    nash_policy_outputs = nash_policy.scan_with_outputs(frames, initial_states)
    nash_policy_imitation_loss = layout.game_slice(
        nash_policy_outputs.imitation_loss)  # [T', B, 2]
    context = layout.context(
        nash_policy_outputs.outputs, nash_policy_outputs.hidden_states, frames)

    nash_cross_entropy = self.game.cross_entropy(
        nash_policy, context, targets)  # [T', B, 2]

    if self.config.weight_by_advantage:
      # Weight the cross-entropy by how much better the nash distribution does
      # compared to the sample policy, i.e. how much we gain by using the nash
      # distribution for that state.
      nash_advantage = targets.nash_advantage
      nash_cross_entropy *= nash_advantage / nash_advantage.mean()

    # Sampled as the nash_policy would act online; scored in _nash_policy_qs.
    nash_policy_chain = nash_utils.sample_chain(nash_policy, rngs, context)

    # The one-step distributions at the game indices: the output at index t,
    # given the real action there, predicts action t + 1.
    action = frames.state_action.action
    prev_action = layout.game_slice(utils.map_nt(lambda t: t[:-1], action))
    outputs = layout.game_slice(nash_policy_outputs.outputs)  # [T', B, 2, O]

    nash_policy_logits = layout.game_slice(batch_fs([
        do.logits for do in nash_policy_outputs.distances]))  # [T', FS, B, 2]

    actor_logits = batch_fs(fs_actor_logits)
    actor_logits = layout.game_slice(
        utils.map_nt(lambda x: x[1:], actor_logits))

    # The exact KL between autoregressive policies is intractable because
    # later action components condition on earlier sampled ones. Instead we
    # sample actions from the "P" policy, condition both policies on them,
    # and take the analytic per-component KL, which is an unbiased
    # (Rao-Blackwellized) estimate of the true KL.

    # KL(nash_policy || teacher), sampling from the nash_policy.
    nash_policy_sample_outputs = nash_policy.controller_head.sample(
        rngs=rngs,
        inputs=outputs,
        prev_controller_state=prev_action)
    nash_policy_samples = [  # list[Controller[T', B, 2]]
        so.controller_state for so in nash_policy_sample_outputs]
    nash_policy_sample_logits = batch_fs(
        [so.logits for so in nash_policy_sample_outputs])
    teacher_on_nash_policy_samples = teacher.controller_head.distance_outputs(
        inputs=teacher_core_outputs,
        prev_controller_state=prev_action,
        target_controller_state=nash_policy_samples,
    )
    teacher_logits_on_nash_policy_samples = batch_fs(
        [do.logits for do in teacher_on_nash_policy_samples])
    teacher_kl = self._compute_kl(
        nash_policy_sample_logits, teacher_logits_on_nash_policy_samples)  # [T', FS, B, 2]

    # KL(teacher || nash_policy), sampling from the teacher.
    teacher_sample_outputs = teacher.controller_head.sample(
        rngs=rngs,
        inputs=teacher_core_outputs,
        prev_controller_state=prev_action)
    teacher_samples = [so.controller_state for so in teacher_sample_outputs]
    teacher_sample_logits = batch_fs(
        [so.logits for so in teacher_sample_outputs])
    nash_policy_on_teacher_samples = nash_policy.controller_head.distance_outputs(
        inputs=outputs,
        prev_controller_state=prev_action,
        target_controller_state=teacher_samples,
    )
    nash_policy_logits_on_teacher_samples = batch_fs(
        [do.logits for do in nash_policy_on_teacher_samples])
    reverse_teacher_kl = self._compute_kl(
        teacher_sample_logits, nash_policy_logits_on_teacher_samples)

    # The actor_kl is already such an estimate: the trajectory actions were
    # sampled from the actor, whose logits were recorded at sampling time, and
    # the nash_policy is teacher-forced on those same actions.
    actor_kl = self._compute_kl(actor_logits, nash_policy_logits)  # [T', FS, B, 2]
    # Like the KLs, the entropy conditions on prefixes sampled from the
    # nash_policy itself.
    entropy = self._compute_entropy(nash_policy_sample_logits)  # [T', FS, B, 2]

    def fs_mean(x: jax.Array) -> jax.Array:
      assert x.shape[1] == self.frame_skip
      return jnp.mean(x, axis=1)

    losses = [
        self.config.nash_weight * nash_cross_entropy,
        kl_teacher_weights.fwd_weight() * fs_mean(teacher_kl),
        kl_teacher_weights.bwd_weight() * fs_mean(reverse_teacher_kl),
    ]
    nash_policy_total_loss = jax_utils.add_n(losses)

    metrics.update(
        nash_cross_entropy=nash_cross_entropy,
        imitation_loss=nash_policy_imitation_loss,
        total_loss=nash_policy_total_loss,
        teacher_kl=teacher_kl,
        reverse_teacher_kl=reverse_teacher_kl,
        actor_kl=actor_kl,
        entropy=entropy,
    )

    bm_loss = jnp.mean(nash_policy_total_loss, axis=[0, 2])
    bm_metrics = utils.map_single_structure(
      lambda x: jnp.mean(x, axis=0), metrics)

    return (
        bm_loss, bm_metrics,
        layout.carried_state(nash_policy_outputs.hidden_states, frames),
        nash_policy_chain)

  def _nash_policy_qs(
      self,
      q_function: q_lib.QFunction[Action],
      frames: Frames[nash_data.Rank3, Action],  # [T + 1, B, 2]
      policy_samples: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      q_outputs: QFunctionOutputs,
      nash_policy_chain: nash_utils.Chain[Action],  # leaves [T', B, 2]
      nash_solution: nash.NashVariables,  # [N, T', B]
  ) -> Metrics:  # [B]
    """Diagnostics of the nash_policy's sampled chain against the nash
    distributions (chain_game.ChainGame.nash_policy_qs)."""
    metrics = self.game.nash_policy_qs(
        q_function, frames, policy_samples, q_outputs, nash_policy_chain,
        nash_solution)
    return utils.map_single_structure(lambda x: jnp.mean(x, axis=0), metrics)

  def _unroll_kl_teacher_weights(
    self,
    kl_teacher_weights: jax_utils.KLTeacherWeights,
    teacher_kl: jax.Array,
    reverse_teacher_kl: jax.Array,
  ):
    fwd_weight = kl_teacher_weights.fwd_weight()
    bwd_weight = kl_teacher_weights.bwd_weight()

    # High weight lowers the KL, so if the KL is high, we want to increase the weight.
    fwd_loss = -fwd_weight * (jnp.mean(teacher_kl) - self.config.target_teacher_kl)
    bwd_loss = -bwd_weight * (jnp.mean(reverse_teacher_kl) - self.config.target_reverse_teacher_kl)
    total_loss = fwd_loss + bwd_loss

    metrics = dict(
        fwd_weight=fwd_weight,
        bwd_weight=bwd_weight,
        fwd_loss=fwd_loss,
        bwd_loss=bwd_loss,
        total_loss=total_loss,
    )

    return total_loss, metrics

  @jax_utils.annotate_function
  def step_sample_policy(
      self,
      tm_frames: Frames[nash_data.Rank3, Action],  # [T + 1, B, 2]
      initial_state: RecurrentState,
  ):
    return self.run_sample_policy(tm_frames, initial_state)

  @jax_utils.annotate_function
  def step_q_function(
      self,
      tm_frames: Frames[nash_data.Rank3, Action],  # [T + 1, B, 2]
      initial_state: RecurrentState,
      policy_samples: nash_utils.Chain[Action],
      train: bool,
  ) -> QFunctionOutputs:
    fn = self.train_q_function if train else self.run_q_function
    lambda_ = self.config.gae_lambda if train else 1.0
    outputs = fn(tm_frames, initial_state, policy_samples, lambda_)
    return QFunctionOutputs(*outputs)

  @jax_utils.annotate_function
  def step_nash_policy(
      self,
      tm_frames: Frames[nash_data.Rank3, Action],  # [T + 1, B, 2]
      initial_state: RecurrentState,
      policy_samples: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      q_values: jax.Array,  # [N, S, S, T', B, 2]
      nash_solution: nash.NashVariables,  # [N, T', B]
      teacher_core_outputs: jax.Array,  # [T', B, 2, O]
      actor_logits: list[Action],  # FS x [T + 1, B, 2]
      train: bool = True,
  ):
    fn = self.train_nash_policy if train else self.run_nash_policy

    return fn(
        tm_frames, initial_state, policy_samples, q_values, nash_solution,
        teacher_core_outputs, actor_logits,
    )

  @jax_utils.annotate_function
  def step_nash_policy_qs(
      self,
      tm_frames: Frames[nash_data.Rank3, Action],  # [T + 1, B, 2]
      policy_samples: nash_utils.Chain[Action],  # leaves [S, T', B, 2]
      q_outputs: QFunctionOutputs,
      nash_policy_chain: nash_utils.Chain[Action],  # leaves [T', B, 2]
      nash_solution: nash.NashVariables,  # [N, T', B]
  ) -> Metrics:
    return self.run_nash_policy_qs(
        tm_frames, policy_samples, q_outputs, nash_policy_chain, nash_solution)

  def step(
      self,
      trajectory: FrameSkipTrajectory[Action],  # [T, B, 2], on device
      initial_states: dict[str, RecurrentState],  # [B, 2]
      step: int,
      train: bool = True,
  ) -> tuple[dict, RecurrentState]:
    # Pairs each state with the actions the actor had committed when it saw
    # that state (see get_delayed_frames), so every unroll below sees exactly
    # what the actor saw. The prefix rewards are kept so that chains with
    # different prefixes are scored on the rewards those prefixes earn; every
    # one of the T rewards then pairs with a state. The actor logits come out
    # aligned with the policy outputs.
    so_frames = get_delayed_frames(
        trajectory, self.skip_delay, keep_prefix_rewards=True)
    frames = from_so_frames(so_frames)
    actor_logits = [so.logits for so in so_frames.state_action.action]

    final_states = dict(initial_states)
    metrics = {}

    (
      metrics[SAMPLE_POLICY],
      final_states[SAMPLE_POLICY],
      policy_samples,
    ) = self.step_sample_policy(frames, initial_states[SAMPLE_POLICY])

    q_outputs = self.step_q_function(
        frames, initial_states[Q_FUNCTION], policy_samples, train=train)
    metrics[Q_FUNCTION] = q_outputs.metrics
    final_states[Q_FUNCTION] = q_outputs.final_state

    (
      nash_variables,
      metrics[NASH],
    ) = self.compute_nash(q_outputs.q_values)

    (
      teacher_core_outputs,
      final_states[TEACHER],
    ) = self.run_teacher(frames, initial_states[TEACHER])

    # Need to make a copy since the original one gets donated
    initial_nash_state = copy_struct(initial_states[NASH_POLICY])
    (
      metrics[NASH_POLICY],
      final_states[NASH_POLICY],
      nash_policy_chain,
    ) = self.step_nash_policy(
        frames, initial_states[NASH_POLICY], policy_samples,
        q_outputs.q_values, nash_variables, teacher_core_outputs,
        actor_logits, train=train)

    metrics[NASH_POLICY].update(self.step_nash_policy_qs(
        frames, policy_samples, q_outputs, nash_policy_chain, nash_variables))

    post_update_metrics = self.post_update(
        frames, initial_nash_state, actor_logits)
    metrics[NASH_POLICY].update(post_update_metrics)

    for path, value in jax.tree.leaves_with_path(jax.device_get(post_update_metrics)):
      if np.any(np.isnan(value)):
        raise ValueError(f'NaN in post_update_metrics at {path}')

    if train:
      metrics['kl_teacher_weights'] = self.train_kl_teacher_weights(
          teacher_kl=metrics[NASH_POLICY]['teacher_kl'],
          reverse_teacher_kl=metrics[NASH_POLICY]['reverse_teacher_kl'],
      )[0]

    if train and step % self.config.epoch_length == 0:
      self._update_policy()

    return metrics, final_states
