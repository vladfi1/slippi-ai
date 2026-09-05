import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx

from slippi_ai import data, utils
from slippi_ai.jax import rl_lib
from slippi_ai.jax.networks import RecurrentState
from slippi_ai.jax import networks, jax_utils
from slippi_ai.jax import embed as embed_lib
from slippi_ai.jax import epinet as epinet_lib
from slippi_ai.types import Controller, Action

class QOutputs(tp.NamedTuple):
  returns: jax.Array  # [N, T, B, 2]
  advantages: jax.Array  # [N, T, B, 2]
  values: jax.Array  # [N, T, B, 2]
  q_values: jax.Array  # [N, T, B, 2]
  loss: jax.Array  # [T, B, 2]
  # hidden_states: RecurrentState  # [T, B]
  metrics: dict  # [T, B, 2]

class UnrollOutputs(tp.NamedTuple):
  values: jax.Array  # [N, T, B, 2]
  q_values: jax.Array  # [N, T, B, 2]

# Rank2 = tuple[int, int]
Rank3 = tuple[int, int, int]

def to_merged_outputs(outputs: jax.Array) -> jax.Array:
  """Goes from [..., 2, O] to [..., 2, 2O]."""
  *batch, n, o = outputs.shape
  assert n == 2
  # return jnp.reshape(
  #   jnp.stack([outputs, jnp.flip(outputs, axis=2)], axis=2),
  #   (t, b, 2, 2 * o))
  shape = (*batch, n * o)
  p0_outputs = jnp.reshape(outputs, shape)
  p1_outputs = jnp.reshape(jnp.flip(outputs, axis=-2), shape)
  return jnp.stack([p0_outputs, p1_outputs], axis=-2)

@dataclasses.dataclass
class HeadConfig:
  num_layers: int = 1
  hidden_size: int = 128
  epinet: epinet_lib.EpinetConfig = dataclasses.field(
      default_factory=epinet_lib.EpinetConfig)

@dataclasses.dataclass
class QFunctionConfig:
  embed: embed_lib.EmbedConfig = dataclasses.field(default_factory=embed_lib.EmbedConfig)
  num_names: int = 16
  core_net: dict = dataclasses.field(default_factory=networks.default_config)
  # Separate config for the action RNN (and its state initializer).
  # Decoupled from the core_net so they can have different architectures.
  action_net: dict = dataclasses.field(default_factory=networks.default_config)
  head: HeadConfig = dataclasses.field(default_factory=HeadConfig)

  advantage_qs: bool = True  # Have q-head predict advantages
  frame_skip: int = 1  # Number of actions per frame-skip step

class QFunction(nnx.Module, tp.Generic[Action]):
  """Two-player Q-function.

  Currently each player's trajectory is processed separately by the same
  recurrent network, and then merged for the final value and q heads. We should
  instead merge at the beginning.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      config: QFunctionConfig,
      embed_action: embed_lib.Embedding[Controller, Action],
      frame_skip: int,
  ):
    self.config = config
    self.embed_action = embed_action
    self.frame_skip = frame_skip
    self.core_net = networks.build_embed_network(
        rngs, config.embed, config.num_names, config.core_net,
        frame_skip=frame_skip, embed_action=self.embed_action)
    # Takes core_net output and produces the initial state for action_rnn.
    self.action_init = networks.construct_network(
        rngs, input_size=self.core_net.output_size, **config.action_net)
    # Unrolled over the frame_skip actions to compute Q-values.
    self.action_net = networks.construct_network(
        rngs, input_size=self.embed_action.size, **config.action_net)

    self.value_head = jax_utils.MLP(
      rngs=rngs,
      input_size=2 * self.core_net.output_size,
      features=[config.head.hidden_size] * config.head.num_layers + [1],
      activate_final=False,
    )
    self.q_head = jax_utils.MLP(
      rngs=rngs,
      input_size=2 * self.action_net.output_size,
      features=[config.head.hidden_size] * config.head.num_layers + [1],
      activate_final=False,
    )

    # The heads consume the merged (both-player) features, so the epinets do
    # too; the epistemic index is shared between the two players, as it indexes
    # a hypothesis about the single underlying q-function.
    self.value_epinet = epinet_lib.Epinet(
        rngs, 2 * self.core_net.output_size, 1, config.head.epinet)
    self.q_epinet = epinet_lib.Epinet(
        rngs, 2 * self.action_net.output_size, 1, config.head.epinet)

  def sample_index(
      self,
      rngs: nnx.Rngs,
      batch_shape: tuple[int, ...],
  ) -> jax.Array:
    """Samples epistemic indices z ~ N(0, I).

    Callers pass a batch_shape with a trailing 1 for the player axis, e.g.
    [N, B, 1], so that z broadcasts over both players and all timesteps.
    """
    z = jax.random.normal(
        rngs.epinet(), batch_shape + (self.config.head.epinet.index_dim,))
    return z.astype(jax_utils.module_dtype(self))

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> networks.RecurrentState:
    states = [self.core_net.initial_state(batch_size, rngs) for _ in range(2)]
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *states)

  def _embed_action(self, action: Action) -> jax.Array:
    x = self.embed_action(action)
    return x.astype(jax_utils.module_dtype(self))

  def _values_from_outputs(
      self,
      outputs: jax.Array,  # [..., 2, O_core]
      z: jax.Array,  # [..., D_Z]
  ) -> jax.Array:  # [..., 2]
    merged_outputs = to_merged_outputs(outputs)
    values = self.value_head(merged_outputs) + self.value_epinet(merged_outputs, z)
    return jnp.squeeze(values, -1)

  def _q_values_from_outputs(
      self,
      outputs: jax.Array,  # [..., 2, O_action]
      values: jax.Array,  # [..., 2]
      z: jax.Array,  # [..., D_Z]
  ) -> jax.Array:  # [..., 2]
    merged_outputs = to_merged_outputs(outputs)
    qs = self.q_head(merged_outputs) + self.q_epinet(merged_outputs, z)
    qs = jnp.squeeze(qs, -1)

    if self.config.advantage_qs:
      return values + qs
    else:
      return qs

  def _action_net_initial_state(
      self, core_outputs: jax.Array,  # [..., 2, O_core]
  ) -> RecurrentState:
    """Projects core_net outputs to an initial state for the action_net."""
    batch_shape = core_outputs.shape[:-1]
    zero_state = self.action_init.initial_state(batch_shape, rngs=nnx.Rngs(0))
    _, init_state = self.action_init.step(core_outputs, zero_state)
    return init_state

  def q_values_from_action_state(
      self,
      values: jax.Array,  # [..., 2]
      action_init_state: networks.RecurrentState,  # [..., 2, H]
      actions: list[Action],  # frame_skip x [..., 2]
      z: jax.Array,  # [..., D_Z]
  ) -> jax.Array:  # [..., 2]
    embedded = [self._embed_action(a) for a in actions]
    stacked = jnp.stack(embedded, axis=0)  # [FS, ..., embed_size]
    reset = jnp.zeros(stacked.shape[:-1], dtype=bool)  # [FS, ...]
    outputs, _ = self.action_net.unroll(stacked, reset, action_init_state)
    return self._q_values_from_outputs(outputs[-1], values, z)

  def q_values_from_core_outputs(
      self,
      values: jax.Array,  # [..., 2]
      core_outputs: jax.Array,  # [..., 2, O_core]
      actions: list[Action],  # frame_skip x [..., 2]
      z: jax.Array,  # [..., D_Z]
  ) -> jax.Array:  # [..., 2]
    init_state = self._action_net_initial_state(core_outputs)
    return self.q_values_from_action_state(values, init_state, actions, z)

  def _indexed_values_from_outputs(
      self,
      outputs: jax.Array,  # [T, B, 2, O_core]
      zs: jax.Array,  # [N, B, 1, D_Z]
  ) -> jax.Array:  # [N, T, B, 2]
    """Per-index values by broadcasting; the base value_head runs only once."""
    merged_outputs = to_merged_outputs(outputs)
    values = self.value_head(merged_outputs)  # [T, B, 2, 1], shared across indices
    zs = jnp.expand_dims(zs, 1)  # [N, 1, B, 1, D_Z]: broadcasts over T and players
    values = values + self.value_epinet(merged_outputs, zs)  # [N, T, B, 2, 1]
    return jnp.squeeze(values, -1)

  def _indexed_q_values_from_outputs(
      self,
      outputs: jax.Array,  # [T, B, 2, O_action]
      values: jax.Array,  # [N, T, B, 2] per-index values
      zs: jax.Array,  # [N, B, 1, D_Z]
  ) -> jax.Array:  # [N, T, B, 2]
    """Per-index q-values by broadcasting; the base q_head runs only once.

    Only the (small) epinet is evaluated per index -- its input genuinely
    depends on z -- while the base head's output is shared across indices.
    """
    merged_outputs = to_merged_outputs(outputs)
    qs = self.q_head(merged_outputs)  # [T, B, 2, 1], shared across indices
    zs = jnp.expand_dims(zs, 1)  # [N, 1, B, 1, D_Z]: broadcasts over T and players
    qs = qs + self.q_epinet(merged_outputs, zs)  # [N, T, B, 2, 1]
    qs = jnp.squeeze(qs, -1)

    if self.config.advantage_qs:
      return values + qs
    else:
      return qs

  def indexed_q_values_from_action_state(
      self,
      values: jax.Array,  # [N, T, B, 2] per-index values
      action_init_state: networks.RecurrentState,  # [T, B, 2, H]
      actions: list[Action],  # frame_skip x [T, B, 2]
      zs: jax.Array,  # [N, B, 1, D_Z]
  ) -> jax.Array:  # [N, T, B, 2]
    """Per-index q-values; the action_net is unrolled once, shared across indices."""
    embedded = [self._embed_action(a) for a in actions]
    stacked = jnp.stack(embedded, axis=0)  # [FS, ..., embed_size]
    reset = jnp.zeros(stacked.shape[:-1], dtype=bool)  # [FS, ...]
    outputs, _ = self.action_net.unroll(stacked, reset, action_init_state)
    return self._indexed_q_values_from_outputs(outputs[-1], values, zs)

  def multi_q_values_from_action_state(
      self,
      values: jax.Array,  # [T, B, 2]
      action_init_state: networks.RecurrentState,  # [T, B, 2, H]
      actions: list[Action],  # frame_skip x [S, T, B, 2]
      z: jax.Array,  # [T, B, D_Z] (or broadcastable)
      batch_size: tp.Optional[int] = 0,  # 0 is equivalent to vmap
  ) -> jax.Array:  # [S, S, T, B, 2]  ([p0-idx, p1-idx, ...])
    embedded = [self._embed_action(a) for a in actions]  # frame_skip x [S, T, B, 2, E]
    action_inputs = jnp.stack(embedded, axis=1)  # [S, FS, T, B, 2, E]

    def process_one_sample(embedded_fs: jax.Array) -> jax.Array:
      # embedded_fs: [FS, T, B, 2, E]
      reset = jnp.zeros(embedded_fs.shape[:-1], dtype=bool)
      outputs, _ = self.action_net.unroll(embedded_fs, reset, action_init_state)
      return outputs[-1]  # [T, B, 2, O]

    action_outputs = jax_utils.lax_map(
        process_one_sample,
        action_inputs,
        batch_size=batch_size,
    )
    p0_outputs, p1_outputs = jnp.unstack(action_outputs, axis=-2)  # [S, T, B, O]

    num_samples = action_outputs.shape[0]

    if batch_size is None:
      inner_bs = None
      outer_bs = None
    elif batch_size == 0:
      inner_bs = 0
      outer_bs = 0
    elif batch_size <= num_samples:
      inner_bs = batch_size
      outer_bs = None
    else:
      inner_bs = 0
      outer_bs = batch_size // num_samples

    def p0_fn(p0_outputs: jax.Array):
      def p1_fn(p1_outputs: jax.Array):
        outputs = jnp.stack([p0_outputs, p1_outputs], axis=-2)  # [T, B, 2, O]
        return self._q_values_from_outputs(outputs, values, z)

      return jax_utils.lax_map(p1_fn, p1_outputs, batch_size=inner_bs)

    return jax_utils.lax_map(p0_fn, p0_outputs, batch_size=outer_bs)

  def multi_index_q_values_from_action_state(
      self,
      values: jax.Array,  # [N, T, B, 2] per-index values
      action_init_state: networks.RecurrentState,  # [T, B, 2, H]
      actions: list[Action],  # frame_skip x [S, T, B, 2]
      zs: jax.Array,  # [N, B, 1, D_Z]
      batch_size: tp.Optional[int] = 0,  # 0 is equivalent to vmap
  ) -> jax.Array:  # [N, S, S, T, B, 2]  ([index, p0-idx, p1-idx, ...])
    """Like multi_q_values_from_action_state, but per epistemic index.

    The action_net is unrolled once per sample and shared across indices; only
    the q-head is evaluated per index. The values must be the per-index values
    computed with the same zs (e.g. from loss_and_action_state).
    """
    embedded = [self._embed_action(a) for a in actions]  # frame_skip x [S, T, B, 2, E]
    action_inputs = jnp.stack(embedded, axis=1)  # [S, FS, T, B, 2, E]

    def process_one_sample(embedded_fs: jax.Array) -> jax.Array:
      # embedded_fs: [FS, T, B, 2, E]
      reset = jnp.zeros(embedded_fs.shape[:-1], dtype=bool)
      outputs, _ = self.action_net.unroll(embedded_fs, reset, action_init_state)
      return outputs[-1]  # [T, B, 2, O]

    action_outputs = jax_utils.lax_map(
        process_one_sample,
        action_inputs,
        batch_size=batch_size,
    )
    p0_outputs, p1_outputs = jnp.unstack(action_outputs, axis=-2)  # [S, T, B, O]

    num_samples = action_outputs.shape[0]

    if batch_size is None:
      inner_bs = None
      outer_bs = None
    elif batch_size == 0:
      inner_bs = 0
      outer_bs = 0
    elif batch_size <= num_samples:
      inner_bs = batch_size
      outer_bs = None
    else:
      inner_bs = 0
      outer_bs = batch_size // num_samples

    def p0_fn(p0_outputs: jax.Array):
      def p1_fn(p1_outputs: jax.Array):
        outputs = jnp.stack([p0_outputs, p1_outputs], axis=-2)  # [T, B, 2, O]
        return self._indexed_q_values_from_outputs(outputs, values, zs)  # [N, T, B, 2]

      return jax_utils.lax_map(p1_fn, p1_outputs, batch_size=inner_bs)

    q_values = jax_utils.lax_map(
        p0_fn, p0_outputs, batch_size=outer_bs)  # [S, S, N, T, B, 2]
    return jnp.moveaxis(q_values, 2, 0)  # [N, S, S, T, B, 2]

  def unroll(
      self,
      state_action: data.StateAction[Rank3, Action],  # [T, B, 2]
      is_resetting: jax.Array,  # [T, B, 2]
      next_actions: list[Action],  # frame_skip x [T, B, 2]
      initial_state: RecurrentState,  #  [B, 2]
      zs: jax.Array,  # [N, B, 1, D_Z]
  ) -> tuple[UnrollOutputs, RecurrentState]:
    """Outputs have shape [N, T, B, 2], one per epistemic index."""
    core_outputs, final_state = self.core_net.unroll(
        state_action, is_resetting, initial_state)

    # The core_net, action_net, and base heads don't depend on the epistemic
    # index, so they are evaluated once; only the epinets run per index.
    values = self._indexed_values_from_outputs(core_outputs, zs)  # [N, T, B, 2]

    init_state = self._action_net_initial_state(core_outputs)
    embedded = [self._embed_action(a) for a in next_actions]
    stacked = jnp.stack(embedded, axis=0)  # [FS, T, B, 2, E]
    reset = jnp.zeros(stacked.shape[:-1], dtype=bool)
    action_outputs, _ = self.action_net.unroll(stacked, reset, init_state)
    q_values = self._indexed_q_values_from_outputs(
        action_outputs[-1], values, zs)  # [N, T, B, 2]

    return UnrollOutputs(values=values, q_values=q_values), final_state

  def loss_batched(
      self,
      frames: data.Frames[Rank3, Action],  # [T + 1, B, 2]
      initial_state: RecurrentState,  # [B, 2]
      discount: float,
      batch_size: int,  # batch size in time
      rngs: nnx.Rngs,  # for sampling epistemic indices
      lambda_: float = 1.0,
      eval_lambdas: list[float] = [0],
      num_index_samples: int = 1,
  ) -> tp.Tuple[QOutputs, RecurrentState]:
    total_unroll_length = frames.reward.shape[0]  # T
    num_batches, r = divmod(total_unroll_length, batch_size)
    if r != 0:
      raise ValueError(f'Unroll length {total_unroll_length} is not divisible by batch size {batch_size}.')

    def to_batched(x: jax.Array) -> jax.Array:
      assert x.shape[0] == total_unroll_length
      return x.reshape((num_batches, batch_size) + x.shape[1:])

    state_action, is_resetting = utils.map_nt(
        lambda x: to_batched(x[:-1]),
        (frames.state_action, frames.is_resetting))
    next_actions = utils.map_nt(
        lambda x: to_batched(x[1:]),
        frames.state_action.action)

    # Epistemic indices are sampled once per batch element and shared across
    # the whole unroll (and both players), including the bootstrap value, so
    # that each index regresses to its own self-consistent targets.
    zs = self.sample_index(
        rngs, (num_index_samples, frames.reward.shape[1], 1))

    time_axis = 1  # outputs are [N, T, B, 2]

    # nnx will complain about trace levels if we use jax.lax.scan
    scan_fn = nnx.scan(
        nnx.remat(QFunction[Action].unroll),
        in_axes=(None, 0, 0, 0, nnx.Carry, None),
        out_axes=(time_axis, nnx.Carry),
    )

    unroll_outputs, final_state = scan_fn(
        self, state_action, is_resetting, next_actions, initial_state, zs)

    # Reshape outputs back to [N, T, B, 2]
    def to_unbatched(x: jax.Array) -> jax.Array:
      assert x.shape[time_axis] == num_batches
      assert x.shape[time_axis + 1] == batch_size
      return x.reshape(
          x.shape[:time_axis] + (total_unroll_length,)
          + x.shape[time_axis + 2:])

    unroll_outputs = utils.map_nt(to_unbatched, unroll_outputs)
    values, q_values = unroll_outputs

    last_state_action, last_is_resetting = utils.map_nt(
        lambda x: x[-1], (frames.state_action, frames.is_resetting))
    last_output, _ = self.core_net.step_with_reset(
        last_state_action, last_is_resetting, final_state)

    # With no time axis, zs [N, B, 1, D_Z] broadcasts directly against the
    # [B, 2] batch shape, giving per-index values with one base-head pass.
    last_value = self._values_from_outputs(last_output, zs)  # [N, B, 2]

    outputs = self._ensemble_outputs(
        frames, values, q_values, last_value, discount, lambda_)

    for eval_lambda in eval_lambdas:
      eval_outputs = self._ensemble_outputs(
          frames, values, q_values, last_value, discount, eval_lambda)
      outputs.metrics[f'lambda_{eval_lambda:.1f}'] = eval_outputs.metrics

    return outputs, final_state

  def loss_and_action_state(
      self,
      frames: data.Frames[Rank3, Action],  # [T + 1, B, 2]
      initial_state: RecurrentState,
      discount: float,
      rngs: nnx.Rngs,  # for sampling epistemic indices
      lambda_: float = 1.0,
      num_index_samples: int = 1,
  ) -> tp.Tuple[QOutputs, RecurrentState, RecurrentState, jax.Array]:
    """Returns (q_outputs, action_init_state, final_state, zs).

    action_init_state has shape [T, B, 2, H] (action_net initial state per step).
    final_state is the core_net recurrent state after the last frame.
    zs are the sampled epistemic indices, [N, B, 1, D_Z]; the returned
    per-index values in q_outputs pair with them, so callers computing further
    q-values from action_init_state must reuse these same indices.
    """
    state_action_T = utils.map_nt(lambda x: x[:-1], frames.state_action)
    core_outputs, final_state = self.core_net.unroll(
        state_action_T, frames.is_resetting[:-1], initial_state)

    # The epistemic indices are shared across the whole unroll (and both
    # players), including the bootstrap value, so that each index regresses to
    # its own self-consistent targets.
    zs = self.sample_index(
        rngs, (num_index_samples, frames.reward.shape[1], 1))

    values = self._indexed_values_from_outputs(core_outputs, zs)  # [N, T, B, 2]

    last_output, _ = self.core_net.step_with_reset(
        utils.map_nt(lambda x: x[-1], frames.state_action),
        frames.is_resetting[-1], final_state)
    # With no time axis, zs [N, B, 1, D_Z] broadcasts directly against the
    # [B, 2] batch shape, giving per-index values with one base-head pass.
    last_value = self._values_from_outputs(last_output, zs)  # [N, B, 2]

    action_init_state = self._action_net_initial_state(core_outputs)

    next_actions = jax.tree.map(
        lambda t: t[1:], frames.state_action.action)
    q_values = self.indexed_q_values_from_action_state(
        values, action_init_state, next_actions, zs)  # [N, T, B, 2]

    outputs = self._ensemble_outputs(
        frames, values, q_values, last_value, discount, lambda_)

    return outputs, action_init_state, final_state, zs

  def _ensemble_outputs(
      self,
      frames: data.Frames[Rank3, Action],
      values: jax.Array,  # [N, T, B, 2]
      q_values: jax.Array,  # [N, T, B, 2]
      last_value: jax.Array,  # [N, B, 2]
      discount: float,
      lambda_: float,
  ) -> QOutputs:
    """Combines per-index predictions into ensemble QOutputs.

    The loss and metrics are averaged over the sampled indices, while the
    mean prediction over indices is evaluated as an ensemble; its metrics
    (not trained on) are the "regular" ones, so as to be comparable with
    previous non-epinet runs. The returned returns/advantages/values/q_values
    are per-index.
    """
    outputs = nnx.vmap(
        QFunction[Action]._get_outputs,
        in_axes=(None, None, 0, 0, 0, None, None),
    )(self, frames, values, q_values, last_value, discount, lambda_)
    indexed_metrics = jax.tree.map(
        lambda x: jnp.mean(x, axis=0), outputs.metrics)

    ensemble_outputs = self._get_outputs(
        frames, jnp.mean(values, axis=0), jnp.mean(q_values, axis=0),
        jnp.mean(last_value, axis=0), discount, lambda_)
    metrics = ensemble_outputs.metrics

    for name, vs in [('v', values), ('q', q_values)]:
      indexed_metrics[name].update(epinet_lib.epistemic_metrics(vs))

    advantages = q_values - values
    indexed_metrics['advantage'] = dict(
        mean=jnp.mean(advantages, axis=0),
        **epinet_lib.epistemic_metrics(advantages),
    )

    metrics['indexed'] = indexed_metrics
    return outputs._replace(
        loss=jnp.mean(outputs.loss, axis=0),  # [T, B, 2]
        metrics=metrics,
    )

  def _get_outputs(
      self,
      frames: data.Frames[Rank3, Action],
      values: jax.Array,
      q_values: jax.Array,
      last_value: jax.Array,
      discount: float,
      lambda_: float,
  ):
    value_targets = rl_lib.generalized_returns_with_resetting(
        rewards=frames.reward,
        values=values,
        is_resetting=frames.is_resetting[1:],
        bootstrap=last_value,
        discount=discount,
        lambda_=lambda_,
    )
    value_targets = jax.lax.stop_gradient(value_targets)

    advantages = value_targets - values
    value_loss = jnp.square(advantages)

    _, value_variance = jax_utils.mean_and_variance(value_targets)
    uev = value_loss / (value_variance + 1e-8)

    q_loss = jnp.square(value_targets - q_values)
    quev = q_loss / (value_variance + 1e-8)
    uev_delta = uev - quev

    metrics = {
        'reward': {
            'mean': frames.reward,
            'std': jnp.std(frames.reward, keepdims=True),
        },
        'v': {
            'loss': value_loss,
            'uev': uev,
        },
        'q': {
            'loss': q_loss,
            'uev': quev,
            'uev_delta': uev_delta,
            # Take log to result in a geometric mean.
            'rel_v_loss': jnp.log((value_loss + 1e-8) / (q_loss + 1e-8)),
        },
    }

    return QOutputs(
        returns=value_targets,
        advantages=advantages,
        values=values,
        q_values=q_values,
        loss=value_loss + q_loss,
        # hidden_states=hidden_states,
        metrics=metrics,
    )

def build_q_function(rngs: nnx.Rngs, config: QFunctionConfig) -> QFunction[tp.Any]:
  embed_action = config.embed.controller.make_embedding()
  return QFunction(rngs, config, embed_action, config.frame_skip)
