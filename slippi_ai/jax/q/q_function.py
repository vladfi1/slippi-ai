import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx

from slippi_ai import data, utils, flag_utils
from slippi_ai.jax import rl_lib
from slippi_ai.jax.networks import RecurrentState
from slippi_ai.jax import networks, jax_utils
from slippi_ai.jax import embed as embed_lib
from slippi_ai.jax import epinet as epinet_lib
from slippi_ai.types import Controller, Action

class QOutputs(tp.NamedTuple):
  returns: jax.Array  # [N, T, B]
  advantages: jax.Array  # [N, T, B]
  values: jax.Array  # [N, T, B]
  q_values: jax.Array  # [N, T, B]
  loss: jax.Array  # [T, B]
  # hidden_states: RecurrentState  # [T, B]
  metrics: dict  # [T, B]

class UnrollOutputs(tp.NamedTuple):
  values: jax.Array  # [N, T, B]
  q_values: jax.Array  # [N, T, B]

Rank2 = tuple[int, int]

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
  """Single-player Q-function with frame-skip support.

  The core_net processes the (frame-skipped) state/action trajectory and
  produces a value estimate per step. To compute the Q-value of taking a
  particular sequence of `frame_skip` actions, a separate action_net is
  initialized from the core_net's output (via action_init) and unrolled over
  the frame_skip actions; the final output feeds the q-head.
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
    # Takes core_net output and produces the initial state for action_net.
    self.action_init = networks.construct_network(
        rngs, input_size=self.core_net.output_size, **config.action_net)
    # Unrolled over the frame_skip actions to compute Q-values.
    self.action_net = networks.construct_network(
        rngs, input_size=self.embed_action.size, **config.action_net)

    self.value_head = jax_utils.MLP(
      rngs=rngs,
      input_size=self.core_net.output_size,
      features=[config.head.hidden_size] * config.head.num_layers + [1],
      activate_final=False,
    )
    self.q_head = jax_utils.MLP(
      rngs=rngs,
      input_size=self.action_net.output_size,
      features=[config.head.hidden_size] * config.head.num_layers + [1],
      activate_final=False,
    )

    self.value_epinet = epinet_lib.Epinet(
        rngs, self.core_net.output_size, 1, config.head.epinet)
    self.q_epinet = epinet_lib.Epinet(
        rngs, self.action_net.output_size, 1, config.head.epinet)

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> networks.RecurrentState:
    return self.core_net.initial_state(batch_size, rngs)

  def _embed_action(self, action: Action) -> jax.Array:
    x = self.embed_action(action)
    return x.astype(jax_utils.module_dtype(self))

  def sample_index(
      self,
      rngs: nnx.Rngs,
      batch_shape: tuple[int, ...],
  ) -> jax.Array:
    """Samples epistemic indices z ~ N(0, I)."""
    z = jax.random.normal(
        rngs.epinet(), batch_shape + (self.config.head.epinet.index_dim,))
    return z.astype(jax_utils.module_dtype(self))

  def _values_from_outputs(
      self,
      outputs: jax.Array,  # [..., O_core]
      z: jax.Array,  # [..., D_Z]
  ) -> jax.Array:
    """[..., O_core] -> [...]"""
    values = self.value_head(outputs) + self.value_epinet(outputs, z)
    return jnp.squeeze(values, -1)

  def _q_values_from_outputs(
      self,
      outputs: jax.Array,  # [..., O_action]
      values: jax.Array,  # [...]
      z: jax.Array,  # [..., D_Z]
  ) -> jax.Array:  # [...]
    qs = self.q_head(outputs) + self.q_epinet(outputs, z)
    qs = jnp.squeeze(qs, -1)

    if self.config.advantage_qs:
      return values + qs
    else:
      return qs

  def _action_net_initial_state(
      self, core_outputs: jax.Array,  # [..., O_core]
  ) -> RecurrentState:
    """Projects core_net outputs to an initial state for the action_net."""
    batch_shape = core_outputs.shape[:-1]
    zero_state = self.action_init.initial_state(batch_shape, rngs=nnx.Rngs(0))
    _, init_state = self.action_init.step(core_outputs, zero_state)
    return init_state

  def q_values_from_core_outputs(
      self,
      core_outputs: jax.Array,  # [T, B, O_core]
      actions: list[Action],  # frame_skip x [...]
      rngs: nnx.Rngs,
      num_index_samples: int,
  ) -> jax.Array:  # [N, ...]
    """Per-index q-values; the action_net is unrolled once, shared across indices."""
    embedded = [self._embed_action(a) for a in actions]
    stacked = jnp.stack(embedded, axis=0)  # [FS, ..., embed_size]
    reset = jnp.zeros(stacked.shape[:-1], dtype=bool)  # [FS, ...]
    action_init_state = self._action_net_initial_state(core_outputs)
    outputs, _ = self.action_net.unroll(stacked, reset, action_init_state)

    zs = self.sample_index(
        rngs, (num_index_samples, core_outputs.shape[1]))
    values = nnx.vmap(
      QFunction._values_from_outputs, in_axes=(None, None, 0),
    )(self, core_outputs, zs)  # [N, T, B]

    return nnx.vmap(
        QFunction._q_values_from_outputs, in_axes=(None, None, 0, 0),
    )(self, outputs[-1], values, zs)

  def multi_index_q_values_from_core_outputs(
      self,
      core_outputs: jax.Array,  # [T, B, O_core]
      actions: list[Action],  # frame_skip x [S, T, B]
      rngs: nnx.Rngs,
      num_index_samples: int,
      batch_size: int = 0,  # 0 is equivalent to vmap
  ) -> jax.Array:  # [N, S, T, B]
    """Per-index q-values for multiple sampled action sequences."""

    zs = self.sample_index(
        rngs, (num_index_samples, core_outputs.shape[1]))
    values = nnx.vmap(
      QFunction._values_from_outputs, in_axes=(None, None, 0),
    )(self, core_outputs, zs)  # [N, T, B]

    embedded = [self._embed_action(a) for a in actions]  # frame_skip x [S, T, B, E]
    action_inputs = jnp.stack(embedded, axis=1)  # [S, FS, T, B, E]
    action_init_state = self._action_net_initial_state(core_outputs)

    multi_index_head = nnx.vmap(
        QFunction._q_values_from_outputs, in_axes=(None, None, 0, 0))

    def process_one_sample(
        q_function: QFunction[Action],
        embedded_fs: jax.Array,  # [FS, T, B, E]
    ) -> jax.Array:
      reset = jnp.zeros(embedded_fs.shape[:-1], dtype=bool)
      outputs, _ = q_function.action_net.unroll(
          embedded_fs, reset, action_init_state)
      return multi_index_head(q_function, outputs[-1], values, zs)  # [N, T, B]

    process_all_samples = jax_utils.lax_map_fn(
        process_one_sample,
        microbatch_size=batch_size,
        input_batch_dims=(None, 0),
        output_batch_dims=1,
    )

    return process_all_samples(self, action_inputs)  # [N, S, T, B]

  def unroll(
      self,
      state_action: data.StateAction[Rank2, Action],  # [T, B]
      is_resetting: jax.Array,  # [T, B]
      next_actions: list[Action],  # frame_skip x [T, B]
      initial_state: RecurrentState,  # [B]
      zs: jax.Array,  # [N, B, D_Z]
  ) -> tuple[UnrollOutputs, RecurrentState]:
    """Outputs have shape [N, T, B], one per epistemic index."""
    core_outputs, final_state = self.core_net.unroll(
        state_action, is_resetting, initial_state)

    # The core_net and action_net don't depend on the epistemic index, so they
    # are unrolled once; only the heads are evaluated per index.
    values = nnx.vmap(
        QFunction._values_from_outputs, in_axes=(None, None, 0),
    )(self, core_outputs, zs)  # [N, T, B]

    init_state = self._action_net_initial_state(core_outputs)
    embedded = [self._embed_action(a) for a in next_actions]
    stacked = jnp.stack(embedded, axis=0)  # [FS, T, B, E]
    reset = jnp.zeros(stacked.shape[:-1], dtype=bool)
    action_outputs, _ = self.action_net.unroll(stacked, reset, init_state)
    q_values = nnx.vmap(
        QFunction._q_values_from_outputs, in_axes=(None, None, 0, 0),
    )(self, action_outputs[-1], values, zs)  # [N, T, B]

    return UnrollOutputs(values=values, q_values=q_values), final_state

  def loss_batched(
      self,
      frames: data.Frames[Rank2, Action],  # [T + 1, B]
      initial_state: RecurrentState,  # [B]
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
    # the whole unroll, including the bootstrap value, so that each index
    # regresses to its own self-consistent targets.
    zs = self.sample_index(
        rngs, (num_index_samples, frames.reward.shape[1]))

    time_axis = 1  # outputs are [N, T, B]

    # nnx will complain about trace levels if we use jax.lax.scan
    scan_fn = nnx.scan(
        nnx.remat(QFunction[Action].unroll),
        in_axes=(None, 0, 0, 0, nnx.Carry, None),
        out_axes=(time_axis, nnx.Carry),
    )

    unroll_outputs, final_state = scan_fn(
        self, state_action, is_resetting, next_actions, initial_state, zs)

    # Reshape outputs back to [N, T, B]
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

    last_value = nnx.vmap(
        QFunction._values_from_outputs, in_axes=(None, None, 0),
    )(self, last_output, zs)  # [N, B]

    outputs = self._ensemble_outputs(
        frames, values, q_values, last_value, discount, lambda_)
    for eval_lambda in eval_lambdas:
      eval_outputs = self._ensemble_outputs(
          frames, values, q_values, last_value, discount, eval_lambda)
      outputs.metrics[f'lambda_{eval_lambda:.1f}'] = eval_outputs.metrics

    return outputs, final_state

  def loss_and_core_outputs(
      self,
      frames: data.Frames[Rank2, Action],  # [T + 1, B]
      initial_state: RecurrentState,
      rngs: nnx.Rngs,
      num_index_samples: int,
      discount: float,
      lambda_: float = 1.0,
  ) -> tp.Tuple[QOutputs, jax.Array, RecurrentState]:
    """Returns (q_outputs, core_outputs, final_state).

    q_outputs include the epistemic index, except for the loss and metrics
    which are averaged over the epistemic indices; see _ensemble_outputs.
    core_outputs has shape [T, B, O_core].
    final_state is the core_net recurrent state after the last frame.
    """
    state_action_T = utils.map_nt(lambda x: x[:-1], frames.state_action)
    core_outputs, final_state = self.core_net.unroll(
        state_action_T, frames.is_resetting[:-1], initial_state)

    # The epistemic indices are shared across the whole unroll, including the
    # bootstrap value, so that each index regresses to its own targets.
    zs = self.sample_index(
        rngs, (num_index_samples, core_outputs.shape[1]))
    values = nnx.vmap(
        QFunction._values_from_outputs, in_axes=(None, None, 0),
    )(self, core_outputs, zs)  # [N, T, B]

    last_output, _ = self.core_net.step_with_reset(
        utils.map_nt(lambda x: x[-1], frames.state_action),
        frames.is_resetting[-1], final_state)
    last_value = nnx.vmap(
        QFunction._values_from_outputs, in_axes=(None, None, 0),
    )(self, last_output, zs)  # [N, B]

    action_init_state = self._action_net_initial_state(core_outputs)

    next_actions = jax.tree.map(
        lambda t: t[1:], frames.state_action.action)

    # Like q_values_from_action_state, but pairing each index's q-head with
    # its own value estimate.
    embedded = [self._embed_action(a) for a in next_actions]
    stacked = jnp.stack(embedded, axis=0)  # [FS, T, B, E]
    reset = jnp.zeros(stacked.shape[:-1], dtype=bool)
    action_outputs, _ = self.action_net.unroll(stacked, reset, action_init_state)
    q_values = nnx.vmap(
        QFunction._q_values_from_outputs, in_axes=(None, None, 0, 0),
    )(self, action_outputs[-1], values, zs)  # [N, T, B]

    outputs = self._ensemble_outputs(
        frames, values, q_values, last_value, discount, lambda_)

    return outputs, core_outputs, final_state

  def _ensemble_outputs(
      self,
      frames: data.Frames[Rank2, Action],
      values: jax.Array,  # [N, T, B]
      q_values: jax.Array,  # [N, T, B]
      last_value: jax.Array,  # [N, B]
      discount: float,
      lambda_: float,
  ) -> QOutputs:
    """Combines per-index predictions into ensemble QOutputs.

    The loss and metrics are averaged over the sampled indices, while the
    mean prediction over indices is evaluated as an ensemble; its metrics
    (not trained on) are reported under 'ensemble'. The returned
    returns/advantages/values/q_values are per-index.
    """
    outputs = nnx.vmap(
        QFunction[Action]._get_outputs,
        in_axes=(None, None, 0, 0, 0, None, None),
    )(self, frames, values, q_values, last_value, discount, lambda_)
    ensemble_outputs = self._get_outputs(
        frames, jnp.mean(values, axis=0), jnp.mean(q_values, axis=0),
        jnp.mean(last_value, axis=0), discount, lambda_)
    metrics = jax.tree.map(
        lambda x: jnp.mean(x, axis=0), outputs.metrics)
    metrics['ensemble'] = ensemble_outputs.metrics
    return outputs._replace(
        loss=jnp.mean(outputs.loss, axis=0),
        metrics=metrics,
    )

  def _get_outputs(
      self,
      frames: data.Frames[Rank2, Action],
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

def q_function_from_config(config: dict) -> QFunction[tp.Any]:
  q_config = flag_utils.dataclass_from_dict(QFunctionConfig, config)
  return build_q_function(nnx.Rngs(0), q_config)

class FakeQFunction:

  def initial_state(self, batch_size: int) -> RecurrentState:
    del batch_size
    return ()

  def loss(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
      discount: float,
  ) -> tp.Tuple[QOutputs, RecurrentState]:
    del discount

    returns = jnp.zeros_like(frames.reward)

    outputs = QOutputs(
        returns=returns,
        values=returns,
        q_values=returns,
        loss=returns,
        advantages=returns,
        # hidden_states=(),
        metrics={},
    )

    return outputs, initial_state
