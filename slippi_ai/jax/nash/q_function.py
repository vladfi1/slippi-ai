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
from slippi_ai.types import Controller, Action

class QOutputs(tp.NamedTuple):
  returns: jax.Array  # [T, B]
  advantages: jax.Array  # [T, B]
  values: jax.Array  # [T, B]
  q_values: jax.Array  # [T, B]
  loss: jax.Array
  # hidden_states: RecurrentState  # [T, B]
  metrics: dict

class UnrollOutputs(tp.NamedTuple):
  values: jax.Array  # [T, B]
  q_values: jax.Array  # [T, B]

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

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> networks.RecurrentState:
    states = [self.core_net.initial_state(batch_size, rngs) for _ in range(2)]
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *states)

  def _values_from_outputs(self, outputs: jax.Array) -> jax.Array:
    merged_outputs = to_merged_outputs(outputs)
    return jnp.squeeze(self.value_head(merged_outputs), -1)

  def _q_values_from_outputs(
      self,
      outputs: jax.Array,  # [..., 2, O]
      values: jax.Array,  # [..., 2]
  ) -> jax.Array:  # [..., 2]
    merged_outputs = to_merged_outputs(outputs)
    qs = jnp.squeeze(self.q_head(merged_outputs), -1)

    if self.config.advantage_qs:
      return values + qs
    else:
      return qs

  def _action_rnn_initial_state(
      self, core_outputs: jax.Array,  # [..., 2, O_core]
  ) -> RecurrentState:
    """Projects core_net outputs to an initial state for action_rnn."""
    batch_shape = core_outputs.shape[:-1]
    zero_state = self.action_init.initial_state(batch_shape, rngs=nnx.Rngs(0))
    _, init_state = self.action_init.step(core_outputs, zero_state)
    return init_state

  def q_values_from_core_outputs(
      self,
      values: jax.Array,  # [..., 2]
      core_outputs: jax.Array,  # [..., 2, O_core]
      actions: list[Action],  # frame_skip x [..., 2]
  ) -> jax.Array:  # [..., 2]
    init_state = self._action_rnn_initial_state(core_outputs)
    embedded = [self.embed_action(a) for a in actions]
    stacked = jnp.stack(embedded, axis=0)  # [FS, ..., embed_size]
    reset = jnp.zeros(stacked.shape[:-1], dtype=bool)  # [FS, ...]
    outputs, _ = self.action_net.unroll(stacked, reset, init_state)
    return self._q_values_from_outputs(outputs[-1], values)

  def multi_q_values_from_core_outputs(
      self,
      values: jax.Array,  # [T, B, 2]
      core_outputs: jax.Array,  # [T, B, 2, O_core]
      actions: Action,  # [S, T, B, 2]
      batch_size: tp.Optional[int] = 0,  # 0 is equivalent to vmap
  ) -> jax.Array:  # [S, S, T, B, 2]
    action_inputs = self.embed_action(actions)
    action_rnn_state = self._action_rnn_initial_state(core_outputs)

    action_outputs, _ = jax_utils.lax_map(
        lambda x: self.action_net.step(x, action_rnn_state),
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
        return self._q_values_from_outputs(outputs, values)

      return jax_utils.lax_map(p1_fn, p1_outputs, batch_size=inner_bs)

    return jax_utils.lax_map(p0_fn, p0_outputs, batch_size=outer_bs)

  def unroll(
      self,
      state_action: data.StateAction[Rank3, Action],  # [T, B, 2]
      is_resetting: jax.Array,  # [T, B, 2]
      next_actions: list[Action],  # frame_skip x [T, B, 2]
      initial_state: RecurrentState,  #  [B, 2]
  ) -> tuple[UnrollOutputs, RecurrentState]:
    core_outputs, final_state = self.core_net.unroll(
        state_action, is_resetting, initial_state)
    values = self._values_from_outputs(core_outputs)

    q_values = self.q_values_from_core_outputs(
        values, core_outputs, next_actions)

    return UnrollOutputs(values=values, q_values=q_values), final_state

  def loss_batched(
      self,
      frames: data.Frames[Rank3, Action],  # [T + 1, B, 2]
      initial_state: RecurrentState,  # [B, 2]
      discount: float,
      batch_size: int,  # batch size in time
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

    # nnx will complain about trace levels if we use jax.lax.scan
    scan_fn = nnx.scan(
        nnx.remat(QFunction[Action].unroll),
        in_axes=(None, 0, 0, 0, nnx.Carry),
        out_axes=(0, nnx.Carry),
    )

    unroll_outputs, final_state = scan_fn(
        self, state_action, is_resetting, next_actions, initial_state)

    # Reshape outputs back to [T, B, 2]
    def to_unbatched(x: jax.Array) -> jax.Array:
      assert x.shape[0] == num_batches
      assert x.shape[1] == batch_size
      return x.reshape((total_unroll_length,) + x.shape[2:])

    unroll_outputs = utils.map_nt(to_unbatched, unroll_outputs)
    values, q_values = unroll_outputs

    last_state_action, last_is_resetting = utils.map_nt(
        lambda x: x[-1], (frames.state_action, frames.is_resetting))
    last_output, _ = self.core_net.step_with_reset(
        last_state_action, last_is_resetting, final_state)

    last_value = self._values_from_outputs(last_output)

    outputs = self._get_outputs(
        frames=frames,
        values=values,
        q_values=q_values,
        last_value=last_value,
        discount=discount,
    )

    return outputs, final_state

  def loss_and_core_outputs(
      self,
      frames: data.Frames[Rank3, Action],  # [T + 1, B, 2]
      initial_state: RecurrentState,
      discount: float,
  ) -> tp.Tuple[QOutputs, jax.Array, RecurrentState]:
    """Returns (q_outputs, core_outputs, final_state).

    core_outputs has shape [T, B, 2, O].
    final_state is the core_net recurrent state after the last frame.
    """
    state_action_T = utils.map_nt(lambda x: x[:-1], frames.state_action)
    core_outputs, final_state = self.core_net.unroll(
        state_action_T, frames.is_resetting[:-1], initial_state)

    values = self._values_from_outputs(core_outputs)

    last_output, _ = self.core_net.step_with_reset(
        utils.map_nt(lambda x: x[-1], frames.state_action),
        frames.is_resetting[-1], final_state)
    last_value = self._values_from_outputs(last_output)

    next_actions = jax.tree.map(
        lambda t: t[1:], frames.state_action.action)
    q_values = self.q_values_from_core_outputs(
        values, core_outputs, next_actions)

    outputs = self._get_outputs(
        frames=frames,
        values=values,
        q_values=q_values,
        last_value=last_value,
        discount=discount,
    )

    return outputs, core_outputs, final_state

  def _get_outputs(
      self,
      frames: data.Frames[Rank3, Action],
      values: jax.Array,
      q_values: jax.Array,
      last_value: jax.Array,
      discount: float,
  ):
    value_targets = rl_lib.generalized_returns_with_resetting(
        rewards=frames.reward,
        values=values,
        is_resetting=frames.is_resetting[1:],
        bootstrap=last_value,
        discount=discount ** self.frame_skip,
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
