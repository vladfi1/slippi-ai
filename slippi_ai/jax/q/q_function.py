import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx

from slippi_ai import data
from slippi_ai.jax import rl_lib
from slippi_ai.jax.networks import RecurrentState
from slippi_ai.jax import embed, networks, jax_utils
from slippi_ai.jax.embed import Action
from slippi_ai.types import Controller

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

Rank2 = tuple[int, int]

class QFunction(nnx.Module, tp.Generic[Action]):

  def __init__(
      self,
      rngs: nnx.Rngs,
      embed_action: embed.Embedding[Controller, Action],
      embed_config: embed.EmbedConfig,
      num_names: int,
      network_config: dict,
  ):
    self.embed_action = embed_action
    self.core_net = networks.build_embed_network(
        rngs, embed_config, num_names, network_config, embed_action=embed_action)
    self.action_net = networks.construct_network(
        rngs, input_size=self.embed_action.size, **network_config)

    self.value_head = nnx.Linear(
      in_features=self.core_net.output_size,
      out_features=1, rngs=rngs)
    self.q_head = nnx.Linear(
      in_features=self.core_net.output_size,
      out_features=1, rngs=rngs)

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> networks.RecurrentState:
    return self.core_net.initial_state(batch_size, rngs)

  def unroll(
      self,
      state_action: data.StateAction[Rank2, Action],
      is_resetting: jax.Array,
      next_actions: Action,
      initial_state: RecurrentState,
  ) -> tuple[UnrollOutputs, RecurrentState]:
    outputs, hidden_states = self.core_net.scan(
        state_action, is_resetting, initial_state)
    values = jnp.squeeze(self.value_head(outputs), -1)
    q_values = self.q_values_from_hidden_states(
        values, hidden_states, next_actions)

    final_state = jax.tree.map(lambda t: t[-1], hidden_states)

    return UnrollOutputs(values=values, q_values=q_values), final_state

  def loss_batched(
      self,
      frames: data.Frames[Rank2, Action],
      initial_state: RecurrentState,
      discount: float,
      batch_size: int,  # batch size in time
  ) -> tp.Tuple[QOutputs, RecurrentState]:
    total_unroll_length = frames.reward.shape[0]
    num_batches, r = divmod(total_unroll_length, batch_size)
    if r != 0:
      raise ValueError(f'Unroll length {total_unroll_length} is not divisible by batch size {batch_size}.')

    def to_batched(x: jax.Array) -> jax.Array:
      assert x.shape[0] == total_unroll_length
      return x.reshape((num_batches, batch_size) + x.shape[1:])

    state_action, is_resetting = jax.tree.map(
        lambda x: to_batched(x[:-1]),
        (frames.state_action, frames.is_resetting))
    next_actions = jax.tree.map(lambda x: to_batched(x[1:]), frames.state_action.action)

    # nnx will complain about trace levels if we use jax.lax.scan
    scan_fn = nnx.scan(
        nnx.remat(QFunction[Action].unroll),
        in_axes=(None, 0, 0, 0, nnx.Carry),
        out_axes=(0, nnx.Carry),
    )

    unroll_outputs, final_state = scan_fn(
        self, state_action, is_resetting, next_actions, initial_state)

    # Reshape outputs back to [T, B]
    def to_unbatched(x: jax.Array) -> jax.Array:
      assert x.shape[:2] == (num_batches, batch_size)
      return x.reshape((total_unroll_length,) + x.shape[2:])

    unroll_outputs: UnrollOutputs = jax.tree.map(to_unbatched, unroll_outputs)
    values, q_values = unroll_outputs

    last_state_action, last_is_resetting = jax.tree.map(
        lambda x: x[-1], (frames.state_action, frames.is_resetting))
    last_output, _ = self.core_net.step_with_reset(
        last_state_action, last_is_resetting, final_state)
    last_value = jnp.squeeze(self.value_head(last_output), -1)

    outputs = self._get_outputs(
        frames=frames,
        values=values,
        q_values=q_values,
        last_value=last_value,
        discount=discount,
    )

    return outputs, final_state

  def loss(
      self,
      frames: data.Frames[Rank2, Action],
      initial_state: RecurrentState,
      discount: float,
  ) -> tp.Tuple[QOutputs, RecurrentState]:
    """Computes prediction loss on a batch of frames.

    Args:
      frames: Time-major batch of states, actions, and rewards.
        Assumed to have one frame of overlap.
      initial_state: Batch of initial recurrent states.
      discount: Per-frame discount factor for returns.
    """
    all_outputs, all_hidden_states = self.core_net.scan(
        frames.state_action, frames.is_resetting, initial_state)

    hidden_states = jax.tree.map(lambda t: t[:-1], all_hidden_states)
    final_state = jax.tree.map(lambda t: t[-1], hidden_states)

    all_values = jnp.squeeze(self.value_head(all_outputs), -1)
    values, last_value = all_values[:-1], all_values[-1]

    next_actions = jax.tree.map(
        lambda t: t[1:], frames.state_action.action)
    # Here we are batching over time (and batch)
    q_values = self.q_values_from_hidden_states(
        values, hidden_states, next_actions)

    outputs = self._get_outputs(
        frames=frames,
        values=values,
        q_values=q_values,
        last_value=last_value,
        discount=discount,
    )

    return outputs, final_state

  def _get_outputs(
      self,
      frames: data.Frames[Rank2, Action],
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
        discount=discount,
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

  def q_values_from_hidden_states(
      self,
      values: jax.Array,
      hidden_states: RecurrentState,
      actions: Action,
  ) -> jax.Array:
    action_inputs = self.embed_action(actions)
    action_outputs, _ = self.action_net.step(action_inputs, hidden_states)
    advantages = jnp.squeeze(self.q_head(action_outputs), -1)
    return values + advantages

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
