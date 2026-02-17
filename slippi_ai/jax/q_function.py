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
  hidden_states: RecurrentState  # [T, B]
  metrics: dict

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
    rewards = frames.reward

    all_outputs, all_hidden_states = self.core_net.scan(
        frames.state_action, frames.is_resetting, initial_state)

    outputs, last_output = all_outputs[:-1], all_outputs[-1]
    hidden_states = jax.tree.map(lambda t: t[:-1], all_hidden_states)
    del all_outputs, all_hidden_states
    final_state = jax.tree.map(lambda t: t[-1], hidden_states)

    # Includes "overlap" frame.
    # unroll_length = state_action.state.stage.shape[0] - delay

    # Compute value loss
    values = jnp.squeeze(self.value_head(outputs), -1)
    last_value = jnp.squeeze(self.value_head(last_output), -1)

    value_targets = rl_lib.generalized_returns_with_resetting(
        rewards=rewards,
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

    # Compute Q loss
    next_actions = jax.tree.map(
        lambda t: t[1:], frames.state_action.action)
    # Here we are batching over time (and batch)
    q_values = self.q_values_from_hidden_states(
        hidden_states, next_actions)

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

    outputs = QOutputs(
        returns=value_targets,
        advantages=advantages,
        values=values,
        q_values=q_values,
        loss=value_loss + q_loss,
        hidden_states=hidden_states,
        metrics=metrics,
    )

    return outputs, final_state

  def q_values_from_hidden_states(
      self,
      hidden_states: RecurrentState,
      actions: Action,
  ) -> jax.Array:
    action_inputs = self.embed_action(actions)
    action_outputs, _ = self.action_net.step(action_inputs, hidden_states)
    return jnp.squeeze(self.q_head(action_outputs), -1)

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
        hidden_states=(),
        metrics={},
    )

    return outputs, initial_state
