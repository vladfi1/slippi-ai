import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx

from melee.enums import Action

from slippi_ai import data, types, utils
from slippi_ai.jax import embed, networks, jax_utils, rl_lib
from slippi_ai.jax.networks import RecurrentState

Array = jax.Array


class ValueOutputs(tp.NamedTuple):
  returns: Array  # [T, B]
  advantages: Array  # [T, B]
  loss: Array
  metrics: dict

def player_respawn(player: types.Player) -> Array:
  """Returns a boolean mask indicating when a player respawns."""
  actions = player.action
  # Note: players seem to be able to skip ON_HALO_WAIT
  respawn = Action.ON_HALO_DESCENT.value
  return jnp.logical_and(actions[:-1] != respawn, actions[1:] == respawn)


class ValueFunction(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      network_config: dict,
      num_names: int,
      embed_config: embed.EmbedConfig,
  ):
    self.network = networks.build_embed_network(
        rngs=rngs,
        embed_config=embed_config,
        num_names=num_names,
        network_config=network_config,
    )
    self.value_head = nnx.Linear(self.network.output_size, 1, rngs=rngs)

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return self.network.initial_state(batch_size, rngs)

  def loss(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
      discount: float,
      discount_on_death: tp.Optional[float] = None,
  ) -> tp.Tuple[ValueOutputs, RecurrentState]:
    """Computes prediction loss on a batch of frames.

    Args:
      frames: Time-major batch of states, actions, and rewards.
        Assumed to have one frame of overlap.
      initial_state: Batch of initial recurrent states.
      discount: Per-frame discount factor for returns.
      discount_on_death: Discount factor to use when either player *respawns*.
        The reward for KOs comes on the frame of death, which precedes respawn.
    """
    rewards = frames.reward

    inputs = utils.map_nt(lambda t: t[:-1], frames.state_action)
    last_input = utils.map_nt(lambda t: t[-1], frames.state_action)
    outputs, final_state = self.network.unroll(
        inputs, frames.is_resetting[:-1], initial_state)

    values = jnp.squeeze(self.value_head(outputs), -1)
    last_output, _ = self.network.step_with_reset(
        last_input, frames.is_resetting[-1], final_state)
    last_value = jnp.squeeze(self.value_head(last_output), -1)
    discounts = jnp.full_like(rewards, discount)

    if discount_on_death is not None:
      respawn_happened = jnp.logical_or(
          player_respawn(frames.state_action.state.p0),
          player_respawn(frames.state_action.state.p1))

      discounts_on_death = jnp.full_like(discounts, discount_on_death)

      discounts = jnp.where(respawn_happened, discounts_on_death, discounts)

    value_targets = rl_lib.discounted_returns(
        rewards=rewards,
        discounts=discounts,
        bootstrap=last_value)
    value_targets = jax.lax.stop_gradient(value_targets)
    advantages = value_targets - values
    value_loss = jnp.square(advantages)

    _, value_variance = jax_utils.mean_and_variance(value_targets)
    uev = value_loss / (value_variance + 1e-8)

    metrics = {
        # Scalar metrics aren't allowed in shard_map
        # 'reward': jax_utils.get_stats(rewards),
        # 'return': jax_utils.get_stats(value_targets),
        'loss': value_loss,
        'variance': value_variance[None, None],
        'uev': uev,  # unexplained variance
    }

    outputs = ValueOutputs(
        returns=value_targets,
        advantages=advantages,
        loss=value_loss,
        metrics=metrics,
    )

    return outputs, final_state


class FakeValueFunction(nnx.Module):

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    del batch_size, rngs
    return ()

  def loss(self, frames: data.Frames, initial_state, discount, **_):
    del discount

    outputs = ValueOutputs(
        returns=jnp.zeros_like(frames.reward),
        loss=jnp.array(0.0),
        advantages=jnp.zeros_like(frames.reward),
        metrics={},
    )

    return outputs, initial_state
