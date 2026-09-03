import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx

from melee.enums import Action

from slippi_ai import data, types, utils
from slippi_ai.data import Rank2
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


class ValueFunction(nnx.Module, tp.Generic[types.Action]):

  def __init__(
      self,
      rngs: nnx.Rngs,
      network_config: dict,
      num_names: int,
      embed_config: embed.EmbedConfig[types.Action],
      frame_skip: int,
  ):
    self.frame_skip = frame_skip
    self.network = networks.build_embed_network(
        rngs=rngs,
        embed_config=embed_config,
        num_names=num_names,
        network_config=network_config,
        frame_skip=frame_skip,
    )
    self.value_head = nnx.Linear(self.network.output_size, 1, rngs=rngs)

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return self.network.initial_state(batch_size, rngs)

  def unroll(
      self,
      state_action: data.StateAction,
      is_resetting: Array,
      initial_state: RecurrentState,
  ) -> tp.Tuple[Array, RecurrentState]:
    outputs, final_state = self.network.unroll(
        state_action, is_resetting, initial_state)
    values = jnp.squeeze(self.value_head(outputs), -1)
    return values, final_state

  def loss(
      self,
      frames: data.Frames[Rank2, types.Action],
      initial_state: RecurrentState,
      discount: float,
      discount_on_death: tp.Optional[float] = None,
      lambda_: float = 1.0,
      advantage_lambda: tp.Optional[float] = None,
  ) -> tp.Tuple[ValueOutputs, RecurrentState]:
    """Computes prediction loss on a batch of frames.

    Args:
      frames: Time-major batch of states, actions, and rewards.
        Assumed to have one frame of overlap.
      initial_state: Batch of initial recurrent states.
      discount: Per-frame discount factor for returns.
      discount_on_death: Discount factor to use when either player *respawns*.
        The reward for KOs comes on the frame of death, which precedes respawn.
      lambda_: TD-lambda parameter for the value targets; 1 gives discounted
        returns while 0 gives one-step TD targets.
      advantage_lambda: TD-lambda parameter for the returned advantages; None
        uses lambda_. Allows e.g. training the values with low-variance TD(0)
        targets while still providing unbiased lambda=1 advantages.
    """
    inputs = utils.map_nt(lambda t: t[:-1], frames.state_action)
    values, final_state = self.unroll(
        inputs, frames.is_resetting[:-1], initial_state)

    outputs = self._get_outputs(
        frames=frames,
        values=values,
        final_state=final_state,
        discount=discount,
        discount_on_death=discount_on_death,
        lambda_=lambda_,
        advantage_lambda=advantage_lambda,
    )

    return outputs, final_state

  def loss_batched(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
      discount: float,
      batch_size: int,  # batch size in time
      discount_on_death: tp.Optional[float] = None,
      lambda_: float = 1.0,
      advantage_lambda: tp.Optional[float] = None,
  ) -> tp.Tuple[ValueOutputs, RecurrentState]:
    """Like `loss`, but unrolls the network in chunks of `batch_size` frames."""
    total_unroll_length = frames.reward.shape[0]
    num_batches, r = divmod(total_unroll_length, batch_size)
    if r != 0:
      raise ValueError(
          f'Unroll length {total_unroll_length} is not divisible by '
          f'batch size {batch_size}.')

    def to_batched(x: Array) -> Array:
      assert x.shape[0] == total_unroll_length
      return x.reshape((num_batches, batch_size) + x.shape[1:])

    inputs, is_resetting = jax.tree.map(
        lambda x: to_batched(x[:-1]),
        (frames.state_action, frames.is_resetting))

    # nnx will complain about trace levels if we use jax.lax.scan
    scan_fn = nnx.scan(
        nnx.remat(ValueFunction.unroll),
        in_axes=(None, 0, 0, nnx.Carry),
        out_axes=(0, nnx.Carry),
    )

    values, final_state = scan_fn(self, inputs, is_resetting, initial_state)

    # Reshape values back to [T, B]
    assert values.shape[:2] == (num_batches, batch_size)
    values = values.reshape((total_unroll_length,) + values.shape[2:])

    outputs = self._get_outputs(
        frames=frames,
        values=values,
        final_state=final_state,
        discount=discount,
        discount_on_death=discount_on_death,
        lambda_=lambda_,
        advantage_lambda=advantage_lambda,
    )

    return outputs, final_state

  def _get_outputs(
      self,
      frames: data.Frames,
      values: Array,
      final_state: RecurrentState,
      discount: float,
      discount_on_death: tp.Optional[float] = None,
      lambda_: float = 1.0,
      advantage_lambda: tp.Optional[float] = None,
  ) -> ValueOutputs:
    rewards = frames.reward

    last_input = utils.map_nt(lambda t: t[-1], frames.state_action)
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

    # Don't bootstrap across game boundaries.
    discounts = jnp.where(frames.is_resetting[1:], 0.0, discounts)

    # Return computation is done in float32 as it doesn't work well in bf16.
    value_targets = rl_lib.generalized_returns(
        rewards=rewards,
        discounts=discounts,
        values=values,
        bootstrap=last_value,
        lambdas=jnp.full_like(discounts, lambda_))
    value_targets = jax.lax.stop_gradient(value_targets)
    td_errors = value_targets - values
    value_loss = jnp.square(td_errors)

    if advantage_lambda is None or advantage_lambda == lambda_:
      advantages = td_errors
    else:
      advantage_targets = rl_lib.generalized_returns(
          rewards=rewards,
          discounts=discounts,
          values=values,
          bootstrap=last_value,
          lambdas=jnp.full_like(discounts, advantage_lambda))
      advantages = jax.lax.stop_gradient(advantage_targets) - values

    _, value_variance = jax_utils.mean_and_variance(value_targets)
    uev = value_loss / (value_variance + 1e-8)

    metrics = {
        # Scalar metrics aren't allowed in shard_map
        # 'reward': jax_utils.get_stats(rewards),
        # 'return': jax_utils.get_stats(value_targets),
        'loss': value_loss,
        # Note: full_like raises an error due to pvary.
        # TODO: this is probably a jax bug and we should report it.
        'variance': jnp.full(value_loss.shape, value_variance),
        'uev': uev,  # unexplained variance
    }

    return ValueOutputs(
        returns=value_targets,
        advantages=advantages,
        loss=value_loss,
        metrics=metrics,
    )

