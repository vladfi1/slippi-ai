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
from slippi_ai.types import Controller, Action

Gaussian = rl_lib.Gaussian

class QOutputs(tp.NamedTuple):
  returns: jax.Array  # [T, B]
  advantages: jax.Array  # [T, B]
  values: jax.Array  # [T, B]
  q_values: jax.Array  # [T, B]
  loss: jax.Array
  # hidden_states: RecurrentState  # [T, B]
  metrics: dict

class UnrollOutputs(tp.NamedTuple):
  values: Gaussian  # [T, B]
  q_values: Gaussian  # [T, B]

Rank2 = tuple[int, int]

# Floor on predicted variances for numerical stability.
_MIN_VARIANCE = 1e-6
# Floor on target variances so divergences stay finite even when the
# bootstrapped return is (near) deterministic.
_VAR_EPS = 1e-8

def gaussian_kl(p: Gaussian, q: Gaussian) -> jax.Array:
  """KL(P || Q) between two diagonal Gaussians."""
  return 0.5 * (
      jnp.log(q.variance) - jnp.log(p.variance)
      + (p.variance + jnp.square(p.mean - q.mean)) / q.variance
      - 1.0)

def _half_normal_abs_mean(g: Gaussian) -> jax.Array:
  """E[|Z|] for Z ~ N(g)."""
  mu = g.mean
  var = g.variance
  sigma = jnp.sqrt(var)
  return (
      sigma * jnp.sqrt(2 / jnp.pi) * jnp.exp(-jnp.square(mu) / (2 * var))
      + mu * jax.scipy.special.erf(mu / (sigma * jnp.sqrt(2.0))))

def gaussian_cramer(p: Gaussian, q: Gaussian) -> jax.Array:
  """Cramer distance (integral of squared CDF difference) between Gaussians.

  Uses the closed form via the energy-distance identity:
    int (F_P - F_Q)^2 dx = E|X - Y| - (E|X - X'| + E|Y - Y'|) / 2,
  where X ~ P, Y ~ Q (and primed copies are iid). For Gaussians this reduces to
  the expressions below. The distance is symmetric in P and Q.
  """
  stddev_p = jnp.sqrt(p.variance)
  stddev_q = jnp.sqrt(q.variance)
  difference = Gaussian(mean=p.mean - q.mean, variance=p.variance + q.variance)
  cross = _half_normal_abs_mean(difference)
  return cross - (stddev_p + stddev_q) / jnp.sqrt(jnp.pi)

_DISTANCES = {
    'kl': gaussian_kl,
    'cramer': gaussian_cramer,
}

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
  distance: str = 'kl'  # Distributional loss between Gaussians: 'kl' or 'cramer'

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

    # Heads output two scalars per step: (mean, raw_variance).
    self.value_head = jax_utils.MLP(
      rngs=rngs,
      input_size=self.core_net.output_size,
      features=[config.head.hidden_size] * config.head.num_layers + [2],
      activate_final=False,
    )
    self.q_head = jax_utils.MLP(
      rngs=rngs,
      input_size=self.action_net.output_size,
      features=[config.head.hidden_size] * config.head.num_layers + [2],
      activate_final=False,
    )

    if config.distance not in _DISTANCES:
      raise ValueError(
          f'Unknown distance {config.distance!r}; expected one of {list(_DISTANCES)}.')
    self._distance_fn = _DISTANCES[config.distance]

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> networks.RecurrentState:
    return self.core_net.initial_state(batch_size, rngs)

  def _embed_action(self, action: Action) -> jax.Array:
    x = self.embed_action(action)
    return x.astype(jax_utils.module_dtype(self))

  def _gaussian_from_head(self, head_output: jax.Array) -> Gaussian:
    """[..., 2] -> Gaussian([...], [...])."""
    mean = head_output[..., 0]
    variance = jax.nn.softplus(head_output[..., 1]) + _MIN_VARIANCE
    return Gaussian(mean=mean, variance=variance)

  def _value_dist_from_outputs(self, outputs: jax.Array) -> Gaussian:
    """[..., O_core] -> Gaussian([...], [...])."""
    return self._gaussian_from_head(self.value_head(outputs))

  def _values_from_outputs(self, outputs: jax.Array) -> jax.Array:
    """[..., O_core] -> [...]"""
    return self._value_dist_from_outputs(outputs).mean

  def _q_dist_from_outputs(
      self,
      outputs: jax.Array,  # [..., O_action]
      values: Gaussian,  # [...]
  ) -> Gaussian:  # [...]
    q_dist = self._gaussian_from_head(self.q_head(outputs))

    if self.config.advantage_qs:
      # The q-head predicts an advantage on the mean; its variance is the
      # q-value's own (absolute) uncertainty.
      q_dist = q_dist._replace(mean=values.mean + q_dist.mean)

    return q_dist

  def _q_values_from_outputs(
      self,
      outputs: jax.Array,  # [..., O_action]
      values: jax.Array,  # [...]
  ) -> jax.Array:  # [...]
    qs = self.q_head(outputs)[..., 0]

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

  def q_values_from_action_state(
      self,
      values: jax.Array,  # [...]
      action_init_state: networks.RecurrentState,  # [..., H]
      actions: list[Action],  # frame_skip x [...]
  ) -> jax.Array:  # [...]
    embedded = [self._embed_action(a) for a in actions]
    stacked = jnp.stack(embedded, axis=0)  # [FS, ..., embed_size]
    reset = jnp.zeros(stacked.shape[:-1], dtype=bool)  # [FS, ...]
    outputs, _ = self.action_net.unroll(stacked, reset, action_init_state)
    return self._q_values_from_outputs(outputs[-1], values)

  def q_values_from_core_outputs(
      self,
      values: jax.Array,  # [...]
      core_outputs: jax.Array,  # [..., O_core]
      actions: list[Action],  # frame_skip x [...]
  ) -> jax.Array:  # [...]
    init_state = self._action_net_initial_state(core_outputs)
    return self.q_values_from_action_state(values, init_state, actions)

  def _q_dist_from_action_state(
      self,
      values: Gaussian,  # [...]
      action_init_state: networks.RecurrentState,  # [..., H]
      actions: list[Action],  # frame_skip x [...]
  ) -> Gaussian:  # [...]
    embedded = [self._embed_action(a) for a in actions]
    stacked = jnp.stack(embedded, axis=0)  # [FS, ..., embed_size]
    reset = jnp.zeros(stacked.shape[:-1], dtype=bool)  # [FS, ...]
    outputs, _ = self.action_net.unroll(stacked, reset, action_init_state)
    return self._q_dist_from_outputs(outputs[-1], values)

  def _q_dist_from_core_outputs(
      self,
      values: Gaussian,  # [...]
      core_outputs: jax.Array,  # [..., O_core]
      actions: list[Action],  # frame_skip x [...]
  ) -> Gaussian:  # [...]
    init_state = self._action_net_initial_state(core_outputs)
    return self._q_dist_from_action_state(values, init_state, actions)

  def multi_q_values_from_action_state(
      self,
      values: jax.Array,  # [T, B]
      action_init_state: networks.RecurrentState,  # [T, B, H]
      actions: list[Action],  # frame_skip x [S, T, B]
      batch_size: tp.Optional[int] = 0,  # 0 is equivalent to vmap
  ) -> jax.Array:  # [S, T, B]
    embedded = [self._embed_action(a) for a in actions]  # frame_skip x [S, T, B, E]
    action_inputs = jnp.stack(embedded, axis=1)  # [S, FS, T, B, E]

    def process_one_sample(embedded_fs: jax.Array) -> jax.Array:
      # embedded_fs: [FS, T, B, E]
      reset = jnp.zeros(embedded_fs.shape[:-1], dtype=bool)
      outputs, _ = self.action_net.unroll(embedded_fs, reset, action_init_state)
      return self._q_values_from_outputs(outputs[-1], values)  # [T, B]

    return jax_utils.lax_map(
        process_one_sample,
        action_inputs,
        batch_size=batch_size,
    )

  def unroll(
      self,
      state_action: data.StateAction[Rank2, Action],  # [T, B]
      is_resetting: jax.Array,  # [T, B]
      next_actions: list[Action],  # frame_skip x [T, B]
      initial_state: RecurrentState,  # [B]
  ) -> tuple[UnrollOutputs, RecurrentState]:
    core_outputs, final_state = self.core_net.unroll(
        state_action, is_resetting, initial_state)
    values = self._value_dist_from_outputs(core_outputs)

    q_values = self._q_dist_from_core_outputs(
        values, core_outputs, next_actions)

    return UnrollOutputs(values=values, q_values=q_values), final_state

  def loss_batched(
      self,
      frames: data.Frames[Rank2, Action],  # [T + 1, B]
      initial_state: RecurrentState,  # [B]
      discount: float,
      batch_size: int,  # batch size in time
      lambda_: float = 1.0,
      eval_lambdas: list[float] = [0],
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

    # Reshape outputs back to [T, B]
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

    last_value = self._value_dist_from_outputs(last_output)

    outputs = self._get_outputs(
        frames=frames,
        values=values,
        q_values=q_values,
        last_value=last_value,
        discount=discount,
        lambda_=lambda_,
    )

    for eval_lambda in eval_lambdas:
      eval_outputs = self._get_outputs(
          frames=frames,
          values=values,
          q_values=q_values,
          last_value=last_value,
          discount=discount,
          lambda_=eval_lambda,
      )
      outputs.metrics[f'lambda_{eval_lambda:.1f}'] = eval_outputs.metrics

    return outputs, final_state

  def loss_and_action_state(
      self,
      frames: data.Frames[Rank2, Action],  # [T + 1, B]
      initial_state: RecurrentState,
      discount: float,
      lambda_: float = 1.0,
  ) -> tp.Tuple[QOutputs, RecurrentState, RecurrentState]:
    """Returns (q_outputs, action_init_state, final_state).

    action_init_state has shape [T, B, H] (action_net initial state per step).
    final_state is the core_net recurrent state after the last frame.
    """
    state_action_T = utils.map_nt(lambda x: x[:-1], frames.state_action)
    core_outputs, final_state = self.core_net.unroll(
        state_action_T, frames.is_resetting[:-1], initial_state)

    values = self._value_dist_from_outputs(core_outputs)

    last_output, _ = self.core_net.step_with_reset(
        utils.map_nt(lambda x: x[-1], frames.state_action),
        frames.is_resetting[-1], final_state)
    last_value = self._value_dist_from_outputs(last_output)

    action_init_state = self._action_net_initial_state(core_outputs)

    next_actions = jax.tree.map(
        lambda t: t[1:], frames.state_action.action)
    q_values = self._q_dist_from_action_state(
        values, action_init_state, next_actions)

    outputs = self._get_outputs(
        frames=frames,
        values=values,
        q_values=q_values,
        last_value=last_value,
        discount=discount,
        lambda_=lambda_,
    )

    return outputs, action_init_state, final_state

  def _get_outputs(
      self,
      frames: data.Frames[Rank2, Action],
      values: Gaussian,
      q_values: Gaussian,
      last_value: Gaussian,
      discount: float,
      lambda_: float,
  ):
    target = rl_lib.generalized_returns_gaussian_with_resetting(
        rewards=frames.reward,
        values=values,
        is_resetting=frames.is_resetting[1:],
        bootstrap=last_value,
        discount=discount,
        lambda_=lambda_,
    )
    target = jax.lax.stop_gradient(target)
    # Floor the target variance so divergences stay finite for (near)
    # deterministic returns (e.g. variance 0 at resets when lambda_ == 1).
    target = target._replace(variance=target.variance + _VAR_EPS)

    advantages = target.mean - values.mean

    # Keep the "loss" metric as L2 on the means for comparison with prior runs.
    value_loss = jnp.square(advantages)
    q_loss = jnp.square(target.mean - q_values.mean)

    # Distributional divergences (the actually-optimized losses).
    value_div = self._distance_fn(target, values)
    q_div = self._distance_fn(target, q_values)

    _, value_variance = jax_utils.mean_and_variance(target.mean)
    uev = value_loss / (value_variance + 1e-8)

    quev = q_loss / (value_variance + 1e-8)
    uev_delta = uev - quev

    metrics = {
        'v': {
            'loss': value_loss,
            'uev': uev,
            'div': value_div,
            self.config.distance: value_div,
            'variance': values.variance,
        },
        'q': {
            'loss': q_loss,
            'uev': quev,
            'uev_delta': uev_delta,
            # Take log to result in a geometric mean.
            'rel_v_loss': jnp.log((value_loss + 1e-8) / (q_loss + 1e-8)),
            'div': q_div,
            self.config.distance: q_div,
            'variance': q_values.variance,
        },
    }

    return QOutputs(
        returns=target.mean,
        advantages=advantages,
        values=values.mean,
        q_values=q_values.mean,
        loss=value_div + q_div,
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
