import abc
import copy
from typing import Any, Callable, Optional, Tuple
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx

from slippi_ai.jax import jax_utils
from slippi_ai.jax import embed as embed_lib
from slippi_ai.data import StateAction
from slippi_ai.types import Game

Array = jax.Array

RecurrentState = tp.Any
Inputs = Array


def where_pytree(cond: Array, x, y):
  """Like jnp.where but broadcasts cond over pytree leaves."""
  return jax.tree.map(lambda a, b: jax_utils.where(cond, a, b), x, y)

InputTree = tp.TypeVar('InputTree')

def dynamic_rnn(
    cell_fn: Callable[[InputTree, RecurrentState], Tuple[Array, RecurrentState]],
    inputs: InputTree,
    initial_state: RecurrentState,
) -> Tuple[Array, RecurrentState]:
  """Unrolls an RNN over time, returning outputs and final state.

  Args:
    cell_fn: Function (inputs, state) -> (outputs, new_state)
    inputs: Inputs with time as first axis
    initial_state: Initial recurrent state

  Returns:
    outputs: Stacked outputs over time
    final_state: Final recurrent state
  """
  def scan_fn(state, x):
    outputs, new_state = cell_fn(x, state)
    return new_state, outputs

  final_state, outputs = jax.lax.scan(scan_fn, initial_state, inputs)
  return outputs, final_state


def scan_rnn(
    cell_fn: Callable[[InputTree, RecurrentState], Tuple[Array, RecurrentState]],
    inputs: InputTree,
    initial_state: RecurrentState,
) -> Tuple[Array, RecurrentState]:
  """Like dynamic_rnn but returns all intermediate hidden states.

  Args:
    cell_fn: Function (inputs, state) -> (outputs, new_state)
    inputs: Inputs with time as first axis
    initial_state: Initial recurrent state

  Returns:
    outputs: Stacked outputs over time
    hidden_states: All intermediate hidden states
  """
  def scan_fn(state, x):
    outputs, new_state = cell_fn(x, state)
    return new_state, (outputs, new_state)

  _, (outputs, hidden_states) = jax.lax.scan(scan_fn, initial_state, inputs)
  return outputs, hidden_states


class Network(nnx.Module, abc.ABC):

  @staticmethod
  def default_config() -> dict[str, tp.Any]:
    '''Returns the default config for this Network.'''
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def output_size(self) -> int:
    '''Returns the output size of the network.'''

  @abc.abstractmethod
  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    '''Returns the initial state for a batch of size batch_size.'''

  def step(
      self,
      inputs: Inputs,
      prev_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    '''Step without reset.

    Arguments:
      inputs: (batch_size, x_dim)
      prev_state: (batch, state_dim)

    Returns a tuple (outputs, final_state)
      outputs: (batch, out_dim)
      final_state: (batch, state_dim)
    '''
    raise NotImplementedError()

  def step_with_reset(
      self,
      inputs: Inputs,
      reset: Array,
      prev_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    batch_size = reset.shape[0]
    rngs = nnx.Rngs(0)  # TODO: pass rngs properly
    initial_state = where_pytree(
        reset, self.initial_state(batch_size, rngs), prev_state)
    return self.step(inputs, initial_state)

  def _step_with_reset(
      self,
      inputs_and_reset: tuple[Inputs, Array],
      prev_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    """Used for unroll/scan."""
    inputs, reset = inputs_and_reset
    return self.step_with_reset(inputs, reset, prev_state)

  def unroll(
      self,
      inputs: Inputs,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    '''
    Arguments:
      inputs: (time, batch, x_dim)
      reset: (time, batch)
      initial_state: (batch, state_dim)

    Returns a tuple (outputs, final_state)
      outputs: (time, batch, out_dim)
      final_state: (batch, state_dim)
    '''
    return dynamic_rnn(
        self._step_with_reset, (inputs, reset), initial_state)

  def scan(
      self,
      inputs: Inputs,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    '''Like unroll but also returns intermediate hidden states.

    Arguments:
      inputs: (time, batch, x_dim)
      reset: (time, batch)
      initial_state: (batch, state_dim)

    Returns a tuple (outputs, hidden_states)
      outputs: (time, batch, out_dim)
      hidden_states: (time, batch, state_dim)
    '''
    return scan_rnn(
        self._step_with_reset, (inputs, reset), initial_state)


class MLP(Network):

  @staticmethod
  def default_config() -> dict[str, tp.Any]:
    return dict(
      depth=2,
      width=128,
    )

  def __init__(self, rngs: nnx.Rngs, input_size: int, depth: int, width: int):
    self._width = width
    self._mlp = jax_utils.MLP(
        rngs=rngs,
        input_size=input_size,
        features=[width] * depth,
    )

  @property
  def output_size(self) -> int:
    return self._width

  def initial_state(self, batch_size, rngs):
    return ()

  def step(self, inputs, prev_state):
    del prev_state
    return self._mlp(inputs), ()

  def unroll(self, inputs, reset, initial_state):
    del reset, initial_state
    return self._mlp(inputs), ()

class RecurrentWrapper(Network):
  """Wraps an RNN cell as a Network."""

  def __init__(self, core: nnx.RNNCellBase, output_size: int):
    self._core = core
    self._output_size = output_size

  @property
  def output_size(self) -> int:
    return self._output_size

  def initial_state(self, batch_size: int, rngs: nnx.Rngs):
    input_shape = (batch_size, getattr(self._core, 'in_features', 1))
    return self._core.initialize_carry(input_shape, rngs)

  def step(self, inputs, prev_state):
    # flax's RNNCells have the arguments reversed
    next_state, output = self._core(prev_state, inputs)
    return output, next_state


class FFWWrapper(Network):
  """Wraps a feedforward module as a Network.

  Assumes that the module can handle multiple batch dimensions.
  """

  def __init__(self, module: tp.Callable[[Array], Array], output_size: int):
    self._module = module
    self._output_size = output_size

  @property
  def output_size(self) -> int:
    return self._output_size

  def initial_state(self, batch_size: int, rngs: nnx.Rngs):
    del batch_size, rngs
    return ()

  def step(self, inputs, prev_state):
    del prev_state
    return self._module(inputs), ()

  def unroll(self, inputs, reset, initial_state):
    del reset, initial_state
    return self._module(inputs), ()

  def scan(self, inputs, reset, initial_state):
    del reset, initial_state
    return self._module(inputs), ()


class Sequential(Network):

  def __init__(self, layers: list[Network]):
    self._layers = nnx.List(layers)

  @property
  def output_size(self) -> int:
    return self._layers[-1].output_size

  def initial_state(self, batch_size: int, rngs: nnx.Rngs):
    return [layer.initial_state(batch_size, rngs) for layer in self._layers]

  def step(self, inputs, prev_state):
    next_states: list[RecurrentState] = []
    for layer, state in zip(self._layers, prev_state):
      inputs, next_state = layer.step(inputs, state)
      next_states.append(next_state)
    return inputs, next_states

  def unroll(self, inputs, reset, initial_state):
    final_states: list[RecurrentState] = []
    for layer, state in zip(self._layers, initial_state):
      inputs, final_state = layer.unroll(inputs, reset, state)
      final_states.append(final_state)
    return inputs, final_states

  def scan(self, inputs, reset, initial_state):
    hidden_states: list[RecurrentState] = []
    for layer, state in zip(self._layers, initial_state):
      inputs, hidden_state = layer.scan(inputs, reset, state)
      hidden_states.append(hidden_state)
    return inputs, hidden_states


class ResidualWrapper(Network):

  def __init__(self, net: Network):
    self._net = net

  @property
  def output_size(self) -> int:
    return self._net.output_size

  def initial_state(self, batch_size: int, rngs: nnx.Rngs):
    return self._net.initial_state(batch_size, rngs)

  def step(self, inputs, prev_state):
    outputs, next_state = self._net.step(inputs, prev_state)
    return inputs + outputs, next_state

  def unroll(self, inputs, reset, initial_state):
    outputs, final_state = self._net.unroll(inputs, reset, initial_state)
    return inputs + outputs, final_state

  def scan(self, inputs, reset, initial_state):
    outputs, hidden_state = self._net.scan(inputs, reset, initial_state)
    return inputs + outputs, hidden_state


class GRU(RecurrentWrapper):

  @staticmethod
  def default_config() -> dict[str, tp.Any]:
    return dict(
      hidden_size=128,
    )

  def __init__(self, rngs: nnx.Rngs, input_size: int, hidden_size: int):
    super().__init__(
        nnx.GRUCell(
            in_features=input_size,
            hidden_features=hidden_size,
            rngs=rngs),
        output_size=hidden_size)

class LSTM(RecurrentWrapper):

  @staticmethod
  def default_config() -> dict[str, tp.Any]:
    return dict(
      hidden_size=128,
    )

  def __init__(self, rngs: nnx.Rngs, input_size: int, hidden_size: int):
    super().__init__(
        nnx.LSTMCell(
            in_features=input_size,
            hidden_features=hidden_size,
            rngs=rngs),
        output_size=hidden_size)


class ResBlock(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      residual_size: int,
      hidden_size: Optional[int] = None,
      activation: Callable[[Array], Array] = nnx.relu,
  ):
    self.layernorm = nnx.LayerNorm(residual_size, rngs=rngs)
    self.linear1 = nnx.Linear(residual_size, hidden_size or residual_size, rngs=rngs)
    self.activation = activation
    # Initialize the output projection to zero so resnet starts as identity
    self.linear2 = nnx.Linear(
        hidden_size or residual_size, residual_size,
        kernel_init=nnx.initializers.zeros_init(),
        rngs=rngs)

  def __call__(self, residual: Array) -> Array:
    x = self.layernorm(residual)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    return residual + x


class TransformerLike(Sequential):
  """Alternates recurrent and FFW layers."""

  @staticmethod
  def default_config() -> dict[str, tp.Any]:
    return dict(
        hidden_size=128,
        num_layers=1,
        ffw_multiplier=4,
        recurrent_layer='lstm',
        activation='gelu',
    )

  def __init__(
      self,
      rngs: nnx.Rngs,
      input_size: int,
      hidden_size: int,
      num_layers: int,
      ffw_multiplier: int,
      recurrent_layer: str,
      activation: str,
  ):
    self._hidden_size = hidden_size
    self._num_layers = num_layers
    self._ffw_multiplier = ffw_multiplier

    recurrent_constructor = dict(
        lstm=LSTM,
        gru=GRU,
    )[recurrent_layer]

    activation_fn = dict(
        relu=nnx.relu,
        gelu=nnx.gelu,
        tanh=nnx.tanh,
    )[activation]

    layers = []

    # We need to encode for the first residual
    encoder = nnx.Linear(input_size, hidden_size, rngs=rngs)
    layers.append(FFWWrapper(encoder, output_size=hidden_size))

    for _ in range(num_layers):
      recurrent = ResidualWrapper(
          recurrent_constructor(rngs, hidden_size, hidden_size))
      layers.append(recurrent)

      ffw_layer = ResBlock(
          rngs, hidden_size, hidden_size * ffw_multiplier,
          activation=activation_fn)
      layers.append(FFWWrapper(ffw_layer, output_size=hidden_size))

    super().__init__(layers)


class StateActionNetwork(nnx.Module, abc.ABC):
  """Like a network, but takes StateAction as input."""

  @property
  @abc.abstractmethod
  def output_size(self) -> int:
    """Returns the output size of the network."""

  @abc.abstractmethod
  def dummy(self, shape: Tuple[int, ...]) -> StateAction:
    """Returns a dummy input of the given shape."""

  @abc.abstractmethod
  def encode(self, state_action: StateAction) -> StateAction:
    """Encodes the state and action into a form suitable for the network."""

  @abc.abstractmethod
  def encode_game(self, game: Game) -> Game:
    """Like encode but only the gamestate.

    Useful for agents who already have encoded actions.
    Assumes that the controller_head's encoding matches that of the network.
    """

  @abc.abstractmethod
  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    pass

  @abc.abstractmethod
  def step(
      self,
      state_action: StateAction,
      prev_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    pass

  def step_with_reset(
      self,
      state_action: StateAction,
      reset: Array,
      prev_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    batch_size = reset.shape[0]
    rngs = nnx.Rngs(0)  # TODO: pass rngs properly
    initial_state = where_pytree(
        reset, self.initial_state(batch_size, rngs), prev_state)
    return self.step(state_action, initial_state)

  def _step_with_reset(
      self,
      inputs: tuple[StateAction, Array],
      prev_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    """Used for unroll/scan."""
    state_action, reset = inputs
    return self.step_with_reset(state_action, reset, prev_state)

  def unroll(
      self,
      state_action: StateAction,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    return dynamic_rnn(
        self._step_with_reset, (state_action, reset), initial_state)

  def scan(
      self,
      state_action: StateAction,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    return scan_rnn(
        self._step_with_reset, (state_action, reset), initial_state)


class SimpleEmbedNetwork(StateActionNetwork):
  """Embeds the state and action using provided embedding module."""

  def __init__(
      self,
      embed_game: embed_lib.Embedding[Game, Game],
      embed_state_action: embed_lib.Embedding[StateAction, StateAction],
      network: Network,
  ):
    self._embed_game = embed_game
    self._embed_state_action = embed_state_action
    self._network = network

  @property
  def output_size(self) -> int:
    return self._network.output_size

  def dummy(self, shape: Tuple[int, ...]) -> StateAction:
    return self._embed_state_action.dummy(shape)

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return self._network.initial_state(batch_size, rngs)

  def encode(self, state_action: StateAction) -> StateAction:
    return self._embed_state_action.from_state(state_action)

  def encode_game(self, game: Game) -> Game:
    return self._embed_game.from_state(game)

  def step(
      self,
      state_action: StateAction,
      prev_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    embedded = self._embed_state_action(state_action)
    return self._network.step(embedded, prev_state)

  def unroll(
      self,
      state_action: StateAction,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    embedded = self._embed_state_action(state_action)
    return self._network.unroll(embedded, reset, initial_state)

  def scan(
      self,
      state_action: StateAction,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    embedded = self._embed_state_action(state_action)
    return self._network.scan(embedded, reset, initial_state)


# Factory functions for network construction

CONSTRUCTORS: dict[str, type[Network]] = dict(
    mlp=MLP,
    gru=GRU,
    lstm=LSTM,
    tx_like=TransformerLike,
)

DEFAULT_CONFIG: dict[str, Any] = dict(
    name='mlp',
    **{name: cls.default_config() for name, cls in CONSTRUCTORS.items()}
)


def default_config() -> dict[str, Any]:
  """Returns a fresh copy of the default config.

  Use this method when using as a dataclass field default value.
  """
  return copy.deepcopy(DEFAULT_CONFIG)


# Alias for train_lib compatibility
default_network_config = default_config


def construct_network(
    rngs: nnx.Rngs,
    input_size: int,
    name: str,
    **config,
) -> Network:
  """Construct a network from config.

  Args:
    rngs: Random number generators for initialization.
    input_size: Size of the input features.
    name: Name of the network type (e.g., 'mlp', 'lstm', 'tx_like').
    **config: Network-specific config dicts keyed by network name.

  Returns:
    Constructed network.
  """
  constructor = CONSTRUCTORS[name]
  return constructor(rngs=rngs, input_size=input_size, **config[name])


def build_embed_network(
    rngs: nnx.Rngs,
    embed_config: embed_lib.EmbedConfig,
    num_names: int,
    network_config: dict,
) -> SimpleEmbedNetwork:
  """Build a SimpleEmbedNetwork from config.

  Args:
    rngs: Random number generators for initialization.
    embed_config: Configuration for embeddings.
    num_names: Number of player names for the name embedding.
    network_config: Network configuration dict.

  Returns:
    Constructed SimpleEmbedNetwork.
  """
  embed_game = embed_config.make_game_embedding()
  embed_state_action = embed_lib.get_state_action_embedding(
      embed_game=embed_game,
      embed_action=embed_config.controller.make_embedding(),
      num_names=num_names,
  )
  input_size = embed_state_action.size

  network = construct_network(
      rngs=rngs,
      input_size=input_size,
      **network_config,
  )

  return SimpleEmbedNetwork(
      embed_game=embed_game,
      embed_state_action=embed_state_action,
      network=network,
  )
