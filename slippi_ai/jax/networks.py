import abc
import copy
import functools
import logging
import math
from typing import Any, Callable, Optional, Tuple
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
import tree

from slippi_ai.jax import jax_utils
from slippi_ai.jax import embed as embed_lib
from slippi_ai.data import StateAction
from slippi_ai.types import Controller, Game, Player, Nana, Item

Array = jax.Array

RecurrentState = Any
ArrayTree = tree.StructureKV[str, Array]
Inputs = ArrayTree
Outputs = ArrayTree

def where_pytree(cond: Array, x, y):
  """Like jnp.where but broadcasts cond over pytree leaves."""
  return jax.tree.map(lambda a, b: jax_utils.where(cond, a, b), x, y)

InputTree = tp.TypeVar('InputTree', bound=Inputs)
OutputTree = tp.TypeVar('OutputTree', bound=Outputs)
OutputTree2 = tp.TypeVar('OutputTree2', bound=Outputs)


class Network(nnx.Module, abc.ABC, tp.Generic[InputTree, OutputTree]):

  @property
  @abc.abstractmethod
  def output_size(self) -> int:
    '''Returns the output size of the network.'''

  @abc.abstractmethod
  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    '''Returns the initial state for a batch of size batch_size.'''

  def step(
      self,
      inputs: InputTree,
      prev_state: RecurrentState,
  ) -> Tuple[OutputTree, RecurrentState]:
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
      inputs: InputTree,
      reset: Array,
      prev_state: RecurrentState,
  ) -> Tuple[OutputTree, RecurrentState]:
    batch_size = reset.shape[0]
    rngs = nnx.Rngs(0)  # TODO: pass rngs properly
    initial_state = where_pytree(
        reset, self.initial_state(batch_size, rngs), prev_state)
    return self.step(inputs, initial_state)

  def _step_with_reset(
      self,
      inputs_and_reset: tuple[InputTree, Array],
      prev_state: RecurrentState,
  ) -> Tuple[OutputTree, RecurrentState]:
    """Used for unroll/scan."""
    inputs, reset = inputs_and_reset
    return self.step_with_reset(inputs, reset, prev_state)

  def unroll(
      self,
      inputs: InputTree,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[OutputTree, RecurrentState]:
    '''
    Arguments:
      inputs: (time, batch, x_dim)
      reset: (time, batch)
      initial_state: (batch, state_dim)

    Returns a tuple (outputs, final_state)
      outputs: (time, batch, out_dim)
      final_state: (batch, state_dim)
    '''
    return jax_utils.dynamic_rnn(
        self._step_with_reset, (inputs, reset), initial_state)

  def scan(
      self,
      inputs: InputTree,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[OutputTree, RecurrentState]:
    '''Like unroll but also returns intermediate hidden states.

    Arguments:
      inputs: (time, batch, x_dim)
      reset: (time, batch)
      initial_state: (batch, state_dim)

    Returns a tuple (outputs, hidden_states)
      outputs: (time, batch, out_dim)
      hidden_states: (time, batch, state_dim)
    '''
    return jax_utils.scan_rnn(
        self._step_with_reset, (inputs, reset), initial_state)

  def append(self, other: 'Network[OutputTree, OutputTree2]') -> 'Network[InputTree, OutputTree2]':
    """Appends another network to this one."""
    return Compose(self, other)


class BuildableNetwork(Network[InputTree, OutputTree], abc.ABC):
  """A Network that can be constructed from a config dict."""

  @classmethod
  def name(cls) -> str:
    return cls.__name__

  @classmethod
  def from_config(
      cls,
      rngs: nnx.Rngs,
      input_size: int,
      **config,
  ) -> tp.Self:
    '''Constructs the Network from the given config.'''
    return cls(rngs=rngs, input_size=input_size, **config)

  @classmethod
  @abc.abstractmethod
  def default_config(cls) -> dict[str, tp.Any]:
    '''Returns the default config for this Network.'''

class MLP(BuildableNetwork[Array, Array]):

  @classmethod
  def name(cls) -> str:
    return 'mlp'

  @classmethod
  def default_config(cls) -> dict[str, tp.Any]:
    return dict(
      depth=2,
      width=128,
    )

  def __init__(
      self,
      rngs: nnx.Rngs,
      input_size: int,
      depth: int,
      width: int,
      activation: tp.Callable[[Array], Array] = nnx.relu,
      activation_final: bool = False,
  ):
    self._width = width
    self._mlp = jax_utils.MLP(
        rngs=rngs,
        input_size=input_size,
        features=[width] * depth,
        activation=activation,
        activate_final=activation_final,
    )
    self._output_size = width if depth > 0 else input_size

  @property
  def output_size(self) -> int:
    return self._output_size

  def initial_state(self, batch_size, rngs):
    return ()

  def step(self, inputs, prev_state):
    del prev_state
    return self._mlp(inputs), ()

  def unroll(self, inputs, reset, initial_state):
    del reset, initial_state
    return self._mlp(inputs), ()

class RecurrentWrapper(Network[Array, Array]):
  """Wraps an RNN cell as a Network."""

  def __init__(
      self,
      core: nnx.RNNCellBase,
      output_size: int,
      remat: bool = False,
  ):
    self._core = core
    self._output_size = output_size
    self._remat = remat

  @property
  def output_size(self) -> int:
    return self._output_size

  def initial_state(self, batch_size: int, rngs: nnx.Rngs):
    input_shape = (batch_size, getattr(self._core, 'in_features', 1))
    return self._core.initialize_carry(input_shape, rngs)

  def step(self, inputs, prev_state):
    # flax's RNNCells have the arguments reversed
    core = self._core
    if self._remat:
      core = nnx.remat(self._core, prevent_cse=False)
    next_state, output = core(prev_state, inputs)
    return output, next_state

class FFWWrapper(Network[InputTree, OutputTree]):
  """Wraps a feedforward module as a Network.

  Assumes that the module can handle multiple batch dimensions.
  """

  def __init__(self, module: tp.Callable[[InputTree], OutputTree], output_size: int):
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

class Linear(FFWWrapper[Array, Array]):
  """Linear layer as a Network."""

  def __init__(self, rngs: nnx.Rngs, in_features: int, out_features: int):
    linear = nnx.Linear(in_features, out_features, rngs=rngs)
    super().__init__(linear, output_size=out_features)

class Identity(Network[InputTree, InputTree]):
  """Identity network."""

  def __init__(
      self,
      output_size: Optional[int] = None,
      io_type: Optional[type[InputTree]] = None,
  ) -> None:
    del io_type
    self._output_size = output_size

  @property
  def output_size(self) -> int:
    if self._output_size is not None:
      return self._output_size

    raise NotImplementedError('Identity network has no fixed output size.')

  def initial_state(self, batch_size: int, rngs: nnx.Rngs):
    del batch_size, rngs
    return ()

  def step(self, inputs, prev_state):
    del prev_state
    return inputs, ()

  def unroll(self, inputs, reset, initial_state):
    del reset, initial_state
    return inputs, ()

  def scan(self, inputs, reset, initial_state):
    del reset, initial_state
    return inputs, ()

MidTree = tp.TypeVar('MidTree', bound=ArrayTree)

class Compose(Network[InputTree, OutputTree]):

  def __init__(
      self,
      first: Network[InputTree, MidTree],
      second: Network[MidTree, OutputTree],
  ):
    self._first = first
    self._second = second

  @property
  def output_size(self) -> int:
    return self._second.output_size

  def initial_state(self, batch_size: int, rngs: nnx.Rngs):
    first_state = self._first.initial_state(batch_size, rngs)
    second_state = self._second.initial_state(batch_size, rngs)
    return (first_state, second_state)

  def step(
      self,
      inputs: InputTree,
      prev_state: Tuple[RecurrentState, RecurrentState],
  ) -> Tuple[OutputTree, Tuple[RecurrentState, RecurrentState]]:
    first_state, second_state = prev_state
    mid_outputs, next_first_state = self._first.step(inputs, first_state)
    outputs, next_second_state = self._second.step(mid_outputs, second_state)
    return outputs, (next_first_state, next_second_state)

  def unroll(
      self,
      inputs: InputTree,
      reset: Array,
      initial_state: Tuple[RecurrentState, RecurrentState],
  ) -> Tuple[OutputTree, Tuple[RecurrentState, RecurrentState]]:
    first_state, second_state = initial_state
    mid_outputs, final_first_state = self._first.unroll(inputs, reset, first_state)
    outputs, final_second_state = self._second.unroll(mid_outputs, reset, second_state)
    return outputs, (final_first_state, final_second_state)

  def scan(
      self,
      inputs: InputTree,
      reset: Array,
      initial_state: Tuple[RecurrentState, RecurrentState],
  ) -> Tuple[OutputTree, Tuple[RecurrentState, RecurrentState]]:
    first_state, second_state = initial_state
    mid_outputs, hidden_first_state = self._first.scan(inputs, reset, first_state)
    outputs, hidden_second_state = self._second.scan(mid_outputs, reset, second_state)
    return outputs, (hidden_first_state, hidden_second_state)

class Sequential(Network[InputTree, InputTree]):

  def __init__(self, layers: tp.Sequence[Network[InputTree, InputTree]]):
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


class ResidualWrapper(Network[InputTree, InputTree]):

  def __init__(self, net: Network[InputTree, InputTree]):
    self._net = net

  @property
  def output_size(self) -> int:
    return self._net.output_size

  def initial_state(self, batch_size: int, rngs: nnx.Rngs):
    return self._net.initial_state(batch_size, rngs)

  def _combine(self, inputs: InputTree, outputs: InputTree) -> InputTree:
    return jax.tree.map(jnp.add, inputs, outputs)

  def step(self, inputs, prev_state):
    outputs, next_state = self._net.step(inputs, prev_state)
    return self._combine(inputs, outputs), next_state

  def unroll(self, inputs, reset, initial_state):
    outputs, final_state = self._net.unroll(inputs, reset, initial_state)
    return self._combine(inputs, outputs), final_state

  def scan(self, inputs, reset, initial_state):
    outputs, final_state = self._net.scan(inputs, reset, initial_state)
    return self._combine(inputs, outputs), final_state

class RematWrapper(Network[InputTree, OutputTree]):

  def __init__(self, net: Network[InputTree, OutputTree]):
    self._net = net

  @property
  def output_size(self) -> int:
    return self._net.output_size

  def initial_state(self, batch_size: int, rngs: nnx.Rngs):
    return self._net.initial_state(batch_size, rngs)

  # Note: we can't directly use jax.remat on self._net.step as it raises issues
  # with nnx stuff inside the method. We also have to use nnx.remat as a
  # decorator on the _unbound_ function rather than the method.

  @nnx.remat
  def step(self, inputs, prev_state):
    return self._net.step(inputs, prev_state)

  @nnx.remat
  def unroll(self, inputs, reset, initial_state):
    return self._net.unroll(inputs, reset, initial_state)

  @nnx.remat
  def scan(self, inputs, reset, initial_state):
    return self._net.scan(inputs, reset, initial_state)

class GRU(RecurrentWrapper, BuildableNetwork[Array, Array]):

  @classmethod
  def name(cls) -> str:
    return 'gru'

  @classmethod
  def default_config(cls) -> dict[str, tp.Any]:
    return dict(
      hidden_size=128,
    )

  def __init__(self, rngs: nnx.Rngs, input_size: int, hidden_size: int, remat: bool = False):
    super().__init__(
        nnx.GRUCell(
            in_features=input_size,
            hidden_features=hidden_size,
            rngs=rngs),
        output_size=hidden_size,
        remat=remat)

class LSTM(RecurrentWrapper, BuildableNetwork[Array, Array]):
  @classmethod
  def name(cls) -> str:
    return 'lstm'

  @classmethod
  def default_config(cls) -> dict[str, tp.Any]:
    return dict(
      hidden_size=128,
    )

  def __init__(self, rngs: nnx.Rngs, input_size: int, hidden_size: int, remat: bool = False):
    super().__init__(
        nnx.LSTMCell(
            in_features=input_size,
            hidden_features=hidden_size,
            rngs=rngs),
        output_size=hidden_size,
        remat=remat)


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


class TransformerLike(Sequential, BuildableNetwork[Array, Array]):
  """Alternates recurrent and FFW layers."""

  @classmethod
  def name(cls) -> str:
    return 'tx_like'

  @classmethod
  def default_config(cls) -> dict[str, tp.Any]:
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
      remat: bool = False,
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

    layers: list[Network[Array, Array]] = []

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

    if remat:
      layers = [RematWrapper(layer) for layer in layers]

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
    return jax_utils.dynamic_rnn(
        self._step_with_reset, (state_action, reset), initial_state)

  def scan(
      self,
      state_action: StateAction,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    return jax_utils.scan_rnn(
        self._step_with_reset, (state_action, reset), initial_state)

class EmbedModule(abc.ABC):

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
  def __call__(self, state_action: StateAction) -> Array:
    """Embeds the state and action into an Array suitable for the network."""

class SimpleEmbedModule(EmbedModule):

  def __init__(
      self,
      embed_game: embed_lib.Embedding[Game, Game],
      embed_state_action: embed_lib.Embedding[StateAction, StateAction],
  ):
    self._embed_game = embed_game
    self._embed_state_action = embed_state_action

  @property
  def output_size(self) -> int:
    return self._embed_state_action.size

  def dummy(self, shape: Tuple[int, ...]) -> StateAction:
    return self._embed_state_action.dummy(shape)

  def encode(self, state_action: StateAction) -> StateAction:
    return self._embed_state_action.from_state(state_action)

  def encode_game(self, game: Game) -> Game:
    return self._embed_game.from_state(game)

  def __call__(self, state_action: StateAction) -> Array:
    return self._embed_state_action(state_action)

class SimpleEmbedNetwork(StateActionNetwork):
  """Embeds the state and action using provided embedding module."""

  def __init__(
      self,
      embed_module: EmbedModule,
      network: Network[Array, Array],
      remat: bool = False,
  ):
    self._embed_module = embed_module
    self._network = network
    self._remat = remat

  @property
  def output_size(self) -> int:
    return self._network.output_size

  def dummy(self, shape: Tuple[int, ...]) -> StateAction:
    return self._embed_module.dummy(shape)

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return self._network.initial_state(batch_size, rngs)

  def encode(self, state_action: StateAction) -> StateAction:
    return self._embed_module.encode(state_action)

  def encode_game(self, game: Game) -> Game:
    return self._embed_module.encode_game(game)

  def _embed(self, state_action: StateAction) -> Array:
    if self._remat:
      return nnx.remat(apply)(self._embed_module, state_action)
    return self._embed_module(state_action)

  def step(
      self,
      state_action: StateAction,
      prev_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    embedded = self._embed(state_action)
    output, next_state = self._network.step(embedded, prev_state)
    assert isinstance(output, Array)
    return output, next_state

  def unroll(
      self,
      state_action: StateAction,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    embedded = self._embed(state_action)
    output, final_state = self._network.unroll(embedded, reset, initial_state)
    assert isinstance(output, Array)
    return output, final_state

  def scan(
      self,
      state_action: StateAction,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    embedded = self._embed(state_action)
    output, final_state = self._network.scan(embedded, reset, initial_state)
    assert isinstance(output, Array)
    return output, final_state

# TODO: unify with controller_heads
class ControllerRNN(nnx.Module):
  """Embed controller using an RNN over its components."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      embed_controller: embed_lib.StructEmbedding[Controller],
      hidden_size: int,
      rnn_cell: str = 'lstm',
  ):
    self._embed_controller = embed_controller
    embed_struct = embed_controller.map(lambda e: e)
    self._embed_flat: list[embed_lib.Embedding] = list(embed_controller.flatten(embed_struct))
    if len(self._embed_flat) == 0:
      raise ValueError('ControllerRNN requires at least one embedded component.')

    rnn_constructor = dict(
        lstm=nnx.LSTMCell,
        gru=nnx.GRUCell,
    )[rnn_cell]

    cells: list[nnx.RNNCellBase] = []

    for e in self._embed_flat:
      cells.append(rnn_constructor(
          in_features=e.size,
          hidden_features=hidden_size,
          rngs=rngs,
      ))

    self._initial_state_fn = cells[0].initialize_carry
    self._cells = nnx.List(cells)

  def __call__(self, controller: Controller) -> Array:
    input_shape = controller.main_stick.x.shape + (self._embed_flat[0].size,)

    # TODO: pass rngs properly? maybe reuse from __init__?
    hidden_state = self._initial_state_fn(input_shape, rngs=nnx.Rngs(0))

    for cell, embed, component in zip(
        self._cells, self._embed_flat,
        self._embed_controller.flatten(controller)):
      hidden_state, output = cell(hidden_state, embed(component))

    return output  # type: ignore

Group = Array | tp.Sequence[Array]

class GroupNetwork(Network[list[InputTree], list[OutputTree]]):

  def __init__(self, networks: list[Network[InputTree, OutputTree]]):
    self._networks = nnx.List(networks)

  @property
  def output_size(self):
    raise NotImplementedError()

  def initial_state(self, batch_size: int, rngs: nnx.Rngs):
    return [
        net.initial_state(batch_size, rngs)
        for net in self._networks]

  def step(
      self,
      inputs: list[InputTree],
      prev_state: RecurrentState,
  ) -> Tuple[list[OutputTree], RecurrentState]:
    outputs = []
    next_states = []
    for net, inp, state in zip(self._networks, inputs, prev_state):
      out, next_state = net.step(inp, state)
      outputs.append(out)
      next_states.append(next_state)
    return outputs, next_states

  def unroll(
      self,
      inputs: list[InputTree],
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[list[OutputTree], RecurrentState]:
    outputs = []
    final_states = []
    for net, inp, state in zip(self._networks, inputs, initial_state):
      out, final_state = net.unroll(inp, reset, state)
      outputs.append(out)
      final_states.append(final_state)
    return outputs, final_states

  def scan(
      self,
      inputs: list[InputTree],
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[list[OutputTree], RecurrentState]:
    outputs = []
    final_states = []
    for net, inp, state in zip(self._networks, inputs, initial_state):
      out, final_state = net.scan(inp, reset, state)
      outputs.append(out)
      final_states.append(final_state)
    return outputs, final_states

P = tp.ParamSpec('P')
T = tp.TypeVar('T')

def apply(f: tp.Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
  return f(*args, **kwargs)

class StackedGroupNetwork(Network[InputTree, OutputTree]):

  def __init__(
      self,
      builder: tp.Callable[[nnx.Rngs], Network[InputTree, OutputTree]],
      width: int,
      rngs: nnx.Rngs,
      axis: int = -2,
  ):
    self.width = width
    self.axis = axis

    stacked_builder = nnx.vmap(builder, in_axes=0, out_axes=0)
    split_rngs = rngs.fork(split=width)
    self._networks = stacked_builder(split_rngs)

  @property
  def output_size(self) -> int:
    return self._networks.output_size

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    @nnx.vmap(in_axes=(0, 0), out_axes=1, axis_size=self.width)
    def init_single(net: Network, r: nnx.Rngs) -> RecurrentState:
      return net.initial_state(batch_size, r)
    return init_single(self._networks, rngs.fork(split=self.width))

  def step(
      self,
      inputs: InputTree,
      prev_state: RecurrentState,
  ) -> Tuple[OutputTree, RecurrentState]:
    @nnx.vmap(in_axes=(0, self.axis, 1), out_axes=(self.axis, 1))
    def step_single(
        net: Network[InputTree, OutputTree],
        inp: InputTree,
        state: RecurrentState,
    ) -> Tuple[OutputTree, RecurrentState]:
      return net.step(inp, state)

    return step_single(self._networks, inputs, prev_state)

  def unroll(
      self,
      inputs: InputTree,
      reset: jax.Array,
      initial_state: RecurrentState,
  ) -> tuple[OutputTree, RecurrentState]:

    @nnx.vmap(in_axes=(0, self.axis, 1), out_axes=(self.axis, 1))
    def unroll_single(
        net: Network[InputTree, OutputTree],
        inp: InputTree,
        state: RecurrentState,
    ) -> Tuple[OutputTree, RecurrentState]:
      return net.unroll(inp, reset, state)

    return unroll_single(self._networks, inputs, initial_state)

  def scan(
      self,
      inputs: InputTree,
      reset: jax.Array,
      initial_state: RecurrentState,
  ) -> tuple[OutputTree, RecurrentState]:
    # Note: hidden_state out axis is 2 because it returns all intermediate states
    @nnx.vmap(in_axes=(0, self.axis, 1), out_axes=(self.axis, 2))
    def scan_single(
        net: Network[InputTree, OutputTree],
        inp: InputTree,
        state: RecurrentState,
    ) -> Tuple[OutputTree, RecurrentState]:
      return net.scan(inp, reset, state)

    return scan_single(self._networks, inputs, initial_state)

class StackedSequential(Network[InputTree, InputTree]):
  """Maintains a stack of networks, applied sequentially with scan."""

  def __init__(
      self,
      builder: tp.Callable[[nnx.Rngs], Network[InputTree, InputTree]],
      depth: int,
      rngs: nnx.Rngs,
  ):
    self.depth = depth

    stacked_builder = nnx.vmap(builder, in_axes=0, out_axes=0)
    split_rngs = rngs.fork(split=depth)
    self._networks = stacked_builder(split_rngs)

  @property
  def output_size(self) -> int:
    return self._networks.output_size

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:

    @nnx.vmap(in_axes=(0, 0), out_axes=1, axis_size=self.depth)
    def init_single(net: Network, r: nnx.Rngs) -> RecurrentState:
      return net.initial_state(batch_size, r)

    return init_single(self._networks, rngs.fork(split=self.depth))

  def step(
      self,
      inputs: InputTree,
      prev_state: RecurrentState,  # [B, depth, ...]
  ) -> tuple[InputTree, RecurrentState]:

    @nnx.scan(
        in_axes=(0, nnx.Carry, 1),
        out_axes=(nnx.Carry, 1),
        length=self.depth,
    )
    def step_single(
        net: Network[InputTree, InputTree],
        inp: InputTree,
        state: RecurrentState,
    ) -> tuple[InputTree, RecurrentState]:
      return net.step(inp, state)

    return step_single(self._networks, inputs, prev_state)

  def unroll(
      self,
      inputs: InputTree,
      reset: jax.Array,
      initial_state: RecurrentState,
  ) -> tuple[InputTree, RecurrentState]:

    @nnx.scan(
        in_axes=(0, nnx.Carry, 1),
        out_axes=(nnx.Carry, 1),
        length=self.depth,
    )
    def unroll_single(
        net: Network[InputTree, InputTree],
        inp: InputTree,
        state: RecurrentState,
    ) -> tuple[InputTree, RecurrentState]:
      return net.unroll(inp, reset, state)

    return unroll_single(self._networks, inputs, initial_state)

  def scan(
      self,
      inputs: InputTree,
      reset: jax.Array,
      initial_state: RecurrentState,
  ) -> tuple[InputTree, RecurrentState]:

    @nnx.scan(
        in_axes=(0, nnx.Carry, 1),
        out_axes=(nnx.Carry, 2),
        length=self.depth,
    )
    def scan_single(
        net: Network[InputTree, InputTree],
        inp: InputTree,
        state: RecurrentState,
    ) -> tuple[InputTree, RecurrentState]:
      return net.scan(inp, reset, state)

    return scan_single(self._networks, inputs, initial_state)


class IndependentMultiHeadAttention(nnx.Module):

  def __init__(
      self,
      num_groups: int,
      features: int,
      num_heads: int,
      rngs: nnx.Rngs,
      normalize_qk: bool = True,
      use_out_linear: bool = True,
  ):
    self.num_groups = num_groups
    self.num_heads = num_heads
    self.feature_dim = features
    self.normalize_qk = normalize_qk
    self.use_out_linear = use_out_linear

    if features % num_heads != 0:
      raise ValueError('features must be divisible by num_heads')
    self.head_dim = features // num_heads

    if self.normalize_qk:
      self.query_ln = nnx.LayerNorm(
        self.head_dim,
        use_bias=False,
        rngs=rngs,
      )
      self.key_ln = nnx.LayerNorm(
        self.head_dim,
        use_bias=False,
        rngs=rngs,
      )

    # Create separate QKV linears for each group
    @nnx.vmap(in_axes=0, out_axes=0)
    def make_qkv_linear(r: nnx.Rngs) -> nnx.LinearGeneral:
        return nnx.LinearGeneral(
            in_features=features,
            out_features=(3, num_heads, self.head_dim),
            rngs=r,
        )
    self.qkv_linear = make_qkv_linear(rngs.fork(split=num_groups))

    if self.use_out_linear:
      # Create separate output linears for each group
      @nnx.vmap(in_axes=0, out_axes=0)
      def make_output_linear(r: nnx.Rngs) -> nnx.LinearGeneral:
        return nnx.LinearGeneral(
            axis=(-2, -1),
            in_features=(num_heads, self.head_dim),
            out_features=features,
            rngs=r,
        )

      self.output_linear = make_output_linear(rngs.fork(split=num_groups))

  def __call__(self, inputs: Array) -> Array:
    assert inputs.shape[-2:] == (self.num_groups, self.feature_dim)

    # Would einsum be better here?
    @nnx.vmap(in_axes=(0, -2), out_axes=-4)
    def qkv_single(qkv_linear: nnx.LinearGeneral, x: Array) -> Array:
      return qkv_linear(x)

    qkv = qkv_single(self.qkv_linear, inputs)  # (..., num_groups, 3, num_heads, head_dim)

    queries, keys, values = jnp.unstack(qkv, axis=-3)

    if self.normalize_qk:
      queries = self.query_ln(queries)
      keys = self.key_ln(keys)

    head_outputs = nnx.dot_product_attention(queries, keys, values)  # (..., num_groups, num_heads, head_dim)

    if self.use_out_linear:
      @nnx.vmap(in_axes=(0, -3), out_axes=-2)
      def output_single(output_linear: nnx.LinearGeneral, x: Array) -> Array:
        # x: (..., num_heads, head_dim)
        return output_linear(x)  # (..., feature_dim)

      outputs = output_single(self.output_linear, head_outputs)  # (..., num_groups, feature_dim)
    else:
      # Concatenate heads back into feature dimension
      outputs = head_outputs.reshape(head_outputs.shape[:-2] + (self.feature_dim,))

    return outputs

class FrameTransformer(StateActionNetwork):

  @classmethod
  def name(cls) -> str:
    return 'frame_tx'

  @classmethod
  def default_config(cls) -> dict[str, tp.Any]:
    return dict(
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        use_self_nana=False,
        rnn_cell='lstm',

        remat_layers=False,
        remat_controller_rnn=True,
        stacked_rnns=True,
        remat_rnns=False,
        stack_layers=False,

        normalize_qk=True,
        use_out_linear=True,

        input_ffw=False,
        skip_rnn=False,
        skip_attention=False,
    )

  def __init__(
      self,
      rngs: nnx.Rngs,
      embed_state_action: embed_lib.StructEmbedding[StateAction],
      hidden_size: int,
      num_heads: int,
      num_layers: int,
      use_self_nana: bool = False,
      rnn_cell: str = 'lstm',
      # Stacked RNNs are faster but use more memory
      stacked_rnns: bool = True,
      # Remat each layer to save memory
      remat_layers: bool = False,
      # The controller_rnn is particularly important to remat
      remat_controller_rnn: bool = True,
      # Remat RNN cells during unroll to save memory
      remat_rnns: bool = False,
      # Faster compilation
      stack_layers: bool = False,

      normalize_qk: bool = True,
      use_out_linear: bool = True,

      # Use a Linear layer instead of an RNN for the initial embedding
      input_ffw: bool = False,

      # Below are just for testing memory usage
      skip_rnn: bool = False,
      skip_attention: bool = False,
  ):
    self._embed_state_action = embed_state_action
    self._hidden_size = hidden_size
    self._use_self_nana = use_self_nana
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._remat_layers = remat_layers
    self._remat_controller_rnn = remat_controller_rnn

    embed_state_action_shallow = embed_state_action.map_shallow(
        lambda e: e)

    self._embed_game = tp.cast(
        embed_lib.StructEmbedding[Game],
        embed_state_action_shallow.state)
    self._embed_controller = tp.cast(
        embed_lib.StructEmbedding[Controller],
        embed_state_action_shallow.action)

    self._controller_rnn = ControllerRNN(
        rngs=rngs,
        embed_controller=self._embed_controller,
        hidden_size=hidden_size,
        rnn_cell=rnn_cell,
    )

    dummy_state_action = embed_state_action.dummy(())
    # Need to use eval_shape_method because make_groups uses a ControllerRNN
    group_shapes = tp.cast(
        list[jax.ShapeDtypeStruct],
        jax_utils.eval_shape_method(self._make_groups, dummy_state_action))
    self._num_groups = len(group_shapes)

    for g in group_shapes:
      assert len(g.shape) == 1
    group_sizes: list[int] = [g.shape[0] for g in group_shapes]

    rnn_constructor = dict(
        lstm=LSTM,
        gru=GRU,
    )[rnn_cell]
    rnn_constructor = functools.partial(rnn_constructor, remat=remat_rnns)

    def stack_groups_fn(xs: list[Array]) -> Array:
      assert len(xs) == len(group_sizes)
      return jnp.stack(xs, axis=-2)
    stack_groups_net = FFWWrapper(stack_groups_fn, output_size=hidden_size)

    def unstack_groups_fn(x: Array) -> list[Array]:
      assert x.shape[-2] == len(group_sizes)
      return list(jnp.unstack(x, axis=-2))
    unstack_groups_net = FFWWrapper(unstack_groups_fn, output_size=hidden_size)

    def recurrent_layer(rngs: nnx.Rngs, stacked: bool):
      make_rnn = lambda r: rnn_constructor(r, input_size=hidden_size, hidden_size=hidden_size)

      if stacked:
        per_group_rnn = StackedGroupNetwork(
            builder=make_rnn,
            width=len(group_sizes),
            rngs=rngs,
            axis=-2,
        )
        return ResidualWrapper(per_group_rnn)

      net = unstack_groups_net
      net = net.append(GroupNetwork([
          ResidualWrapper(make_rnn(rngs))
          for _ in group_sizes]))
      net = net.append(stack_groups_net)
      return net

    # Initial per-attention RNNs take heterogeneous group sizes
    def initial_embed_net(size: int) -> Network[Array, Array]:
      if input_ffw:
        return Linear(rngs, in_features=size, out_features=hidden_size)

      return rnn_constructor(rngs, input_size=size, hidden_size=hidden_size)

    initial_embed = GroupNetwork([initial_embed_net(s) for s in group_sizes])
    net = initial_embed.append(stack_groups_net)

    if remat_layers:
      net = RematWrapper(net)

    def make_block(rngs: nnx.Rngs) -> Network[Array, Array]:
      net = Identity(output_size=hidden_size, io_type=Array)

      if not skip_attention:
        attention = IndependentMultiHeadAttention(
            num_groups=self._num_groups,
            features=hidden_size,
            num_heads=num_heads,
            rngs=rngs,
            normalize_qk=normalize_qk,
            use_out_linear=use_out_linear,
        )
        attention = ResidualWrapper(FFWWrapper(attention, output_size=hidden_size))
        if remat_layers:
          attention = RematWrapper(attention)

        net = net.append(attention)

      if not skip_rnn:
        rec_layer = recurrent_layer(rngs, stacked=stacked_rnns)
        if remat_layers:
          rec_layer = RematWrapper(rec_layer)

        net = net.append(rec_layer)

      return net

    if stack_layers:
      stacked_net = StackedSequential(
          builder=make_block,
          depth=num_layers,
          rngs=rngs,
      )
      net = net.append(stacked_net)
    else:
      for _ in range(num_layers):
        net = net.append(make_block(rngs.fork()))

    self._network = net

  @property
  def output_size(self) -> int:
    return self._hidden_size

  def dummy(self, shape: Tuple[int, ...]) -> StateAction:
    return self._embed_state_action.dummy(shape)

  def encode(self, state_action: StateAction) -> StateAction:
    return self._embed_state_action.from_state(state_action)

  def encode_game(self, game: Game) -> Game:
    return self._embed_game.from_state(game)

  def _player_or_nana_groups(self, player: Player | Nana) -> list[Group]:
    groups = [
        player.percent,
        player.facing,
        [player.x, player.y],
        player.action,
        player.invulnerable,
        player.character,
        player.jumps_left,
        player.shield_strength,
        player.on_ground,
    ]

    if isinstance(player, Nana):
      groups.append(player.exists)

    return groups

  def _player_groups(self, player: Player, with_nana: bool):
    groups = self._player_or_nana_groups(player)

    if with_nana:
      nana_groups = self._player_or_nana_groups(player.nana)
      groups.extend(nana_groups)

    return groups

  def _apply_controller_rnn(self, controller: Controller) -> Array:
    if self._remat_controller_rnn:
      return nnx.remat(apply)(self._controller_rnn, controller)
    return self._controller_rnn(controller)

  def _make_groups(self, state_action: StateAction) -> list[Array]:
    state_action_embed = self._embed_state_action.map(
        lambda e, v: e(v), state_action)

    groups = []
    game = state_action_embed.state

    # Assuming we're training one character vs all others,
    # we only include nana for ourselves if we're training as ICs,
    # but we always include nana for the opponent.
    groups.extend(
        self._player_groups(
            game.p0,
            with_nana=self._use_self_nana,
        ))
    groups.extend(
        self._player_groups(
            game.p1,
            with_nana=True,
        ))

    groups.extend([
        game.stage,
        game.randall,
        game.fod_platforms,
    ])

    # Each item is its own group
    groups.extend(game.items)

    groups.append(tp.cast(Array, state_action_embed.name))

    # Last group is the controller
    groups.append(self._apply_controller_rnn(state_action.action))

    def maybe_concat(g: Group) -> Array:
      if isinstance(g, Array):
        return g
      else:
        return jnp.concatenate(g, axis=-1)

    return [maybe_concat(g) for g in groups]

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return self._network.initial_state(batch_size, rngs)

  def _extract_output(self, outputs: Outputs) -> Array:
    assert isinstance(outputs, Array)
    assert outputs.shape[-1] == self._hidden_size
    assert outputs.shape[-2] == self._num_groups
    return outputs[..., -1, :]

  def step(
      self,
      state_action: StateAction,
      prev_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    groups = self._make_groups(state_action)
    outputs, next_state = self._network.step(groups, prev_state)
    return self._extract_output(outputs), next_state

  def unroll(
      self,
      state_action: StateAction,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    groups = self._make_groups(state_action)
    outputs, final_state = self._network.unroll(groups, reset, initial_state)
    return self._extract_output(outputs), final_state

  def scan(
      self,
      state_action: StateAction,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    groups = self._make_groups(state_action)
    outputs, final_state = self._network.scan(groups, reset, initial_state)
    return self._extract_output(outputs), final_state

class MultiEmbed(nnx.Module):

  def __init__(self, sizes: tuple[int, ...], features: int, rngs: nnx.Rngs):
    self._sizes = sizes
    total_size = math.prod(sizes)
    self._embed = nnx.Embed(
        num_embeddings=total_size,
        features=features,
        rngs=rngs,
    )

  def __call__(self, *inputs: Array) -> Array:
    assert len(inputs) == len(self._sizes)

    index = jnp.zeros_like(inputs[0])
    valid = jnp.full(index.shape, True)
    for size, inp in zip(self._sizes, inputs):
      valid &= (inp >= 0) & (inp < size)
      index *= size
      index += inp

    embed = jnp.where(valid[..., None], self._embed(index), 0)
    return embed

PlayerOrNana = tp.TypeVar('PlayerOrNana', Player, Nana)

class EnhancedEmbedModule(nnx.Module, EmbedModule):

  @classmethod
  def default_config(cls) -> dict[str, tp.Any]:
    return dict(
        hidden_size=128,
        item_mlp_layers=2,
        rnn_cell='lstm',
        use_self_nana=True,
        use_controller_rnn=False,
    )

  def __init__(
      self,
      rngs: nnx.Rngs,
      embed_state_action: embed_lib.StructEmbedding[StateAction],
      hidden_size: int,
      item_mlp_layers: int,
      rnn_cell: str = 'lstm',
      use_self_nana: bool = True,
      use_controller_rnn: bool = False,
  ):
    self._embed_state_action = embed_state_action
    self._use_self_nana = use_self_nana
    embed_state_action_deep = embed_state_action.map(lambda e: e)
    embed_state_action_shallow = embed_state_action.map_shallow(lambda e: e)

    self._embed_game = tp.cast(
        embed_lib.StructEmbedding[Game],
        embed_state_action_shallow.state)
    self._embed_controller = tp.cast(
        embed_lib.StructEmbedding[Controller],
        embed_state_action_shallow.action)

    # Embed all items using the same Linear layer
    self._item_embedding = embed_lib.struct_embedding_from_nt(
        'item',
        tp.cast(Item, embed_state_action_deep.state.items.item_0))
    self._item_linear = nnx.Linear(
        in_features=self._item_embedding.size,
        out_features=hidden_size,
        rngs=rngs,
    )
    self._item_mlp = jax_utils.MLP(
        rngs=rngs,
        input_size=self._item_embedding.size,
        features=[hidden_size] * item_mlp_layers,
    )

    self._use_controller_rnn = use_controller_rnn
    if use_controller_rnn:
      self._controller_rnn = ControllerRNN(
          rngs=rngs,
          embed_controller=self._embed_controller,
          hidden_size=hidden_size,
          rnn_cell=rnn_cell,
      )

    # Assumes that p0 and p1 have the same embedding structure
    embed_char = tp.cast(
        embed_lib.OneHotEmbedding,
        embed_state_action_deep.state.p0.character)
    embed_char.size

    self._embed_char = nnx.Embed(
        num_embeddings=embed_char.size,
        features=hidden_size,
        rngs=rngs,
    )

    embed_action = tp.cast(
        embed_lib.StructEmbedding,
        embed_state_action_deep.state.p0.action)
    self._embed_action = nnx.Embed(
        num_embeddings=embed_action.size,
        features=hidden_size,
        rngs=rngs,
    )

    self._embed_char_action = MultiEmbed(
        sizes=(embed_char.size, embed_action.size),
        features=hidden_size,
        rngs=rngs,
    )

    output_shape = jax_utils.eval_shape_method(self.__call__, embed_state_action.dummy(()))
    assert output_shape.ndim == 1
    self._output_size = output_shape.shape[0]

  @property
  def output_size(self) -> int:
    return self._output_size

  def dummy(self, shape: Tuple[int, ...]) -> StateAction:
    return self._embed_state_action.dummy(shape)

  def encode(self, state_action: StateAction) -> StateAction:
    return self._embed_state_action.from_state(state_action)

  def encode_game(self, game: Game) -> Game:
    return self._embed_game.from_state(game)

  def _embed_player_or_nana(self, raw: PlayerOrNana, default: PlayerOrNana) -> Array:
    action = self._embed_action(raw.action) + self._embed_char_action(raw.character, raw.action)
    parts = [
        default.percent,
        default.facing,
        default.x, default.y,
        action,
        default.invulnerable,
        self._embed_char(raw.character),
        default.jumps_left,
        default.shield_strength,
        default.on_ground,
    ]

    if isinstance(default, Nana):
      parts.append(default.exists)

    return jnp.concatenate(parts, axis=-1)

  def _embed_player(self, raw: Player, default: Player, with_nana: bool) -> Array:
    parts = [self._embed_player_or_nana(raw, default)]

    if with_nana:
      parts.append(self._embed_player_or_nana(raw.nana, default.nana))
    return jnp.concatenate(parts, axis=-1)

  def __call__(self, state_action: StateAction) -> Array:
    raw_game = state_action.state
    default_state_action_embed = self._embed_state_action.map(
        lambda e, v: e(v), state_action)
    default_game = default_state_action_embed.state

    parts = [
        self._embed_player(
            raw_game.p0, default_game.p0,
            with_nana=self._use_self_nana,
        ),
        self._embed_player(
            raw_game.p1, default_game.p1,
            with_nana=True,
        ),
        default_game.stage,
        *default_game.randall,
        *default_game.fod_platforms,
    ]

    # Process items as a batch
    stacked_items: Item = jax.tree.map(
      lambda *args: jnp.stack(args, axis=-1), *raw_game.items)
    item_embed = self._item_embedding(stacked_items)
    item_embed = self._item_linear(item_embed)
    items_embed = jnp.sum(item_embed, axis=-2)  # Sum over items
    parts.append(items_embed)

    parts.append(tp.cast(Array, default_state_action_embed.name))

    if self._use_controller_rnn:
      parts.append(self._controller_rnn(state_action.action))
    else:
      parts.append(self._embed_controller(state_action.action))

    return jnp.concatenate(parts, axis=-1)

# Factory functions for network construction

_simple_networks: list[type[BuildableNetwork[Array, Array]]] = [
    MLP, LSTM, GRU, TransformerLike,
]
SIMPLE_CONSTRUCTORS = {
    cls.name(): cls for cls in _simple_networks
}

OTHER_CONSTRUCTORS = [
    FrameTransformer,
]

NAME_TO_CONSTRUCTOR: dict[str, type[StateActionNetwork]] = {
    cls.name(): cls for cls in OTHER_CONSTRUCTORS
}

DEFAULT_CONFIG: dict[str, Any] = dict(
    name='mlp',
    **{name: cls.default_config() for name, cls in SIMPLE_CONSTRUCTORS.items()},
)
for cls in OTHER_CONSTRUCTORS:
  DEFAULT_CONFIG[cls.name()] = cls.default_config()

DEFAULT_CONFIG['embed'] = dict(
    name='simple',
    simple={},
    enhanced=EnhancedEmbedModule.default_config(),
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
  constructor = SIMPLE_CONSTRUCTORS[name]
  return constructor(rngs=rngs, input_size=input_size, **config[name])

def build_embed_module(
    rngs: nnx.Rngs,
    config: dict,
    embed_state_action: embed_lib.StructEmbedding[StateAction],
    embed_game: embed_lib.StructEmbedding[Game],
):
  name = config['name']
  logging.info('Using embed module: %s', name)

  if name == 'simple':
    return SimpleEmbedModule(
        embed_game=embed_game,
        embed_state_action=embed_state_action,
    )

  if name == 'enhanced':
    return EnhancedEmbedModule(
        rngs=rngs,
        embed_state_action=embed_state_action,
        **config['enhanced'],
    )

  raise ValueError(f'Unknown embed module name: {name}')

def build_embed_network(
    rngs: nnx.Rngs,
    embed_config: embed_lib.EmbedConfig,
    num_names: int,
    network_config: dict,
) -> StateActionNetwork:
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

  name = network_config['name']

  if name in SIMPLE_CONSTRUCTORS:
    embed_module = build_embed_module(
        rngs=rngs,
        config=network_config['embed'],
        embed_state_action=embed_state_action,
        embed_game=embed_game,
    )

    network = construct_network(
        rngs=rngs,
        input_size=embed_module.output_size,
        **network_config,
    )

    return SimpleEmbedNetwork(
        embed_module=embed_module,
        network=network,
    )

  if name in NAME_TO_CONSTRUCTOR:
    return NAME_TO_CONSTRUCTOR[name](
        rngs=rngs,
        embed_state_action=embed_state_action,
        **network_config[name],
    )

  raise ValueError(f'Unknown network name: {name}')