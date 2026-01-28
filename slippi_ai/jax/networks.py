import abc
import copy
from typing import Any, Callable, Optional, Tuple
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
import tree

from slippi_ai.jax import jax_utils
from slippi_ai.jax import embed as embed_lib
from slippi_ai.data import StateAction
from slippi_ai.types import Controller, Game, Player, Nana

Array = jax.Array

RecurrentState = Any
Inputs = tree.StructureKV[str, Array]
Outputs = tree.StructureKV[str, Array]

def where_pytree(cond: Array, x, y):
  """Like jnp.where but broadcasts cond over pytree leaves."""
  return jax.tree.map(lambda a, b: jax_utils.where(cond, a, b), x, y)

InputTree = tp.TypeVar('InputTree')
OutputTree = tp.TypeVar('OutputTree')

def dynamic_rnn(
    cell_fn: Callable[[InputTree, RecurrentState], Tuple[OutputTree, RecurrentState]],
    inputs: InputTree,
    initial_state: RecurrentState,
) -> Tuple[OutputTree, RecurrentState]:
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
    cell_fn: Callable[[InputTree, RecurrentState], Tuple[OutputTree, RecurrentState]],
    inputs: InputTree,
    initial_state: RecurrentState,
) -> Tuple[OutputTree, RecurrentState]:
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
  ) -> Tuple[Outputs, RecurrentState]:
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
  ) -> Tuple[Outputs, RecurrentState]:
    batch_size = reset.shape[0]
    rngs = nnx.Rngs(0)  # TODO: pass rngs properly
    initial_state = where_pytree(
        reset, self.initial_state(batch_size, rngs), prev_state)
    return self.step(inputs, initial_state)

  def _step_with_reset(
      self,
      inputs_and_reset: tuple[Inputs, Array],
      prev_state: RecurrentState,
  ) -> Tuple[Outputs, RecurrentState]:
    """Used for unroll/scan."""
    inputs, reset = inputs_and_reset
    return self.step_with_reset(inputs, reset, prev_state)

  def unroll(
      self,
      inputs: Inputs,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Outputs, RecurrentState]:
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
  ) -> Tuple[Outputs, RecurrentState]:
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
    assert isinstance(inputs, Array)
    # flax's RNNCells have the arguments reversed
    next_state, output = self._core(prev_state, inputs)
    return output, next_state


class FFWWrapper(Network):
  """Wraps a feedforward module as a Network.

  Assumes that the module can handle multiple batch dimensions.
  """

  def __init__(self, module: tp.Callable[[Inputs], Outputs], output_size: int):
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

  def _combine(self, inputs, outputs):
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
    output, next_state = self._network.step(embedded, prev_state)
    assert isinstance(output, Array)
    return output, next_state

  def unroll(
      self,
      state_action: StateAction,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    embedded = self._embed_state_action(state_action)
    output, final_state = self._network.unroll(embedded, reset, initial_state)
    assert isinstance(output, Array)
    return output, final_state

  def scan(
      self,
      state_action: StateAction,
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    embedded = self._embed_state_action(state_action)
    output, final_state = self._network.scan(embedded, reset, initial_state)
    assert isinstance(output, Array)
    return output, final_state

Group = Array | tp.Sequence[Array]

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

class GroupNetwork(Network):

  def __init__(self, networks: list[Network]):
    self._networks = nnx.List(networks)

  @property
  def output_size(self) -> list[int]:
    return [net.output_size for net in self._networks]

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> Any:
    return [
        net.initial_state(batch_size, rngs)
        for net in self._networks]

  def step(
      self,
      inputs: list[Array],
      prev_state: Any,
  ) -> Tuple[list[Array], Any]:
    outputs = []
    next_states = []
    for net, inp, state in zip(self._networks, inputs, prev_state):
      out, next_state = net.step(inp, state)
      outputs.append(out)
      next_states.append(next_state)
    return outputs, next_states

  def unroll(
      self,
      inputs: list[Array],
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[list[Array], RecurrentState]:
    outputs = []
    final_states = []
    for net, inp, state in zip(self._networks, inputs, initial_state):
      out, final_state = net.unroll(inp, reset, state)
      outputs.append(out)
      final_states.append(final_state)
    return outputs, final_states

  def scan(
      self,
      inputs: list[Array],
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[list[Array], RecurrentState]:
    outputs = []
    final_states = []
    for net, inp, state in zip(self._networks, inputs, initial_state):
      out, final_state = net.scan(inp, reset, state)
      outputs.append(out)
      final_states.append(final_state)
    return outputs, final_states

class FrameTransformer(StateActionNetwork):
  name = 'frame_tx'

  @classmethod
  def default_config(cls) -> dict[str, tp.Any]:
    return dict(
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        use_self_nana=False,
        rnn_cell='lstm',
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
  ):
    self._embed_state_action = embed_state_action
    self._hidden_size = hidden_size
    self._use_self_nana = use_self_nana
    self._num_layers = num_layers
    self._num_heads = num_heads
    # self._embed_struct = embed_state_action.map(lambda e: e)

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
    group_shapes: list = jax.eval_shape(self._make_groups, dummy_state_action)

    for g in group_shapes:
      assert isinstance(g, jax.ShapeDtypeStruct)
      assert len(g.shape) == 1
    group_sizes: list[int] = [g.shape[0] for g in group_shapes]

    layers = []

    rnn_constructor = dict(
        lstm=LSTM,
        gru=GRU,
    )[rnn_cell]

    # Initial per-attention RNNs take heterogeneous group sizes
    layers.append(GroupNetwork([
        rnn_constructor(rngs, input_size=s, hidden_size=hidden_size)
        for s in group_sizes]))

    # Stack groups into a single tensor
    stack_groups = FFWWrapper(
        lambda xs: jnp.stack(xs, axis=-2),
        output_size=hidden_size)

    for _ in range(num_layers):
      layers.append(stack_groups)

      attention = nnx.MultiHeadAttention(
          num_heads=num_heads,
          in_features=hidden_size,
          qkv_features=hidden_size,
          out_features=hidden_size,
          rngs=rngs,
          decode=False,
      )
      layers.append(ResidualWrapper(FFWWrapper(attention, output_size=hidden_size)))

      per_group_rnn = GroupNetwork([
          ResidualWrapper(rnn_constructor(rngs, input_size=hidden_size, hidden_size=hidden_size))
          for _ in group_sizes])

      # TODO: optimize this with a single vmap
      layers.extend([
          FFWWrapper(lambda x: jnp.unstack(x, axis=-2), output_size=hidden_size),
          per_group_rnn,
      ])

    self._network = Sequential(layers)

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

  def _make_groups(self, state_action: StateAction) -> list[Array]:
    state_action_embed = self._embed_state_action.map(
        lambda e, v: e(v), state_action)

    groups = []
    game = state_action_embed.state

    for player in [game.p0, game.p1]:
      groups.extend(
          self._player_groups(
              player,
              with_nana=self._use_self_nana,
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
    groups.append(self._controller_rnn(state_action.action))

    def maybe_concat(g: Group) -> Array:
      if isinstance(g, Array):
        return g
      else:
        return jnp.concatenate(g, axis=-1)

    return [maybe_concat(g) for g in groups]

  def initial_state(self, batch_size: int, rngs: nnx.Rngs) -> RecurrentState:
    return self._network.initial_state(batch_size, rngs)

  def _extract_output(self, outputs: Outputs) -> Array:
    assert isinstance(outputs, list)
    output = outputs[-1]
    assert isinstance(output, Array)
    return output

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

# Factory functions for network construction

SIMPLE_CONSTRUCTORS: dict[str, type[Network]] = dict(
    mlp=MLP,
    gru=GRU,
    lstm=LSTM,
    tx_like=TransformerLike,
)

DEFAULT_CONFIG: dict[str, Any] = dict(
    name='mlp',
    **{name: cls.default_config() for name, cls in SIMPLE_CONSTRUCTORS.items()},
)
DEFAULT_CONFIG[FrameTransformer.name] = FrameTransformer.default_config()


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
  input_size = embed_state_action.size

  name = network_config['name']

  if name in SIMPLE_CONSTRUCTORS:
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

  if name == FrameTransformer.name:
    return FrameTransformer(
        rngs=rngs,
        embed_state_action=embed_state_action,
        **network_config[name],
    )

  raise ValueError(f'Unknown network name: {name}')