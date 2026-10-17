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
from slippi_ai.types import (
  Controller, Game, Player, Nana, Item, S,
  Action, StateAction, Frames,
)

Array = jax.Array

Rank = tuple[int, ...]

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

Shape = int | tp.Sequence[int]

class Network(nnx.Module, abc.ABC, tp.Generic[InputTree, OutputTree]):

  def enable_remat(self):
    logging.warning('Remat is not supported for %s', self.__class__.__name__)

  @property
  @abc.abstractmethod
  def output_size(self) -> int:
    '''Returns the output size of the network.'''

  @abc.abstractmethod
  def initial_state(self, batch_size: Shape, rngs: nnx.Rngs) -> RecurrentState:
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
    batch_size = reset.shape
    rngs = nnx.Rngs(0)  # TODO: pass rngs properly
    fresh_state = self.initial_state(batch_size, rngs)
    assert jax_utils.struct_dtype(fresh_state) == jax_utils.struct_dtype(prev_state)
    # Cast fresh_state to match prev_state dtype to avoid fp32 promotion in where
    # fresh_state = jax.tree.map(
    #     lambda f, p: f.astype(p.dtype) if hasattr(p, 'dtype') else f,
    #     fresh_state, prev_state)
    initial_state = where_pytree(reset, fresh_state, prev_state)
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
      input_size: int,
      output_size: int,
      remat: bool = False,
  ):
    self._core = core
    self.input_size = input_size
    self._output_size = output_size
    self._remat = remat

  @property
  def output_size(self) -> int:
    return self._output_size

  def initial_state(self, batch_size: Shape, rngs: nnx.Rngs):
    if isinstance(batch_size, int):
      batch_shape = (batch_size,)
    else:
      batch_shape = tuple(batch_size)
    input_shape = batch_shape + (self.input_size,)
    state = self._core.initialize_carry(input_shape, rngs)
    dtype = jax_utils.module_dtype(self._core)
    return jax_utils.cast_floats_to_dtype(state, dtype)

  def step(self, inputs, prev_state):
    # flax's RNNCells have the arguments reversed
    core = self._core
    if self._remat:
      # core = jax_utils.remat_method(self._core.__call__, prevent_cse=False)
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

  def initial_state(self, batch_size: Shape, rngs: nnx.Rngs):
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

  def initial_state(self, batch_size: Shape, rngs: nnx.Rngs):
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

  def initial_state(self, batch_size: Shape, rngs: nnx.Rngs):
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

  def __init__(
      self,
      layers: tp.Sequence[Network[InputTree, InputTree]],
      remat: bool = False,
  ):
    self._layers = nnx.List(layers)
    self.remat = remat

  def enable_remat(self):
    # TODO: enable remat for all layers? that might be excessive
    self.remat = True

  @property
  def output_size(self) -> int:
    return self._layers[-1].output_size

  def initial_state(self, batch_size: Shape, rngs: nnx.Rngs):
    return [layer.initial_state(batch_size, rngs) for layer in self._layers]

  def step(self, inputs, prev_state):
    next_states: list[RecurrentState] = []
    for layer, state in zip(self._layers, prev_state):
      step_fn = layer.step
      if self.remat:
        step_fn = jax_utils.remat_method(layer.step)
      inputs, next_state = step_fn(inputs, state)
      next_states.append(next_state)
    return inputs, next_states

  def unroll(self, inputs, reset, initial_state):
    final_states: list[RecurrentState] = []
    for layer, state in zip(self._layers, initial_state):
      unroll_fn = layer.unroll
      if self.remat:
        unroll_fn = jax_utils.remat_method(layer.unroll)
      inputs, final_state = unroll_fn(inputs, reset, state)
      final_states.append(final_state)
    return inputs, final_states

  def scan(self, inputs, reset, initial_state):
    hidden_states: list[RecurrentState] = []
    for layer, state in zip(self._layers, initial_state):
      scan_fn = layer.scan
      if self.remat:
        scan_fn = jax_utils.remat_method(layer.scan)
      inputs, hidden_state = scan_fn(inputs, reset, state)
      hidden_states.append(hidden_state)
    return inputs, hidden_states


class ResidualWrapper(Network[InputTree, InputTree]):

  def __init__(self, net: Network[InputTree, InputTree]):
    self._net = net

  @property
  def output_size(self) -> int:
    return self._net.output_size

  def initial_state(self, batch_size: Shape, rngs: nnx.Rngs):
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
            rngs=rngs,
        ),
        input_size=input_size,
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
            rngs=rngs,
        ),
        input_size=input_size,
        output_size=hidden_size,
        remat=remat)


# This deliberately does not use `nnx.LayerNorm`.
#
# Legacy TensorFlow policies used a last-axis layer norm equivalent to:
#
#   centered = x - mean(x)
#   normalized = centered / sqrt(mean(centered ** 2))
#   y = normalized * scale + bias
#
# Flax/NNX LayerNorm includes its own epsilon/stabilization behavior in the
# denominator. That is the default for JAX-native `tx_like` models, but it is
# not algebraically identical to the TensorFlow checkpoint computation.
# Converted recurrent policies are sensitive to this: small activation/logit
# differences can compound over many frames and noticeably change gameplay.
#
# Use this only when checkpoint config explicitly sets `layer_norm =
# "tensorflow"` for legacy checkpoint parity. Converted checkpoints also set
# `gelu_approximate = False` for the same reason: TensorFlow used the exact GELU
# formula, while JAX-native configs default to the faster approximate GELU.
class TensorFlowLayerNorm(nnx.Module):
  """TensorFlow-policy-compatible last-axis layer norm."""

  def __init__(self, residual_size: int, rngs: nnx.Rngs):
    del rngs
    self.scale = nnx.Param(jnp.ones((residual_size,), dtype=jnp.float32))
    self.bias = nnx.Param(jnp.zeros((residual_size,), dtype=jnp.float32))

  def __call__(self, inputs: Array) -> Array:
    centered = inputs - jnp.mean(inputs, axis=-1, keepdims=True)
    stddev = jnp.sqrt(jnp.mean(jnp.square(centered), axis=-1, keepdims=True))
    return centered / stddev * self.scale + self.bias


class ResBlock(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      residual_size: int,
      hidden_size: Optional[int] = None,
      activation: Callable[[Array], Array] = nnx.relu,
      layer_norm: str = 'nnx',
  ):
    if layer_norm == 'nnx':
      self.layernorm = nnx.LayerNorm(residual_size, rngs=rngs)
    elif layer_norm == 'tensorflow':
      self.layernorm = TensorFlowLayerNorm(residual_size, rngs=rngs)
    else:
      raise ValueError(f'Unknown layer norm type: {layer_norm}')
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


def _get_activation(
    activation: str, gelu_approximate: bool = True,
) -> Callable[[Array], Array]:
  return dict(
      relu=nnx.relu,
      gelu=lambda x: nnx.gelu(x, approximate=gelu_approximate),
      tanh=nnx.tanh,
  )[activation]


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
        gelu_approximate=True,
        layer_norm='nnx',
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
      gelu_approximate: bool = True,
      layer_norm: str = 'nnx',
      remat: bool = False,
  ):
    self._hidden_size = hidden_size
    self._num_layers = num_layers
    self._ffw_multiplier = ffw_multiplier

    recurrent_constructor = dict(
        lstm=LSTM,
        gru=GRU,
    )[recurrent_layer]

    activation_fn = _get_activation(activation, gelu_approximate)

    layers: list[Network[Array, Array]] = []

    # We need to encode for the first residual
    encoder = nnx.Linear(input_size, hidden_size, rngs=rngs)
    layers.append(FFWWrapper(encoder, output_size=hidden_size))

    for _ in range(num_layers):
      recurrent = ResidualWrapper(
          recurrent_constructor(rngs, hidden_size, hidden_size, remat=remat))
      layers.append(recurrent)

      ffw_layer = ResBlock(
          rngs, hidden_size, hidden_size * ffw_multiplier,
          activation=activation_fn,
          layer_norm=layer_norm)
      layers.append(FFWWrapper(ffw_layer, output_size=hidden_size))

    # Don't use RematWrapper as it changes the network state structure which
    # prevents loading form checkpoints.

    super().__init__(layers, remat=remat)

class _DepthwiseKernel(nnx.Module):
  """Per-channel weighted sum over taps given as [..., num_taps * size]."""

  def __init__(self, rngs: nnx.Rngs, size: int, num_taps: int):
    self._size = size
    self._num_taps = num_taps
    self.kernel = nnx.Param(
        jax.random.normal(rngs.params(), (num_taps, size)) / math.sqrt(num_taps))
    self.bias = nnx.Param(jnp.zeros((size,)))

  def __call__(self, x: Array) -> Array:
    x = x.reshape(x.shape[:-1] + (self._num_taps, self._size))
    kernel = self.kernel.value.astype(x.dtype)
    return jnp.sum(x * kernel, axis=-2) + self.bias.value.astype(x.dtype)


class DilatedFixedMemoryBlock(Network[Array, Array]):

  def __init__(
      self,
      rngs: nnx.Rngs,
      input_size: int,
      output_size: int,
      dilation: int,
      memory_size: int,
      depthwise: bool = False,
  ):
    self._input_size = input_size
    self._output_size = output_size
    self._dilation = dilation
    self._memory_size = memory_size

    if depthwise:
      if output_size != input_size:
        raise ValueError('Depthwise convolution requires output_size == input_size.')
      self._kernel = _DepthwiseKernel(rngs, input_size, memory_size + 1)
    else:
      self._kernel = nnx.Linear(
          input_size * (memory_size + 1), output_size, rngs=rngs)

  def initial_state(self, batch_size: Shape, rngs: nnx.Rngs):
    del rngs
    if isinstance(batch_size, int):
      batch_shape = (batch_size,)
    else:
      batch_shape = tuple(batch_size)

    dtype = jax_utils.module_dtype(self._kernel)
    node = lambda: jnp.zeros(batch_shape + (self._input_size,), dtype=dtype)

    return [[node() for _ in range(self._memory_size)] for _ in range(self._dilation)]

  def step(self, inputs: Array, prev_state: list[list[Array]]):
    active_nodes = prev_state[0] + [inputs]
    flat_active_nodes = jnp.concatenate(active_nodes, axis=-1)
    outputs = self._kernel(flat_active_nodes)
    next_state = prev_state[1:] + [active_nodes[1:]]
    return outputs, next_state

  @property
  def output_size(self) -> int:
    return self._output_size

  @property
  def _history_len(self) -> int:
    return self._memory_size * self._dilation

  def _state_to_history(self, state: list[list[Array]]) -> Array:
    """Converts the state to the last m*d inputs, oldest first.

    Bucket j, slot i holds x_{t + j - (m - i) * d}, so ordering by time means
    transposing [d, m] -> [m, d] and flattening.
    """
    history = jnp.stack([jnp.stack(memory, axis=0) for memory in state], axis=0)
    history = jnp.swapaxes(history, 0, 1)
    return history.reshape((self._history_len,) + history.shape[2:])

  def _history_to_state(self, history: Array) -> list[list[Array]]:
    history = history.reshape(
        (self._memory_size, self._dilation) + history.shape[1:])
    return [
        [history[i, j] for i in range(self._memory_size)]
        for j in range(self._dilation)
    ]

  def unroll(self, inputs: Array, reset: Array, initial_state: list[list[Array]]):
    if self._memory_size == 0:
      return self._kernel(inputs), initial_state

    m, d = self._memory_size, self._dilation
    T = inputs.shape[0]
    H = self._history_len

    # [H + T, B, D]; x_t is at full[H + t].
    full = jnp.concatenate([self._state_to_history(initial_state), inputs], 0)

    # Steps since the last reset (inclusive); at least H if there was none.
    t = jnp.arange(T)[:, None]
    last_reset = jax.lax.cummax(jnp.where(reset, t, -H - 1), axis=0)
    since_reset = t - last_reset

    # Tap k is x_{t - k * d}, valid iff there was no reset in (t - k * d, t].
    # Taps go oldest first to match the concatenation order in step.
    taps = []
    for k in reversed(range(m + 1)):
      tap = full[H - k * d:H - k * d + T]
      valid = (since_reset >= k * d)[..., None]
      taps.append(jnp.where(valid, tap, 0))
    outputs = self._kernel(jnp.concatenate(taps, axis=-1))

    # The final state is the last H inputs, zeroed if they precede the last reset.
    tail = full[T:]
    tail_t = jnp.arange(T - H, T)[:, None]
    tail = jnp.where((tail_t >= last_reset[-1])[..., None], tail, 0)
    return outputs, self._history_to_state(tail)


def _make_layer_norm(layer_norm: str, size: int, rngs: nnx.Rngs) -> nnx.Module:
  if layer_norm == 'nnx':
    return nnx.LayerNorm(size, rngs=rngs)
  elif layer_norm == 'tensorflow':
    return TensorFlowLayerNorm(size, rngs=rngs)
  raise ValueError(f'Unknown layer norm type: {layer_norm}')


_GATES: dict[str, Callable[[Array, Array], Array]] = dict(
    # WaveNet-style, bounded like an LSTM output.
    tanh_sigmoid=lambda a, b: nnx.tanh(a) * nnx.sigmoid(b),
    # ConvS2S-style gated linear unit.
    glu=lambda a, b: a * nnx.sigmoid(b),
)


class _GatedProjection(nnx.Module):
  """Splits the input in half, gates one half with the other, and projects."""

  def __init__(self, rngs: nnx.Rngs, size: int, gate: str):
    self._gate = _GATES[gate]
    # Zero init so that the residual block starts as identity.
    self.linear = nnx.Linear(
        size, size, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)

  def __call__(self, x: Array) -> Array:
    a, b = jnp.split(x, 2, axis=-1)
    return self.linear(self._gate(a, b))


def gated_conv_block(
    rngs: nnx.Rngs,
    size: int,
    dilation: int,
    memory_size: int,
    gate: str,
    layer_norm: str,
    depthwise: bool = False,
) -> Network[Array, Array]:
  """Pre-norm residual branch: LayerNorm -> dilated conv -> gate -> linear.

  With depthwise=True the conv is per-channel and is followed by a dense
  projection to the gate inputs, as in ConvNeXt/Mamba-style short convs.
  """
  layers: list[Network[Array, Array]] = [
      FFWWrapper(_make_layer_norm(layer_norm, size, rngs), output_size=size),
  ]
  if depthwise:
    layers.append(DilatedFixedMemoryBlock(
        rngs, input_size=size, output_size=size,
        dilation=dilation, memory_size=memory_size, depthwise=True))
    layers.append(Linear(rngs, size, 2 * size))
  else:
    layers.append(DilatedFixedMemoryBlock(
        rngs, input_size=size, output_size=2 * size,
        dilation=dilation, memory_size=memory_size))
  layers.append(FFWWrapper(_GatedProjection(rngs, size, gate), output_size=size))
  return Sequential(layers)


class TemporalConvolution(Sequential, BuildableNetwork[Array, Array]):
  """Like TransformerLike, but with dilated convolutions for memory.

  Layer i is a DilatedFixedMemoryBlock with dilation dilation_base ** i. With
  the default dilation_base of memory_size + 1, the receptive fields of
  successive layers tile without overlap and the whole network sees exactly the
  last (memory_size + 1) ** num_layers frames.

  With block='linear' the conv is applied directly to the residual stream; with
  block='gated' it is a pre-norm gated block whose output projection is zero
  initialized, like the FFW ResBlocks. block='depthwise' is the same but with a
  cheap per-channel conv followed by a dense projection.
  """

  @classmethod
  def name(cls) -> str:
    return 'tcn'

  @classmethod
  def default_config(cls) -> dict[str, tp.Any]:
    return dict(
        hidden_size=128,
        num_layers=1,
        memory_size=2,
        dilation_base=None,
        block='linear',
        gate='tanh_sigmoid',
        ffw_multiplier=4,
        activation='gelu',
        gelu_approximate=True,
        layer_norm='nnx',
    )

  def __init__(
      self,
      rngs: nnx.Rngs,
      input_size: int,
      hidden_size: int,
      num_layers: int,
      memory_size: int,
      ffw_multiplier: int,
      activation: str,
      dilation_base: tp.Optional[int] = None,
      block: str = 'linear',
      gate: str = 'tanh_sigmoid',
      gelu_approximate: bool = True,
      layer_norm: str = 'nnx',
      remat: bool = False,
  ):
    self._hidden_size = hidden_size
    self._num_layers = num_layers
    self._memory_size = memory_size
    self._ffw_multiplier = ffw_multiplier

    if dilation_base is None:
      dilation_base = memory_size + 1

    activation_fn = _get_activation(activation, gelu_approximate)

    layers: list[Network[Array, Array]] = []

    # We need to encode for the first residual
    encoder = nnx.Linear(input_size, hidden_size, rngs=rngs)
    layers.append(FFWWrapper(encoder, output_size=hidden_size))

    dilation = 1
    for _ in range(num_layers):
      if block == 'linear':
        memory = DilatedFixedMemoryBlock(
            rngs,
            input_size=hidden_size,
            output_size=hidden_size,
            dilation=dilation,
            memory_size=memory_size,
        )
      elif block in ('gated', 'depthwise'):
        memory = gated_conv_block(
            rngs, hidden_size, dilation, memory_size,
            gate=gate, layer_norm=layer_norm,
            depthwise=block == 'depthwise')
      else:
        raise ValueError(f'Unknown block type: {block}')
      layers.append(ResidualWrapper(memory))
      dilation *= dilation_base

      ffw_layer = ResBlock(
          rngs, hidden_size, hidden_size * ffw_multiplier,
          activation=activation_fn,
          layer_norm=layer_norm)
      layers.append(FFWWrapper(ffw_layer, output_size=hidden_size))

    super().__init__(layers, remat=remat)


def _ssd_chunk(
    x: Array,  # [Q, B, H, P]
    dt: Array,  # [Q, B, H], fp32, after softplus
    A: Array,  # [H], fp32
    B: Array,  # [Q, B, N]
    C: Array,  # [Q, B, N]
    initial_state: Array,  # [B, H, P, N], fp32
) -> tuple[Array, Array]:
  """SSD over a single chunk, time-major, without the D skip connection.

  The decays are computed in fp32; the large [Q, B, H, P] and [B, H, Q, Q]
  operands are stored in x.dtype and contracted with fp32 accumulation.
  Returns fp32 outputs and final state.
  """
  dtype = x.dtype
  Q = x.shape[0]
  f32 = jnp.float32

  log_decay = jnp.cumsum(A * dt, axis=0)  # [Q, B, H]

  # 1. Within the chunk: y_l = sum_{s <= l} C_l.B_s exp(sum_{s<k<=l} a_k) x_s.
  ld = jnp.transpose(log_decay, (1, 2, 0))  # [B, H, Q]
  segsum = ld[..., :, None] - ld[..., None, :]  # [B, H, l, s]
  causal = jnp.tril(jnp.ones((Q, Q), dtype=bool))
  L = jnp.exp(jnp.where(causal, segsum, -jnp.inf))
  CB = jnp.einsum('lbn,sbn->bls', C, B, preferred_element_type=f32)
  M = (CB[:, None] * L).astype(dtype)  # [B, H, l, s]
  xdt = (x * dt[..., None]).astype(dtype)  # [Q, B, H, P]
  y = jnp.einsum('bhls,sbhp->lbhp', M, xdt, preferred_element_type=f32)

  # 2. The incoming state's contribution to each output.
  y_in = jnp.einsum(
      'lbn,bhpn->lbhp', C, initial_state.astype(dtype),
      preferred_element_type=f32)
  y = y + y_in * jnp.exp(log_decay)[..., None]

  # 3. The state at the end of the chunk.
  decay_to_end = jnp.exp(log_decay[-1:] - log_decay)  # [Q, B, H]
  x_to_end = (xdt * decay_to_end[..., None]).astype(dtype)
  final_state = jnp.einsum(
      'sbn,sbhp->bhpn', B, x_to_end, preferred_element_type=f32)
  final_state += jnp.exp(log_decay[-1])[..., None, None] * initial_state
  return y, final_state


def _ssd(
    x: Array,  # [T, B, H, P]
    dt: Array,  # [T, B, H], fp32, after softplus
    A: Array,  # [H], fp32
    B: Array,  # [T, B, N]
    C: Array,  # [T, B, N]
    initial_state: Array,  # [B, H, P, N], fp32
    chunk_size: int,
) -> tuple[Array, Array]:
  """Chunked SSD scan, time-major, without the D skip connection.

  Same algorithm as mamba2_jax.ssd_forward, but B and C are shared across
  heads (a single group) instead of being broadcast to every head, and chunks
  are processed sequentially. With T <= chunk_size it is a single chunk.
  """
  T = x.shape[0]
  ssd_chunk = jax.checkpoint(_ssd_chunk)
  if T <= chunk_size:
    return ssd_chunk(x, dt, A, B, C, initial_state)

  pad = (chunk_size - T % chunk_size) % chunk_size
  # dt = 0 means padded steps neither decay nor update the state.
  pad_time = lambda a: jnp.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1))
  num_chunks = (T + pad) // chunk_size
  chunk = lambda a: pad_time(a).reshape((num_chunks, chunk_size) + a.shape[1:])

  def step(state, inputs):
    x, dt, B, C = inputs
    y, state = ssd_chunk(x, dt, A, B, C, state)
    return state, y

  final_state, y = jax.lax.scan(
      step, initial_state, tuple(map(chunk, (x, dt, B, C))))
  y = y.reshape((num_chunks * chunk_size,) + y.shape[2:])[:T]
  return y, final_state


class Mamba2Layer(Network[Array, Array]):
  """RMSNorm followed by a Mamba2 mixer (no residual).

  The parameters and projections are those of mamba2_jax.Mamba2Mixer; the SSD
  scan is our own (see _ssd) as the library's runs out of memory at our batch
  sizes, and step is a direct recurrence instead of a padded chunk.

  Resets are only honored at the start of an unroll; a reset in the middle of
  an unroll does not clear the SSM or conv state. Step honors all resets.

  Everything is time-major. Activations are stored in the param dtype, but the
  conv, gated norm and SSD accumulations are computed in fp32, as are the SSD's
  decays and cumulative sums, which are too imprecise in bf16. The recurrent
  state is stored in the param dtype.
  """

  # Overrides the dtype that activations are stored in, for testing.
  _act_dtype: tp.Optional[jnp.dtype] = None

  def __init__(self, rngs: nnx.Rngs, cfg):
    import mamba2_jax
    self._cfg = cfg
    self.norm = mamba2_jax.RMSNorm(
        cfg.hidden_size, eps=cfg.layer_norm_epsilon, rngs=rngs)
    self.mixer = mamba2_jax.Mamba2Mixer(cfg, layer_idx=0, rngs=rngs)

  @property
  def output_size(self) -> int:
    return self._cfg.hidden_size

  def initial_state(self, batch_size: Shape, rngs: nnx.Rngs):
    del rngs
    if isinstance(batch_size, int):
      batch_shape = (batch_size,)
    else:
      batch_shape = tuple(batch_size)
    cfg = self._cfg
    dtype = jax_utils.module_dtype(self)
    conv_dim = cfg.intermediate_size + 2 * cfg.state_size
    conv_state = jnp.zeros(
        batch_shape + (conv_dim, cfg.conv_kernel - 1), dtype=dtype)
    ssm_state = jnp.zeros(
        batch_shape + (cfg.num_heads, cfg.head_dim, cfg.state_size),
        dtype=dtype)
    return conv_state, ssm_state

  def _mix(self, inputs: Array, prev_state, ssm_fn):
    """Runs the mixer on time-major [T, B, D] inputs.

    ssm_fn(x, dt, A, B, C, ssm_state) -> (y, ssm_state), without the D skip,
    with fp32 dt, A and ssm_state, returning fp32 y and ssm_state.
    """
    dtype = inputs.dtype
    act_dtype = self._act_dtype or dtype
    conv_state, ssm_state = prev_state
    mixer = self.mixer
    cfg = self._cfg
    num_heads, head_dim = cfg.num_heads, cfg.head_dim
    inner, state_size = cfg.intermediate_size, cfg.state_size
    f32 = lambda x: x.astype(jnp.float32)

    zxbcdt = mixer.in_proj(self.norm(inputs))
    d_mlp = (zxbcdt.shape[-1] - 2 * inner - 2 * state_size - num_heads) // 2
    z0, x0, z, xBC, dt = jnp.split(
        zxbcdt,
        [d_mlp, 2 * d_mlp, 2 * d_mlp + inner,
         2 * d_mlp + 2 * inner + 2 * state_size],
        axis=-1)

    # Causal depthwise conv over time, computed in fp32.
    T = xBC.shape[0]
    conv = mixer.conv1d.conv
    kernel = f32(conv.kernel.value[:, 0, :])  # [K, C]
    history = jnp.transpose(conv_state, (2, 0, 1)).astype(xBC.dtype)
    padded = jnp.concatenate([history, xBC], axis=0)  # [K - 1 + T, B, C]
    xBC = sum(kernel[j] * f32(padded[j:j + T]) for j in range(cfg.conv_kernel))
    if conv.bias is not None:
      xBC = xBC + f32(conv.bias.value)
    xBC = mixer.act(xBC).astype(act_dtype)
    conv_state = jnp.transpose(padded[T:], (1, 2, 0))

    x, B, C = jnp.split(xBC, [inner, inner + state_size], axis=-1)
    x = x.reshape(x.shape[:-1] + (num_heads, head_dim))

    dt = jax.nn.softplus(f32(dt) + f32(mixer.dt_bias.value))
    dt = jnp.clip(dt, mixer.dt_min, mixer.dt_max)
    A = -jnp.exp(f32(mixer.A_log.value))
    y, ssm_state = ssm_fn(x, dt, A, B, C, f32(ssm_state))
    y = y + f32(mixer.D.value)[:, None] * f32(x)

    # Gated RMSNorm, as mixer.norm, computed in fp32.
    y = y.reshape(y.shape[:-2] + (inner,)) * jax.nn.silu(f32(z))
    variance = jnp.mean(jnp.square(y), axis=-1, keepdims=True)
    y = y * jax.lax.rsqrt(variance + mixer.norm.eps) * f32(mixer.norm.weight.value)
    y = y.astype(dtype)
    if d_mlp > 0:
      y = jnp.concatenate([mixer.act(z0) * x0, y], axis=-1)
    outputs = mixer.out_proj(y).astype(dtype)
    next_state = (conv_state.astype(dtype), ssm_state.astype(dtype))
    return outputs, next_state

  def unroll(self, inputs: Array, reset: Array, initial_state):
    fresh_state = self.initial_state(reset.shape[1:], nnx.Rngs(0))
    initial_state = where_pytree(reset[0], fresh_state, initial_state)

    def ssm_fn(x, dt, A, B, C, ssm_state):
      return _ssd(x, dt, A, B, C, ssm_state, self._cfg.chunk_size)

    return self._mix(inputs, initial_state, ssm_fn)

  def step(self, inputs: Array, prev_state):
    # Flatten the batch dims and add a time dim.
    batch_shape = inputs.shape[:-1]
    flatten = lambda x, n: x.reshape((-1,) + x.shape[x.ndim - n:])
    inputs = flatten(inputs, 1)[None]
    prev_state = jax.tree.map(flatten, prev_state, (2, 3))

    def ssm_fn(x, dt, A, B, C, ssm_state):
      f32 = lambda a: a[0].astype(jnp.float32)
      x, dt, B, C = f32(x), dt[0], f32(B), f32(C)
      decay = jnp.exp(A * dt)[..., None, None]
      update = (dt[..., None] * x)[..., None] * B[:, None, None, :]
      ssm_state = decay * ssm_state + update  # [B, H, P, N]
      y = jnp.einsum('bhpn,bn->bhp', ssm_state, C)
      return y[None], ssm_state

    outputs, next_state = self._mix(inputs, prev_state, ssm_fn)
    unflatten = lambda x: x.reshape(batch_shape + x.shape[1:])
    return unflatten(outputs[0]), jax.tree.map(unflatten, next_state)


class Mamba2(Sequential, BuildableNetwork[Array, Array]):
  """Like TransformerLike, but with Mamba2 mixers for memory.

  Uses the Mamba2Mixer from https://github.com/jax-state-spaces/mamba2-jax.
  Each layer is a pre-norm residual Mamba2 mixer, optionally followed by a FFW
  ResBlock (disabled with ffw_multiplier=0, as in the original Mamba2).
  """

  @classmethod
  def name(cls) -> str:
    return 'mamba2'

  @classmethod
  def default_config(cls) -> dict[str, tp.Any]:
    return dict(
        hidden_size=128,
        num_layers=1,
        state_size=128,
        head_dim=64,
        expand=2,
        conv_kernel=4,
        chunk_size=128,
        ffw_multiplier=4,
        activation='gelu',
        gelu_approximate=True,
        layer_norm='nnx',
        remat=False,
    )

  def __init__(
      self,
      rngs: nnx.Rngs,
      input_size: int,
      hidden_size: int,
      num_layers: int,
      state_size: int,
      head_dim: int,
      expand: int,
      conv_kernel: int,
      chunk_size: int,
      ffw_multiplier: int,
      activation: str,
      gelu_approximate: bool = True,
      layer_norm: str = 'nnx',
      remat: bool = False,
  ):
    import mamba2_jax
    cfg = mamba2_jax.Mamba2Config(
        hidden_size=hidden_size,
        state_size=state_size,
        head_dim=head_dim,
        expand=expand,
        conv_kernel=conv_kernel,
        chunk_size=chunk_size,
        num_hidden_layers=num_layers,
    )
    activation_fn = _get_activation(activation, gelu_approximate)

    layers: list[Network[Array, Array]] = []

    encoder = nnx.Linear(input_size, hidden_size, rngs=rngs)
    layers.append(FFWWrapper(encoder, output_size=hidden_size))

    for _ in range(num_layers):
      layers.append(ResidualWrapper(Mamba2Layer(rngs, cfg)))

      if ffw_multiplier > 0:
        ffw_layer = ResBlock(
            rngs, hidden_size, hidden_size * ffw_multiplier,
            activation=activation_fn,
            layer_norm=layer_norm)
        layers.append(FFWWrapper(ffw_layer, output_size=hidden_size))

    super().__init__(layers, remat=remat)


class StateActionNetwork(nnx.Module, abc.ABC, tp.Generic[Action]):
  """Like a network, but takes StateAction as input."""

  @property
  @abc.abstractmethod
  def output_size(self) -> int:
    """Returns the output size of the network."""

  def enable_remat(self):
    logging.warning('Remat is not supported for %s', self.__class__.__name__)

  @abc.abstractmethod
  def dummy(self, shape: S) -> StateAction[S, Action]:
    """Returns a dummy input of the given shape."""

  @abc.abstractmethod
  def encode(self, state_action: StateAction[S, Controller]) -> StateAction[S, Action]:
    """Encodes the state and action into a form suitable for the network."""

  @abc.abstractmethod
  def encode_game(self, game: Game[S]) -> Game[S]:
    """Like encode but only the gamestate.

    Useful for agents who already have encoded actions.
    Assumes that the controller_head's encoding matches that of the network.
    """

  @abc.abstractmethod
  def initial_state(self, batch_size: Shape, rngs: nnx.Rngs) -> RecurrentState:
    pass

  @abc.abstractmethod
  def step(
      self,
      state_action: StateAction[S, Action],
      prev_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    pass

  def step_with_reset(
      self,
      state_action: StateAction[S, Action],
      reset: Array,
      prev_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    batch_size = reset.shape
    rngs = nnx.Rngs(0)  # TODO: pass rngs properly
    initial_state = where_pytree(
        reset, self.initial_state(batch_size, rngs), prev_state)
    return self.step(state_action, initial_state)

  def _step_with_reset(
      self,
      inputs: tuple[StateAction[S, Action], Array],
      prev_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    """Used for unroll/scan."""
    state_action, reset = inputs
    return self.step_with_reset(state_action, reset, prev_state)

  def unroll(
      self,
      state_action: StateAction[S, Action],
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    return jax_utils.dynamic_rnn(
        self._step_with_reset, (state_action, reset), initial_state)

  def scan(
      self,
      state_action: StateAction[S, Action],
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    return jax_utils.scan_rnn(
        self._step_with_reset, (state_action, reset), initial_state)

class EmbedModule(abc.ABC, tp.Generic[Action]):

  @property
  @abc.abstractmethod
  def output_size(self) -> int:
    """Returns the output size of the network."""

  @abc.abstractmethod
  def dummy(self, shape: S) -> StateAction[S, Action]:
    """Returns a dummy input of the given shape."""

  @abc.abstractmethod
  def encode(self, state_action: StateAction[S, Controller]) -> StateAction[S, Action]:
    """Encodes the state and action into a form suitable for the network."""

  @abc.abstractmethod
  def encode_game(self, game: Game[S]) -> Game[S]:
    """Like encode but only the gamestate.

    Useful for agents who already have encoded actions.
    Assumes that the controller_head's encoding matches that of the network.
    """

  @abc.abstractmethod
  def __call__(self, state_action: StateAction[S, Action], remat: bool = False) -> Array:
    """Embeds the state and action into an Array suitable for the network."""

class Embeddings(tp.NamedTuple, tp.Generic[Action]):
  config: embed_lib.EmbedConfig[Action]
  num_names: int
  frame_skip: int
  embed_game: embed_lib.Embedding[Game, Game]
  embed_action: embed_lib.Embedding[Controller, Action]
  embed_state_action: embed_lib.StateActionEmbedding[Action]

  @classmethod
  def init(
    cls,
    config: embed_lib.EmbedConfig[Action],
    frame_skip: int,
    num_names: int,
    embed_action: tp.Optional[embed_lib.Embedding[Controller, Action]] = None,
  ) -> tp.Self:
    embed_game = config.make_game_embedding()
    embed_action = config.make_controller_embedding() if embed_action is None else embed_action
    embed_state_action = embed_lib.get_state_action_embedding(
        embed_game=embed_game,
        embed_action=embed_action,
        frame_skip=frame_skip,
        num_names=num_names,
        with_rating=config.with_rating,
    )
    return cls(
        config=config,
        num_names=num_names,
        frame_skip=frame_skip,
        embed_game=embed_game,
        embed_action=embed_action,
        embed_state_action=embed_state_action,
    )

class SimpleEmbedModule(EmbedModule[Action]):

  def __init__(
      self,
      embeddings: Embeddings[Action],
  ):
    self._embed_game = embeddings.embed_game
    self._embed_state_action = embeddings.embed_state_action

  @property
  def output_size(self) -> int:
    return self._embed_state_action.size

  def dummy(self, shape: S) -> StateAction[S, Action]:
    return self._embed_state_action.dummy(shape)

  def encode(self, state_action: StateAction[S, Controller]) -> StateAction[S, Action]:
    return self._embed_state_action.from_state(state_action)

  def encode_game(self, game: Game[S]) -> Game[S]:
    return self._embed_game.from_state(game)

  def __call__(self, state_action: StateAction[S, Action], remat: bool = False) -> Array:
    del remat  # unused
    return self._embed_state_action(state_action)

class SimpleEmbedNetwork(StateActionNetwork[Action]):
  """Embeds the state and action using provided embedding module."""

  def __init__(
      self,
      embed_module: EmbedModule[Action],
      network: Network[Array, Array],
      remat: bool = False,
  ):
    self._embed_module = embed_module
    self._network = network
    self._remat = remat

  @property
  def output_size(self) -> int:
    return self._network.output_size

  def enable_remat(self):
    self._remat = True
    self._network.enable_remat()

  def dummy(self, shape: S) -> StateAction[S, Action]:
    return self._embed_module.dummy(shape)

  def initial_state(self, batch_size: Shape, rngs: nnx.Rngs) -> RecurrentState:
    return self._check_state_dtype(self._network.initial_state(batch_size, rngs))

  def encode(self, state_action: StateAction[S, Controller]) -> StateAction[S, Action]:
    return self._embed_module.encode(state_action)

  def encode_game(self, game: Game[S]) -> Game[S]:
    return self._embed_module.encode_game(game)

  def _embed(self, state_action: StateAction[S, Action]) -> Array:
    # if self._remat:
    #   if isinstance(self._embed_module, nnx.Module):
    #     x = nnx.remat(apply)(self._embed_module, state_action, remat=True)
    #   else:
    #     x = nnx.remat(self._embed_module)(state_action)
    # else:
    x = self._embed_module(state_action, remat=self._remat)
    x = x.astype(jax_utils.module_dtype(self))
    return x

  def _check_state_dtype(self, state: RecurrentState) -> RecurrentState:
    # Initial state might come from an fp32 version of the network,
    # so we need to cast it to make sure it's in bf16.
    # return jax_utils.cast_floats_to_dtype(
    #     state, jax_utils.module_dtype(self._network))
    dtype = jax_utils.module_dtype(self)
    state_dtype = jax_utils.struct_dtype(state)
    assert state_dtype in (dtype, None), f"State dtype {state_dtype} does not match module dtype {dtype}"
    return state

  def step(
      self,
      state_action: StateAction[S, Action],
      prev_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    embedded = self._embed(state_action)
    output, next_state = self._network.step(embedded, self._check_state_dtype(prev_state))
    assert isinstance(output, Array)
    return output, next_state

  def unroll(
      self,
      state_action: StateAction[S, Action],
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    embedded = self._embed(state_action)
    output, final_state = self._network.unroll(embedded, reset, self._check_state_dtype(initial_state))
    assert isinstance(output, Array)
    return output, final_state

  def scan(
      self,
      state_action: StateAction[S, Action],
      reset: Array,
      initial_state: RecurrentState,
  ) -> Tuple[Array, RecurrentState]:
    embedded = self._embed(state_action)
    output, final_state = self._network.scan(embedded, reset, self._check_state_dtype(initial_state))
    assert isinstance(output, Array)
    return output, final_state

# TODO: unify with controller_heads
class ControllerRNN(nnx.Module, tp.Generic[Action]):
  """Embed controller using an RNN over its components."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      embed_controller: embed_lib.Embedding[Controller, Action],
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

  def initial_state(self, batch_size: Shape, rngs: nnx.Rngs):
    if isinstance(batch_size, int):
      batch_shape = (batch_size,)
    else:
      batch_shape = tuple(batch_size)
    input_shape = batch_shape + (self._embed_flat[0].size,)
    state = self._initial_state_fn(input_shape, rngs)
    dtype = jax_utils.module_dtype(self)
    return jax_utils.cast_floats_to_dtype(state, dtype)

  def __call__(
      self,
      controller: Action,
      prev_state: RecurrentState,
  ) -> tuple[Array, RecurrentState]:
    dtype = jax_utils.module_dtype(self)

    hidden_state = prev_state

    for cell, embed, component in zip(
        self._cells, self._embed_flat,
        self._embed_controller.flatten(controller)):

      hidden_state, output = cell(hidden_state, embed(component).astype(dtype))

    return output, hidden_state  # type: ignore

  def unroll(
      self,
      controllers: list[Action],
      remat: bool = False,
  ) -> Array:
    batch_shape = next(self._embed_controller.flatten(controllers[0])).shape
    initial_state = self.initial_state(batch_shape, nnx.Rngs(0))

    inputs = self._embed_controller.map(
        lambda _, *xs: jnp.stack(xs), *controllers)  # type: ignore

    cell_fn = ControllerRNN.__call__
    if remat:
      cell_fn = nnx.remat(cell_fn)

    outputs, _ = jax_utils.dynamic_rnn_module(cell_fn, self, inputs, initial_state)
    return outputs[-1]


P = tp.ParamSpec('P')
T = tp.TypeVar('T')

def apply(f: tp.Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
  return f(*args, **kwargs)

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

  def initial_state(self, batch_size: Shape, rngs: nnx.Rngs) -> RecurrentState:

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

class MultiEmbed(nnx.Module):

  def __init__(
      self,
      sizes: tuple[int, ...],
      features: int,
      rngs: nnx.Rngs,
      embedding_init: tp.Optional[nnx.Initializer] = None,
  ):
    self._sizes = sizes
    total_size = math.prod(sizes)
    kwargs = {}
    if embedding_init is not None:
      kwargs['embedding_init'] = embedding_init
    self._embed = nnx.Embed(
        num_embeddings=total_size,
        features=features,
        rngs=rngs,
        **kwargs,
    )

  def __call__(self, *inputs: Array) -> Array:
    assert len(inputs) == len(self._sizes)

    index = jnp.zeros_like(inputs[0])
    # Not jnp.full(index.shape, ...), which jax2onnx can't export with a
    # dynamic batch size.
    valid = jnp.ones_like(index, dtype=bool)
    for size, inp in zip(self._sizes, inputs):
      # Compare in int32, since onnxruntime's CUDA comparisons lack kernels
      # for small unsigned types.
      inp_int32 = inp.astype(jnp.int32)
      valid &= (inp_int32 >= 0) & (inp_int32 < size)
      index *= size
      index += inp

    embed = jnp.where(valid[..., None], self._embed(index), 0)
    return embed

PlayerOrNana = tp.TypeVar('PlayerOrNana', Player, Nana)

class EnhancedEmbedModule(nnx.Module, EmbedModule[Action]):

  @classmethod
  def default_config(cls) -> dict[str, tp.Any]:
    return dict(
        hidden_size=128,
        item_mlp_layers=2,
        rnn_cell='lstm',
        use_self_nana=True,
        use_controller_rnn=False,
        share_crnn=True,
        use_learned_char=True,
        use_learned_action=True,
        use_char_action_joint=True,
        use_item_sum=True,
        use_items=True,
        hybrid_embed=False,
    )

  def __init__(
      self,
      rngs: nnx.Rngs,
      embeddings: Embeddings[Action],
      hidden_size: int,
      item_mlp_layers: int,
      rnn_cell: str = 'lstm',
      use_self_nana: bool = True,
      use_controller_rnn: bool = False,
      share_crnn: bool = True,
      use_learned_char: bool = True,
      use_learned_action: bool = True,
      use_char_action_joint: bool = True,
      use_item_sum: bool = True,
      use_items: bool = True,
      hybrid_embed: bool = False,
  ):
    self._use_self_nana = use_self_nana
    self._use_learned_char = use_learned_char
    self._hybrid_embed = hybrid_embed
    self._use_learned_action = use_learned_action
    self._use_char_action_joint = use_char_action_joint
    self._use_item_sum = use_item_sum
    self._use_items = use_items

    self._embed_game = embeddings.embed_game
    self._embed_controller = embeddings.embed_action
    self._embed_state_action = embeddings.embed_state_action

    # Embed all items using the same Linear layer
    self._item_embedding = embeddings.config.make_item_embedding()
    self._item_mlp = jax_utils.MLP(
        rngs=rngs,
        input_size=self._item_embedding.size,
        features=[hidden_size] * item_mlp_layers,
    )

    self._use_controller_rnn = use_controller_rnn
    if use_controller_rnn:
      if not share_crnn:
        raise NotImplementedError('Separate CRNNs not supported anymore.')

      controller_rnn = ControllerRNN(
          rngs=rngs,
          embed_controller=self._embed_controller,
          hidden_size=hidden_size,
          rnn_cell=rnn_cell,
      )

      # Wrap in List for backwards compatibility with the old non-share_crnn path.
      self._controller_rnns = nnx.List([controller_rnn])

    # Assumes that p0 and p1 have the same embedding structure
    embed_char = embed_lib.embed_char
    embed_char.size

    self._embed_char = nnx.Embed(
        num_embeddings=embed_char.size,
        features=hidden_size,
        rngs=rngs,
    )

    # This refers to the character's action state in-game, not the controller.
    self._embed_action = nnx.Embed(
        num_embeddings=embed_lib.embed_action.size,
        features=hidden_size,
        rngs=rngs,
    )

    self._embed_char_action = MultiEmbed(
        sizes=(embed_char.size, embed_lib.embed_action.size),
        features=hidden_size,
        rngs=rngs,
        embedding_init=nnx.initializers.zeros,
    )

    output_shape = jax_utils.eval_shape_method(self.__call__, self.dummy(()))
    assert output_shape.ndim == 1
    self._output_size = output_shape.shape[0]

  @property
  def output_size(self) -> int:
    return self._output_size

  def dummy(self, shape: S) -> StateAction[S, Action]:
    return self._embed_state_action.dummy(shape)

  def encode(self, state_action: StateAction[S, Controller]):
    return self._embed_state_action.from_state(state_action)

  def encode_game(self, game: Game[S]) -> Game[S]:
    return self._embed_game.from_state(game)

  def _embed_player_or_nana(self, raw: PlayerOrNana, default: PlayerOrNana) -> Array:
    dtype = jax_utils.module_dtype(self)

    if self._use_learned_action:
      action = self._embed_action(raw.action).astype(dtype)
      if self._use_char_action_joint:
        action = action + self._embed_char_action(raw.character, raw.action)
      if self._hybrid_embed:
        action = jnp.concatenate([action, default.action], axis=-1)
    else:
      action = default.action

    if self._use_learned_char:
      char = self._embed_char(raw.character).astype(dtype)
      if self._hybrid_embed:
        char = jnp.concatenate([char, default.character], axis=-1)
    else:
      char = default.character

    parts = [
        default.percent,
        default.facing,
        default.x, default.y,
        action,
        default.invulnerable,
        char,
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

  def __call__(self, state_action: StateAction[tp.Any, Action], remat: bool = False) -> Array:
    dtype = jax_utils.module_dtype(self)

    raw_game = state_action.state
    default_state_action_embed = self._embed_state_action.map(
        lambda e, v: e(v).astype(dtype), state_action)
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

    if self._use_items:
      # Process items as a batch
      stacked_items: Item = jax.tree.map(
        lambda *args: jnp.stack(args, axis=-1), *raw_game.items)
      item_embed = self._item_embedding(stacked_items).astype(dtype)
      assert item_embed.shape[-2:] == (len(raw_game.items), self._item_embedding.size)
      if self._use_item_sum:
        item_embed = self._item_mlp(item_embed)
        item_embed = jnp.where(stacked_items.exists[..., None], item_embed, 0)
        items_embed = jnp.sum(item_embed, axis=-2)
      else:
        # Flatten items to concatenate them
        items_embed = item_embed.reshape(*item_embed.shape[:-2], -1)
      parts.append(items_embed)

    if isinstance(default_state_action_embed.name, jax.Array):
      parts.append(default_state_action_embed.name)

    if isinstance(default_state_action_embed.rating, jax.Array):
      parts.append(default_state_action_embed.rating)

    if self._use_controller_rnn:
      parts.append(self._controller_rnns[0].unroll(state_action.action, remat=remat))
    else:
      parts.extend(map(self._embed_controller, state_action.action))

    return jnp.concatenate(parts, axis=-1)

# Factory functions for network construction

_simple_networks: list[type[BuildableNetwork[Array, Array]]] = [
    MLP, LSTM, GRU, TransformerLike, TemporalConvolution, Mamba2,
]
SIMPLE_CONSTRUCTORS = {
    cls.name(): cls for cls in _simple_networks
}

DEFAULT_CONFIG: dict[str, Any] = dict(
    name='mlp',
    **{name: cls.default_config() for name, cls in SIMPLE_CONSTRUCTORS.items()},
)

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
) -> Network[Array, Array]:
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
  return constructor.from_config(rngs=rngs, input_size=input_size, **config[name])

def build_embed_module(
    rngs: nnx.Rngs,
    config: dict,
    embed_config: embed_lib.EmbedConfig[Action],
    frame_skip: int,
    num_names: int,
    embed_action: tp.Optional[embed_lib.Embedding[Controller, Action]] = None,
):
  embeddings: Embeddings[Action] = Embeddings.init(
      config=embed_config,
      frame_skip=frame_skip,
      num_names=num_names,
      embed_action=embed_action,
  )

  name = config['name']
  logging.info('Using embed module: %s', name)

  if name == 'simple':
    return SimpleEmbedModule(embeddings)

  if name == 'enhanced':
    return EnhancedEmbedModule(
        rngs=rngs,
        embeddings=embeddings,
        **config['enhanced'],
    )

  raise ValueError(f'Unknown embed module name: {name}')

def build_embed_network(
    rngs: nnx.Rngs,
    embed_config: embed_lib.EmbedConfig[Action],
    num_names: int,
    network_config: dict,
    frame_skip: int,
    embed_action: tp.Optional[embed_lib.Embedding[Controller, Action]] = None,
    remat: bool = False,
) -> StateActionNetwork[Action]:
  """Build a SimpleEmbedNetwork from config.

  Args:
    rngs: Random number generators for initialization.
    embed_config: Configuration for embeddings.
    num_names: Number of player names for the name embedding.
    network_config: Network configuration dict.
    frame_skip: Number of frames to skip for the action embedding.
  Returns:
    Constructed SimpleEmbedNetwork.
  """
  name = network_config['name']
  assert name in SIMPLE_CONSTRUCTORS, f'Unknown network name: {name}'

  embed_module = build_embed_module(
      rngs=rngs,
      config=network_config['embed'],
      embed_config=embed_config,
      frame_skip=frame_skip,
      num_names=num_names,
      embed_action=embed_action,
  )

  network = construct_network(
      rngs=rngs,
      input_size=embed_module.output_size,
      **network_config,
  )

  return SimpleEmbedNetwork(
      embed_module=embed_module,
      network=network,
      remat=remat,
  )
