import abc
import copy
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from slippi_ai.jax import embed, jax_utils
from slippi_ai.types import Controller

Array = jax.Array
ControllerType = tp.TypeVar('ControllerType')

class SampleOutputs(tp.NamedTuple):
  controller_state: ControllerType
  logits: ControllerType

class DistanceOutputs(tp.NamedTuple):
  distance: ControllerType
  logits: ControllerType

class ControllerHead(nnx.Module, abc.ABC, tp.Generic[ControllerType]):

  @abc.abstractmethod
  def sample(
      self,
      rngs: nnx.Rngs | nnx.RngStream,
      inputs: Array,
      prev_controller_state: ControllerType,
      temperature: tp.Optional[float] = None,
  ) -> SampleOutputs:
    """Sample a controller state given input features and previous state."""

  @abc.abstractmethod
  def distance(
      self,
      inputs: Array,
      prev_controller_state: ControllerType,
      target_controller_state: ControllerType,
  ) -> DistanceOutputs:
    """A struct of distances (generally, negative log probs)."""

  @abc.abstractmethod
  def controller_embedding(self) -> embed.Embedding[Controller, ControllerType]:
    """Determines how controllers are embedded (e.g. discretized)."""

  @classmethod
  @abc.abstractmethod
  def default_config(cls) -> dict[str, tp.Any]:
    """Returns the default config for this ControllerHead."""


class Independent(ControllerHead[ControllerType]):
  """Models each component of the controller independently."""

  @classmethod
  def default_config(cls):
    return {}

  def __init__(
      self,
      rngs: nnx.Rngs,
      input_size: int,
      embed_controller: embed.Embedding[Controller, ControllerType],
  ):
    self.embed_controller = embed_controller
    self.to_controller_input = nnx.Linear(
        input_size, self.embed_controller.size, rngs=rngs)

  def controller_embedding(self) -> embed.Embedding[Controller, ControllerType]:
    return self.embed_controller

  def controller_prediction(self, inputs, prev_controller_state) -> ControllerType:
    controller_prediction = self.to_controller_input(inputs)

    # TODO: come up with a better way to do this that generalizes nicely across
    # Independent and AutoRegressive ControllerHeads.
    embed_struct = self.embed_controller.map(lambda e: e)
    embed_sizes = [e.size for e in self.embed_controller.flatten(embed_struct)]
    # Use numpy for split indices to avoid tracing issues with JIT
    split_indices = np.cumsum(embed_sizes[:-1]).tolist()
    leaf_logits = jnp.split(controller_prediction, split_indices, axis=-1)
    return self.embed_controller.unflatten(iter(leaf_logits))

  def sample(self, rngs, inputs, prev_controller_state, temperature=None):
    logits = self.controller_prediction(inputs, prev_controller_state)

    sample = self.embed_controller.map(
        lambda e, l: e.sample(rngs, l, temperature=temperature), logits)
    return SampleOutputs(controller_state=sample, logits=logits)

  def distance(self, inputs, prev_controller_state, target_controller_state):
    logits = self.controller_prediction(inputs, prev_controller_state)
    distance = self.embed_controller.map(
        lambda e, l, t: e.distance(l, t), logits, target_controller_state)
    return DistanceOutputs(distance=distance, logits=logits)


class AutoRegressiveComponent(nnx.Module):
  """Autoregressive residual component."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      embedder: embed.Embedding,
      residual_size: int,
      depth: int = 0,
  ):
    self.embedder = embedder

    # Build encoder MLP
    self._encoder = jax_utils.MLP(
        rngs,
        input_size=residual_size + embedder.size,
        features=[residual_size] * depth + [embedder.distribution_size()],
        activation=nnx.relu,
        activation_final=False,
    )

    # The decoder doesn't need depth, because a single Linear decoding a one-hot
    # has full expressive power over the output
    self.decoder = nnx.Linear(
        embedder.size, residual_size,
        kernel_init=nnx.initializers.zeros_init(),
        rngs=rngs)

  def sample(
      self,
      rng: jax.Array,
      residual: Array,
      prev_raw: Array,
      **kwargs,
  ) -> tp.Tuple[Array, SampleOutputs]:
    # Directly connect from the same component at time t-1
    prev_embedding = self.embedder(prev_raw)
    input_ = jnp.concatenate([residual, prev_embedding], axis=-1)
    # Project down to the size desired by the component
    logits = self._encoder(input_)
    # Sample the component
    sample = self.embedder.sample(rng, logits, **kwargs)
    # Condition future components on the current sample
    sample_embedding = self.embedder(sample)
    residual = residual + self.decoder(sample_embedding)
    return residual, SampleOutputs(controller_state=sample, logits=logits)

  def distance(
      self,
      residual: Array,
      prev_raw: Array,
      target_raw: Array,
  ) -> tp.Tuple[Array, DistanceOutputs]:
    # Directly connect from the same component at time t-1
    prev_embedding = self.embedder(prev_raw)
    input_ = jnp.concatenate([residual, prev_embedding], axis=-1)
    # Project down to the size desired by the component
    logits = self._encoder(input_)
    # Compute the distance between prediction and target
    distance = self.embedder.distance(logits, target_raw)
    # Auto-regress using the target (aka teacher forcing)
    target_embedding = self.embedder(target_raw)
    residual = residual + self.decoder(target_embedding)
    return residual, DistanceOutputs(distance=distance, logits=logits)


class AutoRegressive(ControllerHead[ControllerType]):
  """Samples components sequentially conditioned on past samples."""

  @classmethod
  def default_config(cls):
    return dict(
        residual_size=128,
        component_depth=0,
        remat=False,
    )

  def __init__(
      self,
      rngs: nnx.Rngs,
      input_size: int,
      embed_controller: embed.Embedding[Controller, ControllerType],
      residual_size: int,
      component_depth: int,
      remat: bool = False,
  ):
    self.embed_controller = embed_controller
    self.to_residual = nnx.Linear(input_size, residual_size, rngs=rngs)
    self.embed_struct = self.embed_controller.map(lambda e: e)
    self.embed_flat = list(self.embed_controller.flatten(self.embed_struct))
    self.remat = remat
    self.res_blocks = nnx.List([
        AutoRegressiveComponent(
            rngs, e,
            residual_size=residual_size, depth=component_depth)
        for e in self.embed_flat
    ])

  def controller_embedding(self) -> embed.Embedding[Controller, ControllerType]:
    return self.embed_controller

  def sample(
      self,
      rngs: nnx.Rngs | nnx.RngStream,
      inputs, prev_controller_state, temperature=None):
    residual = self.to_residual(inputs)
    prev_controller_flat = list(self.embed_controller.flatten(prev_controller_state))

    sample_outputs: list[SampleOutputs] = []
    for res_block, prev in zip(self.res_blocks, prev_controller_flat):
      sample_fn = jax_utils.remat_method(res_block.sample) if self.remat else res_block.sample
      residual, sample = sample_fn(
          rngs(), residual, prev, temperature=temperature)
      sample_outputs.append(sample)

    samples, logits = zip(*sample_outputs)
    return SampleOutputs(
        controller_state=self.embed_controller.unflatten(iter(samples)),
        logits=self.embed_controller.unflatten(iter(logits)),
    )

  def distance(self, inputs, prev_controller_state, target_controller_state):
    residual = self.to_residual(inputs)
    prev_controller_flat = list(self.embed_controller.flatten(prev_controller_state))
    target_controller_flat = list(self.embed_controller.flatten(target_controller_state))

    distance_outputs: list[DistanceOutputs] = []
    for res_block, prev, target in zip(
        self.res_blocks, prev_controller_flat, target_controller_flat):
      distance_fn = jax_utils.remat_method(res_block.distance) if self.remat else res_block.distance
      residual, distance = distance_fn(residual, prev, target)
      distance_outputs.append(distance)

    distances, logits = zip(*distance_outputs)
    return DistanceOutputs(
        distance=self.embed_controller.unflatten(iter(distances)),
        logits=self.embed_controller.unflatten(iter(logits)),
    )


CONSTRUCTORS: dict[str, type[ControllerHead]] = dict(
    independent=Independent,
    autoregressive=AutoRegressive,
)

DEFAULT_CONFIG: dict[str, tp.Any] = dict({k: c.default_config() for k, c in CONSTRUCTORS.items()})
DEFAULT_CONFIG.update(name='independent')

def default_config() -> dict:
  return copy.deepcopy(DEFAULT_CONFIG)

def construct(
    rngs: nnx.Rngs,
    input_size: int,
    embed_controller: embed.Embedding[Controller, ControllerType],
    name: str,
    **config,
) -> ControllerHead:
  """Construct a controller head from config.

  Args:
    rngs: Random number generators for initialization.
    input_size: Size of the input features from the network.
    embed_controller: Controller embedding to use.
    name: Name of the controller head type ('independent' or 'autoregressive').
    **config: Controller head-specific config dicts keyed by name.

  Returns:
    Constructed controller head.
  """
  kwargs = dict(
      config[name],
      rngs=rngs,
      input_size=input_size,
      embed_controller=embed_controller,
  )
  return CONSTRUCTORS[name](**kwargs)
