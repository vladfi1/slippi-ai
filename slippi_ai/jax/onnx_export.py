"""Export JAX policies to ONNX for inference without jax or tensorflow.

The exported graph performs one agent step: it takes the raw game state, the
previous (encoded) controller, the recurrent state and per-component uniform
noise, and returns the sampled (encoded) controller along with the new
recurrent state. Controllers are decoded outside the graph, in numpy, using
the controller config in the model metadata.

Sampling noise is an explicit input rather than an RNG key so that the graph
uses only standard ONNX ops. During tracing, `jax.random.categorical` and
`jax.random.bernoulli` are replaced by the equivalent computations on that
noise: `argmax(logits + gumbel(u))` and `u < p`, the same formulas jax uses.
"""

import contextlib
import dataclasses
import enum
import json
import logging
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax._src import core as jax_core
from jax._src.interpreters import partial_eval as pe

from slippi_ai import flag_utils, utils
from slippi_ai.agents import Platform
from slippi_ai.data import StateAction
from slippi_ai.types import Game, NAME_DTYPE, reify_tuple_type
from slippi_ai.jax import embed, jax_utils, policies
from slippi_ai.jax import saving as jax_saving

Array = jax.Array


@dataclasses.dataclass
class NoiseSpec:
  kind: str  # 'categorical' or 'bernoulli'
  shape: tuple[int, ...]  # without the batch dimension


class _NoiseSampler:
  """Replaces jax.random sampling with computations on provided noise."""

  def __init__(self, noise: tp.Optional[tp.Sequence[Array]] = None):
    # With no noise, records the sampling calls made instead.
    self._noise = None if noise is None else list(noise)
    self.specs: list[NoiseSpec] = []

  def _next(self, kind: str, shape: tuple[int, ...]) -> tp.Optional[Array]:
    index = len(self.specs)
    self.specs.append(NoiseSpec(kind, tuple(shape[1:])))
    if self._noise is None:
      return None
    noise = self._noise[index]
    if noise.shape[1:] != tuple(shape[1:]):
      raise ValueError(
          f'Noise {index} has shape {noise.shape}, expected {shape}.')
    return noise

  def categorical(self, key, logits, axis=-1, **kwargs):
    del key, kwargs
    if axis != -1:
      raise NotImplementedError('Only axis=-1 is supported.')
    u = self._next('categorical', logits.shape)
    if u is None:
      return jnp.argmax(logits, axis=-1)
    # Clip so that u=0 doesn't produce nan.
    u = jnp.clip(u, jnp.finfo(u.dtype).tiny, 1.)
    gumbel = -jnp.log(-jnp.log(u))
    return jnp.argmax(logits + gumbel, axis=-1)

  def bernoulli(self, key, p, shape=None):
    del key
    if shape is not None:
      raise NotImplementedError('Explicit shape is not supported.')
    u = self._next('bernoulli', p.shape)
    if u is None:
      return p > 0.5
    return u < p

  @contextlib.contextmanager
  def patch(self):
    original = (jax.random.categorical, jax.random.bernoulli)
    jax.random.categorical = self.categorical
    jax.random.bernoulli = self.bernoulli
    try:
      yield self
    finally:
      jax.random.categorical, jax.random.bernoulli = original


def _encode_leaf(embedding: embed.Embedding, x: Array) -> Array:
  """jnp version of `embedding.from_state` for leaf embeddings."""
  from_state = type(embedding).from_state

  if from_state is embed.Embedding.from_state:
    return x.astype(embedding.dtype)

  if from_state is embed.DiscreteEmbedding.from_state:
    assert isinstance(embedding, embed.DiscreteEmbedding)
    return (x * embedding.n + 0.5).astype(embedding.dtype)

  if from_state is embed.OneHotEmbedding.from_state:
    assert isinstance(embedding, embed.OneHotEmbedding)
    policy = embedding.one_hot_policy
    # onnxruntime lacks Clip and Where kernels for some small integer types
    # such as uint16, so do the range logic in int32.
    x = x.astype(jnp.int32)
    if policy in (embed.OneHotPolicy.CLAMP, embed.OneHotPolicy.ERROR):
      # The graph can't raise errors, so ERROR clamps like CLAMP.
      x = jnp.clip(x, 0, embedding.input_size - 1)
    elif policy is embed.OneHotPolicy.EXTRA:
      invalid = (x < 0) | (x >= embedding.input_size)
      x = jnp.where(invalid, embedding.input_size, x)
    # EMPTY: one_hot already maps invalid inputs to all zeros.
    return x.astype(embedding.dtype)

  raise NotImplementedError(
      f'ONNX export does not support {type(embedding).__name__}.from_state')


def encode_game(game_embedding: embed.Embedding[Game, Game], game: Game) -> Game:
  """jnp version of `game_embedding.from_state`, for use inside the graph."""
  return game_embedding.map(_encode_leaf, game)


def embed_config_from_config(config: dict) -> embed.EmbedConfig:
  config = jax_saving.upgrade_config(config)
  return flag_utils.dataclass_from_dict(embed.EmbedConfig, config['embed'])


def game_embedding_from_config(config: dict) -> embed.Embedding[Game, Game]:
  return embed_config_from_config(config).make_game_embedding()


class StepInputs(tp.NamedTuple):
  game: Game  # Raw (unencoded) game.
  needs_reset: Array
  name: Array
  rating: Array
  temperature: Array  # Scalar.
  prev_actions: list  # list[ControllerType], one per frame_skip.
  prev_state: tp.Any  # RecurrentState
  noise: list[Array]


class StepOutputs(tp.NamedTuple):
  # Encoded controllers, one per frame_skip, fed back in as prev_actions.
  actions: list
  state: tp.Any  # RecurrentState


def make_step_fn(
    policy: policies.Policy,
    game_embedding: embed.Embedding[Game, Game],
):
  """Returns step(StepInputs) -> StepOutputs, sampling with inputs.noise.

  The step function optionally takes a _NoiseSampler, used to record the
  noise needed rather than provide it.
  """
  # Merging from pure arrays inside the trace avoids nnx trace-level errors
  # from closing over Variables, as in jax_utils.CachedFunctionalJit.
  graphdef, state = nnx.split(policy)
  params = state.to_pure_dict()

  def step(
      inputs: StepInputs,
      sampler: tp.Optional[_NoiseSampler] = None,
  ) -> StepOutputs:
    if sampler is None:
      sampler = _NoiseSampler(inputs.noise)
    policy = nnx.merge(graphdef, params, copy=True)
    rngs = nnx.Rngs(0)  # Unused: sampling reads from the noise instead.
    state_action = StateAction(
        state=encode_game(game_embedding, inputs.game),
        action=inputs.prev_actions,
        name=inputs.name,
        rating=inputs.rating,
    )
    with sampler.patch():
      sample_outputs, new_state = policy.sample(
          rngs, state_action, inputs.prev_state, inputs.needs_reset,
          temperature=inputs.temperature)
    actions = [so.controller_state for so in sample_outputs]
    return StepOutputs(actions, new_state)

  return step


def dummy_step_inputs(policy: policies.Policy, batch_size: int) -> StepInputs:
  game = utils.map_nt(
      lambda dtype: np.zeros([batch_size], dtype), reify_tuple_type(Game))
  return StepInputs(
      game=game,
      needs_reset=np.zeros([batch_size], np.bool_),
      name=np.zeros([batch_size], NAME_DTYPE),
      rating=np.zeros([batch_size], np.float32),
      temperature=np.ones([], np.float32),
      prev_actions=[policy.controller_head.dummy_controller([batch_size])]
      * policy.frame_skip,
      prev_state=jax.tree.map(
          np.asarray, policy.initial_state(batch_size)),
      noise=[],
  )


def noise_specs(
    policy: policies.Policy,
    game_embedding: embed.Embedding[Game, Game],
) -> list[NoiseSpec]:
  """Determines the noise inputs needed by tracing the step function."""
  step = make_step_fn(policy, game_embedding)
  sampler = _NoiseSampler()
  jax.eval_shape(lambda x: step(x, sampler), dummy_step_inputs(policy, 1))
  return sampler.specs


def _path_name(path) -> str:
  # Matches onnx_policies.flatten_with_names for NamedTuples and lists.
  name = jax.tree_util.keystr(path, simple=True, separator='.')
  return name.replace('[', '').replace(']', '')


def _flatten_with_names(tree) -> tuple[list[str], list[tp.Any], tp.Any]:
  leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(tree)
  names = [_path_name(path) for path, _ in leaves_with_paths]
  leaves = [leaf for _, leaf in leaves_with_paths]
  return names, leaves, treedef


def export(
    policy: policies.Policy,
    game_embedding: embed.Embedding[Game, Game],
    batch_size: tp.Optional[int] = None,
):
  """Exports the policy's step function to an ONNX ModelProto.

  Args:
    policy: The policy to export.
    game_embedding: The policy's game embedding, see game_embedding_from_config.
    batch_size: Fixed batch size, or None for a dynamic batch dimension.
  """
  import jax2onnx
  # jax2onnx logs every input spec at INFO level.
  logging.getLogger('jax2onnx').setLevel(logging.WARNING)

  specs = noise_specs(policy, game_embedding)
  step = make_step_fn(policy, game_embedding)

  example = dummy_step_inputs(policy, 1)
  example = example._replace(noise=[
      np.zeros((1,) + spec.shape, np.float32) for spec in specs])

  input_names, example_leaves, in_treedef = _flatten_with_names(example)

  output_names, _, out_treedef = _flatten_with_names(
      jax.eval_shape(step, example))

  batch_dim = batch_size if batch_size is not None else 'B'

  def input_spec(name: str, x: np.ndarray):
    if name == 'temperature':
      return jax.ShapeDtypeStruct((), x.dtype)
    return jax.ShapeDtypeStruct((batch_dim,) + x.shape[1:], x.dtype)

  input_specs = [
      input_spec(n, x) for n, x in zip(input_names, example_leaves)]

  def flat_step(*leaves):
    inputs = jax.tree.unflatten(in_treedef, leaves)
    outputs = out_treedef.flatten_up_to(step(inputs))
    # ONNX graph outputs need distinct values, so copy any repeats.
    seen = set()
    for i, x in enumerate(outputs):
      if id(x) in seen:
        outputs[i] = jnp.copy(x)
      seen.add(id(outputs[i]))
    return tuple(outputs)

  def dce_flat_step(*leaves):
    # The policy still creates and splits RNG keys, which jax2onnx can't
    # convert. They are unused since sampling reads the noise inputs, so
    # dead-code elimination removes them.
    closed = jax.make_jaxpr(flat_step)(*leaves)
    jaxpr, used_inputs = pe.dce_jaxpr(
        closed.jaxpr, [True] * len(closed.jaxpr.outvars))
    used_leaves = [x for x, used in zip(leaves, used_inputs) if used]
    return tuple(jax_core.eval_jaxpr(jaxpr, closed.consts, *used_leaves))

  return jax2onnx.to_onnx(
      dce_flat_step, input_specs,
      input_names=input_names, output_names=output_names,
      model_name='slippi_ai_policy')


def _to_json_safe(x):
  if isinstance(x, enum.Enum):
    return x.value
  if isinstance(x, dict):
    return {k: _to_json_safe(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [_to_json_safe(v) for v in x]
  return x


def _initial_state_metadata(policy: policies.Policy) -> dict[str, dict]:
  names, leaves, _ = _flatten_with_names(
      StepInputs(*[None] * 6, prev_state=policy.initial_state(1), noise=None))
  # The ONNX agent broadcasts the batch-1 initial state.
  leaves_2 = jax.tree.leaves(policy.initial_state(2))
  for name, x, x2 in zip(names, leaves, leaves_2):
    if not np.array_equal(np.repeat(np.asarray(x), 2, axis=0), np.asarray(x2)):
      raise NotImplementedError(f'Initial state {name} depends on the batch.')
  return {
      name: dict(dtype=str(np.asarray(x).dtype), value=np.asarray(x).tolist())
      for name, x in zip(names, leaves)
  }


def export_state(state: dict, batch_size: tp.Optional[int] = None):
  """Exports a JAX checkpoint state to an ONNX ModelProto with metadata."""
  from slippi_ai import eval_lib, saving
  from slippi_ai import onnx_policies

  config = saving.upgrade_config(state['config'])
  if saving.get_platform(config) is not Platform.JAX:
    raise ValueError(
        'Only JAX checkpoints can be exported; convert TF checkpoints with '
        'scripts/convert_tf_checkpoint_to_jax.py first.')

  policy = saving.load_policy_from_state(state)
  jax_utils.cast_module_state_to_dtype(policy, jnp.float32)
  embed_config = embed_config_from_config(config)
  game_embedding = embed_config.make_game_embedding()

  model = export(policy, game_embedding, batch_size=batch_size)

  onnx_config = _to_json_safe(config)
  onnx_config[saving.PLATFORM_KEY] = Platform.ONNX.value
  onnx_config['policy']['frame_skip'] = policy.frame_skip

  metadata = dict(
      format_version=onnx_policies.FORMAT_VERSION,
      config=onnx_config,
      name_map=state['name_map'],
      agent_config=_to_json_safe(eval_lib.get_agent_config(state)),
      initial_state=_initial_state_metadata(policy),
      # For decoding the graph's actions, see onnx_policies.ControllerDecoder.
      controller=_to_json_safe(dataclasses.asdict(embed_config.controller)),
  )
  entry = model.metadata_props.add()
  entry.key = onnx_policies.METADATA_KEY
  entry.value = json.dumps(metadata)
  return model
