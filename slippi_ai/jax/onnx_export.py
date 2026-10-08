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

from slippi_ai import flag_utils, onnx_policies, utils
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
      # The graph can't raise errors, so ERROR clamps like CLAMP. Not
      # jnp.clip, since onnxruntime's CUDA Clip has no int32 kernel.
      x = jnp.minimum(jnp.maximum(x, 0), embedding.input_size - 1)
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


def _unpack(
    layout: onnx_policies.Layout,
    packed: dict[str, Array],
) -> dict[str, Array]:
  """jnp version of onnx_policies.unpack."""
  flat = {}
  for key, leaves in layout.items():
    x = packed[key]
    parts = jnp.split(x, [leaf.offset for leaf in leaves[1:]], axis=1)
    for leaf, part in zip(leaves, parts):
      flat[leaf.name] = part.reshape(part.shape[:1] + leaf.shape)
  return flat


def _pack(
    layout: onnx_policies.Layout,
    flat: dict[str, Array],
) -> dict[str, Array]:
  """jnp version of onnx_policies.pack."""
  return {
      key: jnp.concatenate([
          flat[leaf.name].reshape(flat[leaf.name].shape[:1] + (leaf.size,))
          for leaf in leaves], axis=1)
      for key, leaves in layout.items()
  }


class Exported(tp.NamedTuple):
  model: tp.Any  # onnx.ModelProto
  input_layout: onnx_policies.Layout
  output_layout: onnx_policies.Layout


def export(
    policy: policies.Policy,
    game_embedding: embed.Embedding[Game, Game],
    batch_size: tp.Optional[int] = None,
) -> Exported:
  """Exports the policy's step function to an ONNX ModelProto.

  The graph's inputs and outputs are packed by dtype, see onnx_policies.

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
  example = example._replace(
      # Packed with the batch; the graph uses the first entry.
      temperature=np.ones([1], np.float32),
      noise=[np.zeros((1,) + spec.shape, np.float32) for spec in specs],
  )

  def batched_step(inputs: StepInputs) -> StepOutputs:
    return step(inputs._replace(temperature=inputs.temperature[0]))

  input_names, example_leaves, in_treedef = _flatten_with_names(example)
  input_layout = onnx_policies.make_layout('inputs', [
      (name, x.dtype, x.shape[1:])
      for name, x in zip(input_names, example_leaves)])

  output_names, output_leaves, out_treedef = _flatten_with_names(
      jax.eval_shape(batched_step, example))
  output_layout = onnx_policies.make_layout('outputs', [
      (name, x.dtype, x.shape[1:])
      for name, x in zip(output_names, output_leaves)])

  batch_dim = batch_size if batch_size is not None else 'B'
  input_specs = [
      jax.ShapeDtypeStruct(
          (batch_dim, onnx_policies.layout_width(leaves)),
          onnx_policies.layout_dtype(key))
      for key, leaves in input_layout.items()]

  def packed_step(*packed):
    flat = _unpack(input_layout, dict(zip(input_layout, packed)))
    inputs = jax.tree.unflatten(in_treedef, [flat[n] for n in input_names])
    outputs = out_treedef.flatten_up_to(batched_step(inputs))
    packed_outputs = _pack(output_layout, dict(zip(output_names, outputs)))
    return tuple(packed_outputs[key] for key in output_layout)

  def dce_packed_step(*packed):
    # The policy still creates and splits RNG keys, which jax2onnx can't
    # convert. They are unused since sampling reads the noise inputs, so
    # dead-code elimination removes them.
    closed = jax.make_jaxpr(packed_step)(*packed)
    jaxpr, used_inputs = pe.dce_jaxpr(
        closed.jaxpr, [True] * len(closed.jaxpr.outvars))
    used_packed = [x for x, used in zip(packed, used_inputs) if used]
    return tuple(jax_core.eval_jaxpr(jaxpr, closed.consts, *used_packed))

  model = jax2onnx.to_onnx(
      dce_packed_step, input_specs,
      input_names=list(input_layout), output_names=list(output_layout),
      model_name='slippi_ai_policy')
  return Exported(model, input_layout, output_layout)


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


def store_weights_as_float16(model, min_size: int = 1024):
  """Stores large float32 initializers as float16, cast back in the graph.

  onnxruntime constant-folds the casts when creating a session, so the model
  still computes in float32; only the weights are rounded. This halves the
  file size without float16 compute, which was no faster at batch size 1 and
  changed sampled actions noticeably more.
  """
  import onnx
  from onnx import numpy_helper

  graph = model.graph
  initializers = []
  casts = []
  for init in graph.initializer:
    if (init.data_type != onnx.TensorProto.FLOAT
        or np.prod(init.dims, dtype=np.int64) < min_size):
      initializers.append(init)
      continue
    fp16 = numpy_helper.from_array(
        numpy_helper.to_array(init).astype(np.float16), init.name + '_fp16')
    initializers.append(fp16)
    casts.append(onnx.helper.make_node(
        'Cast', [fp16.name], [init.name], to=onnx.TensorProto.FLOAT,
        name=init.name + '_cast'))

  del graph.initializer[:]
  graph.initializer.extend(initializers)
  # Casts first, so the nodes stay topologically sorted.
  nodes = casts + list(graph.node)
  del graph.node[:]
  graph.node.extend(nodes)
  return model


def _opponent_names(state: dict) -> tp.Optional[list[str]]:
  """The opponents of train_two/train_many agents; see AgentSummary."""
  if 'opponents' in state:  # train_many
    return list(state['opponents'])
  if 'opponent' in state:  # train_two
    return [state['opponent']]
  return None


def export_state(
    state: dict,
    batch_size: tp.Optional[int] = None,
    weight_dtype: str = 'float16',
):
  """Exports a JAX checkpoint state to an ONNX ModelProto with metadata.

  Args:
    state: The checkpoint state.
    batch_size: Fixed batch size, needed for CUDA graphs; dynamic if None.
    weight_dtype: Storage dtype of the weights. The model always computes in
      float32; float16 halves the file size, see store_weights_as_float16.
  """
  from slippi_ai import eval_lib, saving

  config = saving.upgrade_config(state['config'])
  if saving.get_platform(config) is not Platform.JAX:
    raise ValueError(
        'Only JAX checkpoints can be exported; convert TF checkpoints with '
        'scripts/convert_tf_checkpoint_to_jax.py first.')

  policy = saving.load_policy_from_state(state)
  jax_utils.cast_module_state_to_dtype(policy, jnp.float32)
  embed_config = embed_config_from_config(config)
  game_embedding = embed_config.make_game_embedding()

  model, input_layout, output_layout = export(
      policy, game_embedding, batch_size=batch_size)

  if weight_dtype == 'float16':
    model = store_weights_as_float16(model)
  elif weight_dtype != 'float32':
    raise ValueError(f'Unsupported weight_dtype {weight_dtype}.')

  onnx_config = _to_json_safe(config)
  onnx_config[saving.PLATFORM_KEY] = Platform.ONNX.value
  onnx_config['policy']['frame_skip'] = policy.frame_skip

  metadata = dict(
      format_version=onnx_policies.FORMAT_VERSION,
      config=onnx_config,
      name_map=state['name_map'],
      agent_config=_to_json_safe(eval_lib.get_agent_config(state)),
      opponents=_opponent_names(state),
      initial_state=_initial_state_metadata(policy),
      # For decoding the graph's actions, see onnx_policies.ControllerDecoder.
      controller=_to_json_safe(dataclasses.asdict(embed_config.controller)),
      batch_size=batch_size,
      weight_dtype=weight_dtype,
      inputs=onnx_policies.layout_to_json(input_layout),
      outputs=onnx_policies.layout_to_json(output_layout),
  )
  entry = model.metadata_props.add()
  entry.key = onnx_policies.METADATA_KEY
  entry.value = json.dumps(metadata)
  return model
