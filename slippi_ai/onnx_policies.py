"""Runs exported policies with onnxruntime, without jax or tensorflow.

See slippi_ai/jax/onnx_export.py for how policies are exported. The ONNX graph
does a single (frame-skipped) agent step, including game encoding. Its
logical inputs and outputs are named by their tree paths:

  inputs: game.*, needs_reset, name, rating, temperature,
          prev_actions.<i>.*, prev_state.*, noise.<i>
  outputs: actions.<i>.*, state.*

The encoded actions and state outputs are fed back in as prev_actions and
prev_state on the next step. The actions are also decoded with numpy, see
ControllerDecoder, and sent to dolphin.

The graph's actual inputs and outputs pack these by dtype into one [B, n]
tensor each, named inputs.<dtype> and outputs.<dtype>, since per-tensor
overhead dominates small models on GPUs. The layouts are in the metadata.
"""

import collections
import io
import json
import math
import typing as tp

from absl import logging
import numpy as np

from slippi_ai import agents, controller_heads, policies, utils, winml
from slippi_ai.action_space import custom_v1
from slippi_ai.agents import Platform
from slippi_ai.controller_heads import SampleOutputs
from slippi_ai.types import (
    BoolArray, Controller, Game, NAME_DTYPE, Rank1, reify_tuple_type,
)

METADATA_KEY = 'slippi_ai'
FORMAT_VERSION = 1

RecurrentState = dict[str, np.ndarray]


def flatten_with_names(nest, prefix: str) -> dict[str, np.ndarray]:
  """Flattens NamedTuples and lists, naming leaves by their paths."""
  if isinstance(nest, tuple) and hasattr(nest, '_fields'):
    items = zip(nest._fields, nest)
  elif isinstance(nest, (list, tuple)):
    items = enumerate(nest)
  else:
    return {prefix: nest}

  flat = {}
  for key, value in items:
    flat.update(flatten_with_names(value, f'{prefix}.{key}'))
  return flat


def unflatten_with_names(template, prefix: str, flat: dict[str, np.ndarray]):
  """Inverse of flatten_with_names for NamedTuple templates."""
  if isinstance(template, tuple) and hasattr(template, '_fields'):
    return type(template)(*[
        unflatten_with_names(value, f'{prefix}.{key}', flat)
        for key, value in zip(template._fields, template)])
  return flat[prefix]


class PackedLeaf(tp.NamedTuple):
  name: str
  shape: tuple[int, ...]  # Without the batch dimension.
  offset: int  # Column offset in the packed tensor.
  size: int  # Number of columns, the product of shape.


# Maps packed tensor names, '<prefix>.<dtype>', to the leaves they contain.
Layout = dict[str, list[PackedLeaf]]


def make_layout(
    prefix: str,
    leaves: tp.Iterable[tuple[str, np.dtype, tuple[int, ...]]],
) -> Layout:
  """Packs (name, dtype, shape without batch) leaves by dtype."""
  layout: Layout = {}
  offsets: dict[str, int] = {}
  for name, dtype, shape in leaves:
    key = f'{prefix}.{np.dtype(dtype).name}'
    leaf = PackedLeaf(
        name, tuple(shape), offsets.get(key, 0), math.prod(shape))
    layout.setdefault(key, []).append(leaf)
    offsets[key] = leaf.offset + leaf.size
  return layout


def layout_dtype(key: str) -> np.dtype:
  return np.dtype(key.split('.', 1)[1])


def layout_width(leaves: list[PackedLeaf]) -> int:
  return leaves[-1].offset + leaves[-1].size


def layout_to_json(layout: Layout) -> dict:
  return {key: [leaf._asdict() for leaf in leaves]
          for key, leaves in layout.items()}


def layout_from_json(layout: dict) -> Layout:
  return {
      key: [PackedLeaf(l['name'], tuple(l['shape']), l['offset'], l['size'])
            for l in leaves]
      for key, leaves in layout.items()
  }


def pack(
    layout: Layout,
    flat: dict[str, np.ndarray],
    batch_size: int,
) -> dict[str, np.ndarray]:
  packed = {}
  for key, leaves in layout.items():
    out = np.empty([batch_size, layout_width(leaves)], layout_dtype(key))
    for leaf in leaves:
      out[:, leaf.offset:leaf.offset + leaf.size] = flat[leaf.name].reshape(
          batch_size, leaf.size)
    packed[key] = out
  return packed


def unpack(layout: Layout, packed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  flat = {}
  for key, leaves in layout.items():
    x = packed[key]
    for leaf in leaves:
      flat[leaf.name] = x[:, leaf.offset:leaf.offset + leaf.size].reshape(
          (x.shape[0],) + leaf.shape)
  return flat


CONTROLLER_TEMPLATE: Controller = reify_tuple_type(Controller)


class ControllerDecoder:
  """numpy version of the controller embedding's decode, without jax.

  Built from the policy's embed controller config (jax.embed.ControllerConfig
  as a dict), which the exporter stores in the model metadata.
  """

  def __init__(self, config: dict):
    self.type = config['type']
    if self.type == 'default':
      self._axis_spacing: int = config['default']['axis_spacing']
      self._shoulder_spacing: int = config['default']['shoulder_spacing']
      # Encoded controllers have the same structure as decoded ones.
      self.template = CONTROLLER_TEMPLATE
    elif self.type == 'custom_v1':
      cv1_config = config['custom_v1']
      self._bucketer = custom_v1.Config(
          c_stick_config=custom_v1.PolarStickConfig(
              **cv1_config['c_stick_config']),
          main_stick_config=custom_v1.PolarStickConfig(
              **cv1_config['main_stick_config']),
      ).create_bucketer()
      self.template = custom_v1.ControllerV1(buttons=None, main_stick=None)
    else:
      raise ValueError(f'Unknown controller type {self.type}.')

  def _decode_default(self, action: Controller) -> Controller:
    # Matches jax.embed.get_controller_embedding.
    def decode_discrete(x: np.ndarray, n: int) -> np.ndarray:
      return (x / n).astype(np.float32)

    def decode_axis(x: np.ndarray) -> np.ndarray:
      if self._axis_spacing:
        return decode_discrete(x, self._axis_spacing)
      return x

    return action._replace(
        main_stick=utils.map_nt(decode_axis, action.main_stick),
        c_stick=utils.map_nt(decode_axis, action.c_stick),
        shoulder=decode_discrete(action.shoulder, self._shoulder_spacing),
    )

  def decode(self, action) -> Controller:
    if self.type == 'default':
      return self._decode_default(action)
    return self._bucketer.decode(action)


def _decode_metadata(custom_metadata: tp.Mapping[str, str]) -> dict:
  if METADATA_KEY not in custom_metadata:
    raise ValueError('ONNX model was not exported by slippi-ai.')
  metadata = json.loads(custom_metadata[METADATA_KEY])
  if metadata['format_version'] != FORMAT_VERSION:
    raise ValueError(
        f'Unsupported ONNX model format version {metadata["format_version"]},'
        f' expected {FORMAT_VERSION}.')
  return metadata


def read_metadata(model: bytes) -> dict:
  import onnxruntime as ort
  options = ort.SessionOptions()
  options.log_severity_level = 3  # Errors only; graph cleanup warns.
  session = ort.InferenceSession(
      model, options, providers=['CPUExecutionProvider'])
  return _decode_metadata(session.get_modelmeta().custom_metadata_map)


def _read_varint(f: tp.BinaryIO) -> int:
  result = shift = 0
  while True:
    byte = f.read(1)
    if not byte:
      raise EOFError
    result |= (byte[0] & 0x7f) << shift
    if byte[0] < 0x80:
      return result
    shift += 7


def _protobuf_fields(f: tp.BinaryIO, wanted: set[int]) -> tp.Iterator[tuple[int, bytes]]:
  """Yields (field number, bytes) for the length-delimited fields in `wanted`.

  Other fields are skipped without being read.
  """
  while True:
    try:
      key = _read_varint(f)
    except EOFError:
      return
    field, wire_type = key >> 3, key & 7
    if wire_type == 0:
      _read_varint(f)
    elif wire_type == 1:
      f.seek(8, 1)
    elif wire_type == 5:
      f.seek(4, 1)
    elif wire_type == 2:
      length = _read_varint(f)
      if field in wanted:
        value = f.read(length)
        if len(value) != length:
          raise EOFError('Truncated ONNX file.')
        yield field, value
      else:
        f.seek(length, 1)
    else:
      raise ValueError(f'Not an ONNX file (protobuf wire type {wire_type}).')


# ModelProto.metadata_props and StringStringEntryProto's key and value.
_METADATA_PROPS_FIELD = 14
_ENTRY_KEY_FIELD = 1
_ENTRY_VALUE_FIELD = 2


def read_metadata_from_file(path: str) -> dict:
  """Like read_metadata, but skips over the graph instead of loading it."""
  custom_metadata = {}
  with open(path, 'rb') as f:
    for _, entry in _protobuf_fields(f, {_METADATA_PROPS_FIELD}):
      fields = dict(_protobuf_fields(
          io.BytesIO(entry), {_ENTRY_KEY_FIELD, _ENTRY_VALUE_FIELD}))
      key = fields.get(_ENTRY_KEY_FIELD, b'').decode()
      custom_metadata[key] = fields.get(_ENTRY_VALUE_FIELD, b'').decode()
  return _decode_metadata(custom_metadata)


def load_state_from_disk(path: str, load_model: bool = True) -> dict:
  """Loads an exported model into a state dict like a pickled checkpoint.

  With load_model=False, only the metadata is read; the state has no model
  and is only good for e.g. eval_lib.AgentSummary.
  """
  if load_model:
    with open(path, 'rb') as f:
      model = f.read()
    metadata = read_metadata(model)
  else:
    model = None
    metadata = read_metadata_from_file(path)

  state = dict(
      config=metadata['config'],
      name_map=metadata['name_map'],
      onnx_metadata=metadata,
  )
  if model is not None:
    state['onnx_model'] = model
  if metadata['agent_config'] is not None:
    state['agent_config'] = metadata['agent_config']
  # Missing in models exported before opponents were saved.
  if metadata.get('opponents') is not None:
    state['opponents'] = metadata['opponents']
  return state


class OnnxControllerHead(controller_heads.ControllerHead[Controller]):
  """The agent decodes controllers itself, so this is mostly a no-op."""

  def dummy_controller(self, shape: tp.Sequence[int]) -> Controller:
    return utils.map_nt(lambda dtype: np.zeros(shape, dtype), CONTROLLER_TEMPLATE)

  def dummy_sample_outputs(self, shape: tp.Sequence[int]) -> SampleOutputs[Controller]:
    # Logits aren't exported.
    return SampleOutputs(controller_state=self.dummy_controller(shape), logits=())

  def decode_controller(self, controller_state: Controller) -> Controller:
    return controller_state


class OnnxPolicy(policies.Policy[Controller, RecurrentState]):
  """An exported model and its metadata. Agents create their own sessions."""

  def __init__(self, model: bytes, metadata: tp.Optional[dict] = None):
    self.model = model
    if metadata is None:
      metadata = read_metadata(model)

    policy_config = metadata['config']['policy']
    self._delay: int = policy_config['delay']
    self.frame_skip: int = policy_config.get('frame_skip', 1)
    self._controller_head = OnnxControllerHead()
    self.controller_decoder = ControllerDecoder(metadata['controller'])

    # The fixed batch size the model was exported with, or None if dynamic.
    self.batch_size: tp.Optional[int] = metadata['batch_size']
    self.input_layout = layout_from_json(metadata['inputs'])
    self.output_layout = layout_from_json(metadata['outputs'])
    # Logical inputs: name -> (dtype, shape without batch).
    self.input_specs = {
        leaf.name: (layout_dtype(key), leaf.shape)
        for key, leaves in self.input_layout.items() for leaf in leaves}
    self.noise_shapes = {
        name: shape for name, (_, shape) in self.input_specs.items()
        if name.startswith('noise.')}

    self._initial_state = {
        name: np.array(entry['value'], dtype=entry['dtype'])
        for name, entry in metadata['initial_state'].items()
    }

    # Check that our naming of game inputs matches the exporter's.
    dummy_game = utils.map_nt(
        lambda dtype: np.zeros([1], dtype), reify_tuple_type(Game))
    game_names = set(flatten_with_names(dummy_game, 'game'))
    graph_game_names = {n for n in self.input_specs if n.startswith('game.')}
    if game_names != graph_game_names:
      raise ValueError(
          'ONNX game inputs do not match slippi_ai.types.Game: '
          f'{sorted(game_names ^ graph_game_names)}')

  @property
  def platform(self) -> Platform:
    return Platform.ONNX

  @property
  def delay(self) -> int:
    return self._delay

  @property
  def controller_head(self) -> OnnxControllerHead:
    return self._controller_head

  def encode_game(self, game: Game) -> Game:
    # Games are encoded inside the graph.
    return game

  def initial_state(self, batch_size: int) -> RecurrentState:
    return {
        name: np.repeat(x, batch_size, axis=0)
        for name, x in self._initial_state.items()
    }

  def build_agent(self, batch_size: int, **kwargs) -> 'OnnxAgent':
    return OnnxAgent(self, batch_size, **kwargs)

  def get_state(self):
    raise NotImplementedError('ONNX policy parameters are part of the graph.')

  def set_state(self, state):
    raise NotImplementedError('ONNX policy parameters are part of the graph.')


def load_policy_from_state(state: dict) -> OnnxPolicy:
  return OnnxPolicy(state['onnx_model'], state.get('onnx_metadata'))


CUDA = 'CUDAExecutionProvider'
CPU = 'CPUExecutionProvider'
TENSORRT_RTX = winml.TENSORRT_RTX
# Providers that never need Windows ML.
_BUILT_IN = (CPU, CUDA, 'DmlExecutionProvider')


def _plugin_providers() -> set[str]:
  """Providers that are chosen by device rather than by name: Windows ML's.

  onnxruntime lists them as available once registered, but creating a
  session with them by name fails.
  """
  return set(winml.initialize())


def default_providers() -> list[str]:
  """TensorRT-RTX (Windows ML) if available, else CUDA, else CPU.

  Other providers must be asked for explicitly: DirectML was slower than CPU
  in our tests, and TensorRT needs a separate install.
  """
  import onnxruntime as ort
  if TENSORRT_RTX in _plugin_providers():
    return [TENSORRT_RTX, CPU]
  if CUDA in ort.get_available_providers():
    return [CUDA, CPU]
  return [CPU]


def _plugin_provider_options(provider: str, cuda_graph: bool) -> dict[str, str]:
  if provider == TENSORRT_RTX:
    return {
        'enable_cuda_graph': str(int(cuda_graph)),
        # Compiled kernels, a few MB per model, which halve later setups.
        'nv_runtime_cache_path': winml.cache_dir('tensorrt-rtx'),
    }
  return {}


class SessionRunner:
  """Runs the graph on packed numpy inputs, returning packed numpy outputs.

  With CUDA and a model exported with a fixed batch size, the step is captured
  as a CUDA graph and replayed, which removes most per-kernel launch overhead.
  This needs fixed device buffers, so inputs and outputs are copied through
  IOBinding instead of passed to session.run.
  """

  def __init__(
      self,
      policy: OnnxPolicy,
      batch_size: int,
      providers: tp.Optional[tp.Sequence[str]] = None,
      cuda_graph: bool = True,
  ):
    import onnxruntime as ort

    providers = list(providers or default_providers())
    if CUDA in providers and hasattr(ort, 'preload_dlls'):
      # Finds CUDA and cuDNN from the nvidia-* pip packages, if installed.
      ort.preload_dlls()

    # TensorRT-RTX captures its own CUDA graphs; with CUDA we bind buffers.
    cuda_graph = cuda_graph and providers[0] in (CUDA, TENSORRT_RTX)
    if cuda_graph and policy.batch_size != batch_size:
      logging.warning(
          'Not using CUDA graphs: the model has batch size %s, not %d. '
          'Export with --batch_size=%d to use them.',
          policy.batch_size, batch_size, batch_size)
      cuda_graph = False

    provider_options = [
        (p, {'enable_cuda_graph': '1'}) if p == CUDA and cuda_graph else p
        for p in providers]

    session_options = ort.SessionOptions()
    # Silences warnings about unused initializers in the exported graph.
    session_options.log_severity_level = 3
    if providers[0] not in _BUILT_IN and providers[0] in _plugin_providers():
      self.session = self._plugin_session(
          ort, policy, session_options, providers[0], cuda_graph)
      cuda_graph = False
    else:
      self.session = ort.InferenceSession(
          policy.model, session_options, providers=provider_options)
    self.providers = self.session.get_providers()
    # Dead inputs may have been pruned from the graph.
    self.input_names = [i.name for i in self.session.get_inputs()]
    self.output_names = [o.name for o in self.session.get_outputs()]

    # CUDA may have failed to load, falling back to CPU.
    self.cuda_graph = cuda_graph and self.providers[0] == CUDA
    if self.cuda_graph:
      def device_buffer(key: str, leaves: list[PackedLeaf]):
        return ort.OrtValue.ortvalue_from_shape_and_type(
            [batch_size, layout_width(leaves)], layout_dtype(key), 'cuda', 0)

      self._inputs = {
          key: device_buffer(key, policy.input_layout[key])
          for key in self.input_names}
      self._outputs = {
          key: device_buffer(key, policy.output_layout[key])
          for key in self.output_names}
      self._binding = self.session.io_binding()
      for key, value in self._inputs.items():
        self._binding.bind_ortvalue_input(key, value)
      for key, value in self._outputs.items():
        self._binding.bind_ortvalue_output(key, value)

  @staticmethod
  def _plugin_session(ort, policy, session_options, provider, cuda_graph):
    """A session on a provider chosen by device, falling back to the CPU.

    Such providers can't be combined with others by name; onnxruntime runs
    any nodes they don't support on the CPU.
    """
    devices = [d for d in ort.get_ep_devices() if d.ep_name == provider]
    session_options.add_provider_for_devices(
        devices, _plugin_provider_options(provider, cuda_graph))
    logging.info('Creating a %s session; the first time for a model can '
                 'take a few seconds.', provider)
    try:
      return ort.InferenceSession(policy.model, session_options)
    except Exception as e:  # pylint: disable=broad-except
      logging.warning('Could not use %s, using the CPU: %s', provider, e)
      cpu_options = ort.SessionOptions()
      cpu_options.log_severity_level = session_options.log_severity_level
      return ort.InferenceSession(policy.model, cpu_options, providers=[CPU])

  def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not self.cuda_graph:
      outputs = self.session.run(
          self.output_names, {key: inputs[key] for key in self.input_names})
      return dict(zip(self.output_names, outputs))

    for key, value in self._inputs.items():
      value.update_inplace(inputs[key])
    self.session.run_with_iobinding(self._binding)
    return {key: value.numpy() for key, value in self._outputs.items()}


class OnnxAgent(agents.BasicAgent[Controller, RecurrentState]):
  """Steps an OnnxPolicy, tracking the recurrent state and previous actions."""

  def __init__(
      self,
      policy: OnnxPolicy,
      batch_size: int,
      name_code: tp.Union[int, tp.Sequence[int]],
      rating: float = 0,
      seed: tp.Optional[int] = None,
      sample_kwargs: tp.Optional[dict] = None,
      compile: bool = True,  # Unused; ONNX graphs are always compiled.
      providers: tp.Optional[tp.Sequence[str]] = None,
      cuda_graph: bool = True,
  ):
    """See SessionRunner for providers and cuda_graph."""
    del compile
    sample_kwargs = sample_kwargs or {}
    self._policy = policy
    self._batch_size = batch_size
    self.set_name_code(name_code)
    self._rating = rating
    self._temperature = np.array(
        sample_kwargs.get('temperature', 1.0), dtype=np.float32)
    self._rng = np.random.default_rng(seed)
    self.runner = SessionRunner(policy, batch_size, providers, cuda_graph)

    # Mirror the prev_actions inputs, which are encoded controllers.
    self._prev_actions = {
        name: np.zeros((batch_size,) + shape, dtype)
        for name, (dtype, shape) in policy.input_specs.items()
        if name.startswith('prev_actions.')
    }
    self._hidden_state = policy.initial_state(batch_size)

    self._sample_outputs = collections.deque[SampleOutputs[Controller]]()
    self._needs_reset: BoolArray[Rank1] = np.full([batch_size], False)

  @property
  def platform(self) -> Platform:
    return Platform.ONNX

  @property
  def name_code(self) -> np.ndarray:
    return self._name_code

  def set_name_code(self, name_code: tp.Union[int, tp.Sequence[int]]):
    if isinstance(name_code, int):
      name_code = [name_code] * self._batch_size
    elif len(name_code) != self._batch_size:
      raise ValueError(f'name_code list must have length batch_size={self._batch_size}')
    self._name_code = np.array(name_code, dtype=NAME_DTYPE)

  @property
  def rating(self) -> float:
    return self._rating

  def warmup(self):
    game = utils.map_nt(
        lambda dtype: np.zeros([self._batch_size], dtype),
        reify_tuple_type(Game))
    self.step(game, np.full([self._batch_size], False))

  def dummy_sample_outputs(self, shape: tp.Sequence[int]) -> SampleOutputs[Controller]:
    return self._policy.controller_head.dummy_sample_outputs(shape)

  def hidden_state(self) -> RecurrentState:
    return {name: x.copy() for name, x in self._hidden_state.items()}

  def _run(self, game: Game, needs_reset: np.ndarray) -> list[SampleOutputs[Controller]]:
    feed = flatten_with_names(game, 'game')
    feed.update(self._prev_actions)
    feed.update(self._hidden_state)
    feed['needs_reset'] = needs_reset
    feed['name'] = self._name_code
    feed['rating'] = np.full([self._batch_size], self._rating, np.float32)
    # Packed with the batch; the graph uses the first entry.
    feed['temperature'] = np.full([self._batch_size], self._temperature)
    for name, shape in self._policy.noise_shapes.items():
      feed[name] = self._rng.random(
          (self._batch_size,) + shape, dtype=np.float32)

    outputs = unpack(
        self._policy.output_layout,
        self.runner.run(
            pack(self._policy.input_layout, feed, self._batch_size)))

    for name, value in outputs.items():
      if name.startswith('actions.'):
        self._prev_actions['prev_' + name] = value
      elif name.startswith('state.'):
        self._hidden_state['prev_' + name] = value

    decoder = self._policy.controller_decoder
    return [
        SampleOutputs(
            controller_state=decoder.decode(unflatten_with_names(
                decoder.template, f'actions.{i}', outputs)),
            logits=())
        for i in range(self._policy.frame_skip)
    ]

  def step(
      self,
      game: Game[Rank1],
      needs_reset: BoolArray[Rank1],
  ) -> SampleOutputs[Controller]:
    """Doesn't take into account delay."""
    self._needs_reset |= needs_reset

    # With frame_skip > 1, the policy is only run every frame_skip frames.
    if not self._sample_outputs:
      self._sample_outputs.extend(self._run(game, self._needs_reset))
      self._needs_reset[:] = False

    return self._sample_outputs.popleft()

  def multi_step(
      self,
      states: list[tuple[Game[Rank1], BoolArray[Rank1]]],
  ) -> list[SampleOutputs[Controller]]:
    return [self.step(game, needs_reset) for game, needs_reset in states]
