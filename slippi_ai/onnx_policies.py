"""Runs exported policies with onnxruntime, without jax or tensorflow.

See slippi_ai/jax/onnx_export.py for how policies are exported. The ONNX graph
does a single (frame-skipped) agent step, including game encoding and
controller decoding. Its inputs and outputs are named by their tree paths:

  inputs: game.*, needs_reset, name, rating, temperature,
          prev_actions.<i>.*, prev_state.*, noise.<i>
  outputs: actions.<i>.*, controllers.<i>.*, state.*

The encoded actions and state outputs are fed back in as prev_actions and
prev_state on the next step, and the decoded controllers are sent to dolphin.
"""

import collections
import json
import typing as tp

import numpy as np

from slippi_ai import agents, controller_heads, policies, utils
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


CONTROLLER_TEMPLATE: Controller = reify_tuple_type(Controller)


def _session_metadata(session) -> dict:
  custom_metadata = session.get_modelmeta().custom_metadata_map
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
  return _session_metadata(
      ort.InferenceSession(model, providers=['CPUExecutionProvider']))


def load_state_from_disk(path: str) -> dict:
  """Loads an exported model into a state dict like a pickled checkpoint."""
  with open(path, 'rb') as f:
    model = f.read()
  metadata = read_metadata(model)
  state = dict(
      config=metadata['config'],
      name_map=metadata['name_map'],
      onnx_model=model,
  )
  if metadata['agent_config'] is not None:
    state['agent_config'] = metadata['agent_config']
  return state


class OnnxControllerHead(controller_heads.ControllerHead[Controller]):
  """The graph decodes controllers itself, so this is mostly a no-op."""

  def dummy_controller(self, shape: tp.Sequence[int]) -> Controller:
    return utils.map_nt(lambda dtype: np.zeros(shape, dtype), CONTROLLER_TEMPLATE)

  def dummy_sample_outputs(self, shape: tp.Sequence[int]) -> SampleOutputs[Controller]:
    # Logits aren't exported.
    return SampleOutputs(controller_state=self.dummy_controller(shape), logits=())

  def decode_controller(self, controller_state: Controller) -> Controller:
    return controller_state


class OnnxPolicy(policies.Policy[Controller, RecurrentState]):

  def __init__(
      self,
      model: bytes,
      providers: tp.Optional[tp.Sequence[str]] = None,
  ):
    import onnxruntime as ort

    if providers is None:
      providers = ort.get_available_providers()

    session_options = ort.SessionOptions()
    # Silences warnings about unused initializers in the exported graph.
    session_options.log_severity_level = 3
    self.session = ort.InferenceSession(
        model, session_options, providers=list(providers))

    metadata = _session_metadata(self.session)
    policy_config = metadata['config']['policy']
    self._delay: int = policy_config['delay']
    self.frame_skip: int = policy_config.get('frame_skip', 1)
    self._controller_head = OnnxControllerHead()

    self.input_names = [i.name for i in self.session.get_inputs()]
    self.output_names = [o.name for o in self.session.get_outputs()]
    self.noise_shapes = {
        i.name: tuple(i.shape[1:]) for i in self.session.get_inputs()
        if i.name.startswith('noise.')}

    self._initial_state = {
        name: np.array(entry['value'], dtype=entry['dtype'])
        for name, entry in metadata['initial_state'].items()
    }

    # Check that our naming of game inputs matches the exporter's.
    dummy_game = utils.map_nt(
        lambda dtype: np.zeros([1], dtype), reify_tuple_type(Game))
    game_names = set(flatten_with_names(dummy_game, 'game'))
    graph_game_names = {n for n in self.input_names if n.startswith('game.')}
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


def load_policy_from_state(state: dict, **kwargs) -> OnnxPolicy:
  return OnnxPolicy(state['onnx_model'], **kwargs)


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
  ):
    del compile
    sample_kwargs = sample_kwargs or {}
    self._policy = policy
    self._batch_size = batch_size
    self.set_name_code(name_code)
    self._rating = rating
    self._temperature = np.array(
        sample_kwargs.get('temperature', 1.0), dtype=np.float32)
    self._rng = np.random.default_rng(seed)

    # Mirror the prev_actions inputs, which are encoded controllers.
    self._prev_actions = {
        name: np.zeros(
            [batch_size] + i.shape[1:], dtype=_ORT_TO_NUMPY[i.type])
        for i in policy.session.get_inputs()
        if (name := i.name).startswith('prev_actions.')
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
    feed['temperature'] = self._temperature
    for name, shape in self._policy.noise_shapes.items():
      feed[name] = self._rng.random(
          (self._batch_size,) + shape, dtype=np.float32)

    outputs = self._policy.session.run(
        self._policy.output_names,
        {name: feed[name] for name in self._policy.input_names})
    outputs = dict(zip(self._policy.output_names, outputs))

    for name, value in outputs.items():
      if name.startswith('actions.'):
        self._prev_actions['prev_' + name] = value
      elif name.startswith('state.'):
        self._hidden_state['prev_' + name] = value

    return [
        SampleOutputs(
            controller_state=unflatten_with_names(
                CONTROLLER_TEMPLATE, f'controllers.{i}', outputs),
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


_ORT_TO_NUMPY = {
    'tensor(float)': np.float32,
    'tensor(double)': np.float64,
    'tensor(bool)': np.bool_,
    'tensor(uint8)': np.uint8,
    'tensor(uint16)': np.uint16,
    'tensor(int8)': np.int8,
    'tensor(int16)': np.int16,
    'tensor(int32)': np.int32,
    'tensor(int64)': np.int64,
}
