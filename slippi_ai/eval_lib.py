import contextlib
import enum
import dataclasses
import logging
import os
import threading, queue
import typing as tp

import numpy as np
import fancyflags as ff

import melee

from slippi_ai import (
  dolphin, data, utils, nametags,
  observations, flag_utils, policies,
)
from slippi_ai.policies import RecurrentState
from slippi_ai.types import Game, Controller
from slippi_ai.agents import BasicAgent, BoolArray

import slippi_ai.mirror as mirror_lib
from slippi_ai.controller_lib import send_controller
from slippi_ai.controller_heads import SampleOutputs, ControllerType
from slippi_ai import saving
from slippi_db.parse_libmelee import Parser

def disable_gpus():
  import tensorflow as tf
  tf.config.set_visible_devices([], 'GPU')


class FakeAgent(BasicAgent[ControllerType, RecurrentState]):

  def __init__(
      self,
      policy: policies.Policy[ControllerType, RecurrentState],
      batch_size: int,
  ):
    self._sample_outputs = policy.controller_head.dummy_sample_outputs([batch_size])
    self._hidden_state = policy.initial_state(batch_size)
    self._batch_size = batch_size
    self._name_code = np.zeros(batch_size, dtype=data.NAME_DTYPE)

  def hidden_state(self) -> RecurrentState:
    return super().hidden_state()

  @property
  def name_code(self):
    return self._name_code

  def set_name_code(self, name_code: tp.Union[int, tp.Sequence[int]]):
    if isinstance(name_code, int):
      self._name_code = np.array([name_code] * self._batch_size, dtype=data.NAME_DTYPE)
    else:
      self._name_code = np.array(name_code, dtype=data.NAME_DTYPE)

  def step(
      self,
      game: Game,
      needs_reset: np.ndarray
  ) -> SampleOutputs[ControllerType]:
    del game, needs_reset
    return self._sample_outputs

  def multi_step(
      self,
      states: list[tuple[Game, np.ndarray]],
  ) -> list[SampleOutputs[ControllerType]]:
    return [self._sample_outputs] * len(states)

def build_basic_agent(
    policy: policies.Policy[ControllerType, RecurrentState],
    batch_size: int,
    fake: bool = False,
    tf: dict[str, tp.Any] = {},
    jax: dict[str, tp.Any] = {},
    **kwargs,
) -> tp.Union[FakeAgent, BasicAgent[ControllerType, RecurrentState]]:
  if fake:
    return FakeAgent(policy, batch_size)

  framework_kwargs = tf if policy.platform == policies.Platform.TF else jax

  return policy.build_agent(batch_size, **kwargs, **framework_kwargs)

class DelayedAgent(tp.Generic[ControllerType, RecurrentState]):
  """Wraps a BasicAgent with delay."""

  def __init__(
      self,
      policy: policies.Policy[ControllerType, RecurrentState],
      observation_config: observations.ObservationConfig,
      batch_size: int,
      console_delay: int = 0,
      batch_steps: int = 0,
      **agent_kwargs,
  ):
    self.observation_config = observation_config
    self._batch_steps = batch_steps
    self._input_queue = []

    self._agent = build_basic_agent(
        policy=policy,
        batch_size=batch_size,
        **agent_kwargs)
    self.warmup = self._agent.warmup
    self.policy = policy

    if console_delay > policy.delay - (self.batch_steps - 1):
      raise ValueError(
          f'console delay ({console_delay}) must be <='
          f' policy delay ({policy.delay}) - batch_steps ({self.batch_steps}) + 1')

    self.delay = policy.delay - console_delay
    self._output_queue: utils.PeekableQueue[SampleOutputs[ControllerType]] \
      = utils.PeekableQueue()

    self.dummy_sample_outputs = policy.controller_head.dummy_sample_outputs([batch_size])
    for _ in range(self.delay):
      self._output_queue.put(self.dummy_sample_outputs)

    self.peek_n = self._output_queue.peek_n

    # TODO: put this in the BasicAgent?
    self.step_profiler = utils.Profiler(burnin=1)

  @property
  def batch_steps(self) -> int:
    return self._batch_steps or 1

  @property
  def hidden_state(self) -> RecurrentState:
    return self._agent.hidden_state()

  def pop(self) -> SampleOutputs[ControllerType]:
    outputs = self._output_queue.get()
    if self.policy.platform == policies.Platform.JAX:
      outputs = utils.map_single_structure(np.asarray, outputs)
    return outputs

  def decode_controller(self, controller: ControllerType) -> Controller:
    return self.policy.controller_head.decode_controller(controller)

  @property
  def name_code(self):
    return self._agent.name_code

  def step(
      self,
      game: Game,
      needs_reset: np.ndarray[tuple[int], np.dtype[np.bool]],
  ) -> SampleOutputs[ControllerType]:
    """Synchronous agent step."""
    self.push(game, needs_reset)
    delayed_controller = self.pop()
    return delayed_controller

  # Present the same interface as the async agent.
  def push(self, game: Game, needs_reset: np.ndarray):
    if self._batch_steps == 0:
      with self.step_profiler:
        sampled_controller = self._agent.step(game, needs_reset)
      self._output_queue.put(sampled_controller)
      return

    self._input_queue.append((game, needs_reset))
    if len(self._input_queue) == self._batch_steps:
      with self.step_profiler:
        sample_outputs = self._agent.multi_step(self._input_queue)
      for output in sample_outputs:
        self._output_queue.put(output)
      self._input_queue = []

  @contextlib.contextmanager
  def run(self):
    """For compatibility with the async agent."""
    yield self

  def start(self):
    """For compatibility with the async agent."""

  def stop(self):
    """For compatibility with the async agent."""


def _run_agent(
    agent: BasicAgent[ControllerType, RecurrentState],
    state_queue: queue.Queue,
    controller_queue: queue.Queue,
    batch_steps: int,
    state_queue_profiler: utils.Profiler,
    step_profiler: utils.Profiler,
):
  """Breaks circular references."""
  # Not needed anymore since we rely on context managers instead of __del__.
  while batch_steps == 0:
    with state_queue_profiler:
      next_item: tp.Optional[tuple[Game, BoolArray]] = state_queue.get()
    if next_item is None:
      state_queue.task_done()
      return
    game, needs_reset = next_item
    with step_profiler:
      sampled_controller = agent.step(game, needs_reset)
    controller_queue.put(sampled_controller)
    state_queue.task_done()

  while batch_steps > 0:
    states = []
    for _ in range(batch_steps):
      with state_queue_profiler:
        next_item: tp.Optional[tuple[Game, BoolArray]] = state_queue.get()
      if next_item is None:
        state_queue.task_done()
        return
      states.append(next_item)

    with step_profiler:
      sampled_controllers = agent.multi_step(states)

    for controller in sampled_controllers:
      controller_queue.put(controller)
      state_queue.task_done()

class AsyncDelayedAgent(tp.Generic[ControllerType, RecurrentState]):
  """Delayed agent that runs inference asynchronously."""

  def __init__(
      self,
      policy: policies.Policy,
      observation_config: observations.ObservationConfig,
      batch_size: int,
      console_delay: int = 0,
      batch_steps: int = 0,
      **agent_kwargs,
  ):
    self.observation_config = observation_config
    self._batch_size = batch_size
    self._batch_steps = batch_steps
    self._agent = build_basic_agent(
        policy=policy,
        batch_size=batch_size,
        **agent_kwargs)
    self.warmup = self._agent.warmup
    self.policy = policy

    self.delay = policy.delay - console_delay
    headroom = self.delay - (self.batch_steps - 1)
    if headroom < 0:
      raise ValueError(
          f'No headroom: '
          f'console delay ({console_delay}) must be <='
          f' policy delay ({policy.delay}) - batch_steps ({self.batch_steps}) + 1')
    elif headroom == 0:
      logging.warning('No headroom, agent will effectively run synchronously.')

    self._output_queue: utils.PeekableQueue[SampleOutputs[ControllerType]] \
      = utils.PeekableQueue()

    self.dummy_sample_outputs = policy.controller_head.dummy_sample_outputs([batch_size])
    for _ in range(self.delay):
      self._output_queue.put(self.dummy_sample_outputs)

    self.state_queue_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()
    self._state_queue = queue.Queue()
    self._worker_thread = None

    self.pop = self._output_queue.get
    self.peek_n = self._output_queue.peek_n

  @property
  def batch_steps(self) -> int:
    return self._batch_steps or 1

  @property
  def hidden_state(self):
    return self._agent.hidden_state

  @property
  def name_code(self):
    return self._agent.name_code

  def decode_controller(self, controller: ControllerType) -> Controller:
    return self.policy.controller_head.decode_controller(controller)

  def start(self):
    if self._worker_thread:
      raise RuntimeError('Already started.')

    self._worker_thread = threading.Thread(
        target=_run_agent, kwargs=dict(
            agent=self._agent,
            state_queue=self._state_queue,
            controller_queue=self._output_queue,
            batch_steps=self._batch_steps,
            state_queue_profiler=self.state_queue_profiler,
            step_profiler=self.step_profiler,
        ))
    self._worker_thread.start()

  def stop(self):
    if self._worker_thread:
      self._state_queue.put(None)
      self._worker_thread.join()
      self._worker_thread = None

  @contextlib.contextmanager
  def run(self):
    try:
      self.start()
      yield self
    finally:
      self.stop()

  def push(self, game: Game, needs_reset: np.ndarray):
    self._state_queue.put((game, needs_reset))

  def __del__(self):
    self.stop()

  def step(
      self,
      game: Game,
      needs_reset: np.ndarray
  ) -> SampleOutputs:
    self._state_queue.put((game, needs_reset))
    delayed_controller = self._output_queue.get()
    return delayed_controller

def get_name_code(state: dict, name: str) -> int:
  name_map: dict[str, int] = state['name_map']
  if name not in name_map:
    raise ValueError(f'Nametag must be one of {name_map.keys()}.')
  return name_map[name]

def get_agent_config(state: dict) -> tp.Optional[dict]:
  # TODO: unify self-train and train-two
  if 'rl_config' in state:  # self-train aka rl/run.py
    return state['rl_config']['agent']
  elif 'agent_config' in state:  # rl/train_two.py
    return state['agent_config']
  return None

def get_name_from_rl_state(state: dict) -> tp.Optional[list[str]]:
  # For RL, we know the name that was used during training.
  agent_config = get_agent_config(state)
  if agent_config is None:
    return None
  name = agent_config['name']
  return [name] if isinstance(name, str) else name

def build_delayed_agent(
    state: dict,
    console_delay: int,
    name: tp.Optional[tp.Union[str, list[str]]] = None,
    async_inference: bool = False,
    sample_temperature: float = 1.0,
    platform: tp.Optional[policies.Platform] = None,
    **agent_kwargs,
) -> tp.Union[DelayedAgent, AsyncDelayedAgent]:
  """Build an [Async]DelayedAgent given a state dict.

  Note: various agent parameters may be stored in the state dict. This function
  does the slightly tricky task of figuring out whether to use the parameters
  from the state dict or the function arguments.
  """
  imitation_config = saving.upgrade_config(state['config'], platform=platform)
  policy = saving.load_policy_from_state(state)

  rl_name = get_name_from_rl_state(state)

  if rl_name is not None:
    override = False

    if name is None:
      override = True
    elif isinstance(name, str) and name not in rl_name:
      logging.warning(f'Agent trained with name(s) "{rl_name}", got "{name}"')
      override = True
    elif isinstance(name, list):
      for n in name:
        if n not in rl_name:
          raise ValueError(f'Agent trained with name(s) {rl_name}, got "{n}"')
      logging.info('Requested agent name batch is valid.')

    if override:
      logging.info('Setting agent name to "%s" from RL', rl_name[0])
      name = rl_name[0]

  if name is None:
    # TODO: just pick from the name_map?
    raise ValueError('Must specify an agent name.')

  if isinstance(name, str):
    name_code = get_name_code(state, name)
  else:
    name_code = [get_name_code(state, n) for n in name]

  # Stick the observation_config into the agent for future use.
  observation_config = flag_utils.dataclass_from_dict(
          observations.ObservationConfig, imitation_config['observation'])

  agent_class = AsyncDelayedAgent if async_inference else DelayedAgent
  return agent_class(
      policy=policy,
      observation_config=observation_config,
      name_code=name_code,
      console_delay=console_delay,
      sample_kwargs=dict(temperature=sample_temperature),
      **agent_kwargs,
  )

class NameChangeMode(enum.Enum):
  FIXED = enum.auto()
  CYCLE = enum.auto()
  RANDOM = enum.auto()

class Agent:
  """Wraps a Policy to interact with Dolphin."""

  def __init__(
      self,
      state: dict,
      opponent_port: int,
      config: dict,  # use train.Config instead
      # Actual (netplay) port may different from Controller's (local) port
      port: tp.Optional[int] = None,
      controller: tp.Optional[melee.Controller] = None,
      name_change_mode: NameChangeMode = NameChangeMode.FIXED,
      mirror: bool = False,
      **agent_kwargs,
  ):
    self._controller = controller
    if port:
      self._port = port
    elif controller:
      self._port = controller.port
    else:
      raise ValueError('Must provide either controller or port.')

    self.players = (self._port, opponent_port)
    self.config = config
    self.name_change_mode = name_change_mode
    self.mirror = mirror

    self.name_map: dict[str, int] = state['name_map']
    rl_names = get_name_from_rl_state(state)
    if rl_names is not None:
      self.name_codes = [self.name_map[n] for n in rl_names]
    else:
      self.name_codes = list(set(self.name_map.values()))
    self.name_index = 0

    self._agent = build_delayed_agent(state, batch_size=1, **agent_kwargs)
    self._agent.warmup()
    # Forward async interface
    self.run = self._agent.run
    self.start = self._agent.start
    self.stop = self._agent.stop

    self._observation_filter = observations.build_observation_filter(
        self._agent.observation_config)

  def set_ports(self, port: int, opponent_port: int):
    self.players = (port, opponent_port)

  def set_controller(self, controller: melee.Controller):
    if controller.port != self._port:
      raise ValueError('Controller has wrong port.')
    self._controller = controller

  def update_name(self):
    if self.name_change_mode == NameChangeMode.FIXED:
      return
    elif self.name_change_mode == NameChangeMode.CYCLE:
      self.name_index = (self.name_index + 1) % len(self.name_codes)
    elif self.name_change_mode == NameChangeMode.RANDOM:
      self.name_index = np.random.randint(len(self.name_codes))
    self._agent._agent.set_name_code(self.name_codes[self.name_index])

  def step(self, gamestate: melee.GameState) -> SampleOutputs:
    new_game = gamestate.frame == -123
    if new_game:
      self.update_name()
      self._observation_filter.reset()
      self._parser = Parser(ports=self.players)

    needs_reset = np.array([new_game])
    game = self._parser.get_game(gamestate)
    if self.mirror:
      game = mirror_lib.mirror_game(game)
    game = self._observation_filter.filter(game)
    game = utils.map_single_structure(lambda x: np.expand_dims(x, 0), game)

    sample_outputs = self._agent.step(game, needs_reset)
    action = sample_outputs.controller_state
    # Note: x.item() can return the wrong dtype, e.g. int instead of uint8.
    action = utils.map_single_structure(lambda x: x[0], action)
    action = self._agent.policy.controller_head.decode_controller(action)
    if self.mirror:
      action = mirror_lib.mirror_controller(action)

    assert self._controller is not None
    send_controller(self._controller, action)
    return sample_outputs

def build_agent(
    opponent_port: int,
    name: str = nametags.DEFAULT_NAME,
    port: tp.Optional[int] = None,
    controller: tp.Optional[melee.Controller] = None,
    state: tp.Optional[dict] = None,
    path: tp.Optional[str] = None,
    **agent_kwargs,
) -> Agent:
  if state is None:
    if path is None:
      raise ValueError('Must provide either state or path.')

    state = saving.load_state_from_disk(path)

  return Agent(
      controller=controller,
      port=port,
      opponent_port=opponent_port,
      config=state['config'],
      # The rest are passed through to build_delayed_agent
      state=state,
      name=name,
      **agent_kwargs,
  )

BATCH_AGENT_FLAGS = dict(
    path=ff.String(None, 'Local path to pickled agent state.'),
    sample_temperature=ff.Float(1.0, 'Change sampling temperature at run-time.'),
    compile=ff.Boolean(True, 'Compile the sample function.'),
    name=ff.String(nametags.DEFAULT_NAME, 'Name of the agent.'),
    # arg to build_delayed_agent
    async_inference=ff.Boolean(False, 'run agent asynchronously'),
    fake=ff.Boolean(False, 'Use fake agents.'),
    batch_steps=ff.Integer(0, 'Number of steps to batch (in time)'),

    # Platform-specific flags
    # We only need to set the platform for old agents that don't have it saved.
    platform=ff.EnumClass(None, policies.Platform, 'Platform to use.'),
    tf=dict(
        jit_compile=ff.Boolean(False, 'Jit-compile the sample function.'),
        # Generally we want to set `run_on_cpu` once for all agents.
        # run_on_cpu=ff.Boolean(False, 'Run the agent on the CPU.'),
    ),
    jax=dict(
        seed=ff.Integer(0, 'Random seed for JAX agents.'),
    ),
)

AGENT_FLAGS = dict(
    BATCH_AGENT_FLAGS,
    name_change_mode=ff.EnumClass(
        NameChangeMode.FIXED, NameChangeMode,
        'How to change the agent name.'),
    mirror=ff.Boolean(False, 'Mirror the x axis.'),
)

PLAYER_FLAGS = dict(
    type=ff.Enum('ai', ('ai', 'human', 'cpu'), 'Player type.'),
    character=ff.EnumClass(
        melee.Character.FOX, melee.Character,
        'Character selected by agent or CPU.'),
    level=ff.Integer(9, 'CPU level.'),
    ai=AGENT_FLAGS,
)

BATCH_PLAYER_FLAGS = dict(
    PLAYER_FLAGS,
    ai=BATCH_AGENT_FLAGS,
)

def load_state(agent_kwargs: dict) -> dict:
  path = agent_kwargs.pop('path')
  if 'tag' in agent_kwargs:
    logging.warning('`tag` is no longer used, deleting from agent config')
    del agent_kwargs['tag']
  return saving.load_state_from_disk(path)

def get_player(
    type: str,
    character: melee.Character,
    level: int,
    **_,  # for convenience
) -> dolphin.Player:
  if type == 'ai':
    return dolphin.AI(character)
  elif type == 'human':
    return dolphin.Human()
  elif type == 'cpu':
    return dolphin.CPU(character, level)
  raise ValueError(f'Unknown player type: {type}')

def get_single_character(config: dict) -> tp.Optional[melee.Character]:
  allowed_characters = config['dataset']['allowed_characters']
  character_list = data.chars_from_string(allowed_characters)
  if character_list is None or len(character_list) != 1:
    return
  return character_list[0]


class AgentType(enum.Enum):
  IMITATION = enum.auto()
  RL = enum.auto()

@dataclasses.dataclass
class AgentSummary:
  type: AgentType
  delay: int
  characters: list[melee.Character] | None
  opponents: list[melee.Character] | None
  version: tuple[int, ...]

  @classmethod
  def from_checkpoint(cls, path: str) -> 'AgentSummary':
    combined_state = saving.load_state_from_disk(path)
    config = combined_state['config']
    characters = allowed_characters(config)

    if 'rl_config' in combined_state:
      agent_type = AgentType.RL
      agent_config = combined_state['rl_config']['agent']

      # Self-play RL can be multi-character
      rl_chars = agent_config.get('char')  # field was added more recently
      if rl_chars is not None:
        characters = [melee.Character(c) for c in rl_chars]
      opponents = characters
    elif 'agent_config' in combined_state:
      # train_two is always one character
      agent_type = AgentType.RL
      opponents = [data.name_to_character[combined_state['opponent']]]
    else:
      agent_type = AgentType.IMITATION
      # TODO: read opponents from config, but by default it is all chars
      opponents = None

    version_component = path.split('_')[-1]
    if version_component.startswith('v'):
      version = tuple(map(int, version_component[1:].split('.')))
    else:
      version = (0,)

    return cls(
        type=agent_type,
        delay=config['policy']['delay'],
        characters=characters,
        opponents=opponents,
        version=version,
    )

def get_imitation_agents(
    models_path: str,
    delay: tp.Optional[int],
) -> dict[melee.Character, str]:
  models = os.listdir(models_path)

  imitation_agents = {}
  for model in models:
    summary = AgentSummary.from_checkpoint(os.path.join(models_path, model))
    if not summary.type is AgentType.IMITATION:
      continue

    if summary.delay != delay:
      continue

    characters = summary.characters
    if characters is None:
      # TODO: add all legal characters?
      continue

    for character in characters:
      if character in imitation_agents:
        logging.warning(f'Got multiple imitation agents for {character}: {model}')
      imitation_agents[character] = model

  return imitation_agents

def build_matchup_table(
      models_path: str,
      delay: int,
) -> dict[melee.Character, dict[melee.Character, str]]:
  models = os.listdir(models_path)

  agent_summaries = {
      model: AgentSummary.from_checkpoint(os.path.join(models_path, model))
      for model in models
  }

  agent_summaries = {
      model: summary for model, summary in agent_summaries.items()
      if summary.delay == delay
  }

  table: dict[melee.Character, dict[melee.Character, str]] = {}
  imitation_agents: dict[melee.Character, str] = {}

  for model, summary in agent_summaries.items():
    if summary.type is AgentType.IMITATION:
      for char in summary.characters or []:
        imitation_agents[char] = model
    else:
      if summary.characters is None:
        # TODO: use all legal characters
        continue

      for char in summary.characters:
        opponent_table = table.setdefault(char, {})
        if summary.opponents is None:
          # TODO: use all legal characters
          continue

        for opponent in summary.opponents:
          if opponent in opponent_table:
            existing_model = opponent_table[opponent]
            logging.warning(f'Got multiple agents for {char} vs {opponent}: {existing_model}, {model}')
          opponent_table[opponent] = model

  # Default to RL, then imitation.
  default_agents = imitation_agents.copy()
  for character, opponent_table in table.items():
    # First try ditto agent, otherwise any RL agent.
    if character in opponent_table:
      default_agents[character] = opponent_table[character]
    else:
      default_agents[character] = next(iter(opponent_table.values()))

  for character, model in default_agents.items():
    opponent_table = table.setdefault(character, {})
    for opponent in melee.Character:
      opponent_table.setdefault(opponent, model)

  return table

class EnsembleAgent:

  def __init__(
      self,
      character: melee.Character,
      models_path: str,
      opponent_port: int,
      delay: int,
      **agent_kwargs,
  ):
    self._models_path = models_path
    self.opponent_table = build_matchup_table(models_path, delay)[character]

    self.opponent_port = opponent_port
    self._agent_kwargs = agent_kwargs.copy()
    self._agent_kwargs['opponent_port'] = opponent_port

    self.current_model: tp.Optional[str] = None
    self._agent: tp.Optional[Agent] = None

  def set_ports(self, port, opponent_port):
    if self._agent is not None:
      self._agent.players = (port, opponent_port)
    self.opponent_port = opponent_port
    self._agent_kwargs.update(
        port=port, opponent_port=opponent_port,
    )

  def _get_agent(self, model: str) -> Agent:
    if model == self.current_model:
      return self._agent

    if self._agent:
      self._agent.stop()

    logging.info(f'Setting auto-model to {model}')

    path = os.path.join(self._models_path, model)
    agent_kwargs = dict(self._agent_kwargs, path=path)
    self._agent = build_agent(**agent_kwargs)
    self._agent.start()
    self.current_model = model
    return self._agent

  def step(self, gamestate: melee.GameState) -> SampleOutputs:
    opponent = gamestate.players[self.opponent_port].character
    model = self.opponent_table[opponent]
    agent = self._get_agent(model)
    return agent.step(gamestate)

  def start(self):
    pass

  def stop(self):
    if self._agent:
      self._agent.stop()

def allowed_characters(config: dict) -> tp.Optional[list[melee.Character]]:
  return data.chars_from_string(config['dataset']['allowed_characters'])

def update_character(player: dolphin.AI, config: dict):
  character_list = allowed_characters(config)
  if character_list is None or player.character in character_list:
    return

  if len(character_list) == 1:
    # If there's only one option, then go with that.
    player.character = character_list[0]
    print('Setting character to:', player.character.name)
  else:
    # Could use character_list[0] here, but that might lead to silently never
    # picking the other options.
    raise ValueError(f"Character must be one of {character_list}")
