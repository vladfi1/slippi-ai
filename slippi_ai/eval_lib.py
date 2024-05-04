import contextlib
import dataclasses
import threading, queue
from typing import Callable, Optional, Tuple
import typing as tp

import numpy as np
import fancyflags as ff
import tensorflow as tf

import melee

from slippi_ai import (
  embed, policies, dolphin, saving, data, utils, tf_utils
)
from slippi_ai.controller_lib import send_controller
from slippi_db.parse_libmelee import get_game

def disable_gpus():
  tf.config.set_visible_devices([], 'GPU')


Sample = Callable[
    [embed.StateAction, policies.RecurrentState],
    Tuple[embed.Action, policies.RecurrentState]]


class BasicAgent:
  """Wraps a Policy to track hidden state."""

  def __init__(
      self,
      policy: policies.Policy,
      batch_size: int,
      name_code: int,
      sample_kwargs: dict = {},
      compile: bool = True,
      run_on_cpu: bool = False,
  ):
    self._policy = policy
    self._name_code = name_code
    self._embed_controller = policy.controller_embedding
    self._batch_size = batch_size

    default_controller = self._embed_controller.decode(
        self._embed_controller.dummy([batch_size]))
    self._prev_controller = default_controller

    initial_state = policy.initial_state(batch_size)

    def sample(
        state_action: embed.StateAction,
        prev_state: policies.RecurrentState,
        needs_reset: tf.Tensor,
    ) -> tuple[embed.Action, policies.RecurrentState]:
      prev_state = tf.nest.map_structure(
          lambda x, y: tf_utils.where(needs_reset, x, y),
          initial_state, prev_state)
      return policy.sample(state_action, prev_state, **sample_kwargs)

    def multi_sample(
        states: list[tuple[embed.Game, tf.Tensor]],  # time-indexed
        prev_action: embed.Action,  # only for first step
        initial_state: policies.RecurrentState,
    ) -> Tuple[list[embed.Action], policies.RecurrentState]:
      actions = []
      hidden_state = initial_state
      for game, needs_reset in states:
        state_action = embed.StateAction(
            state=game,
            action=prev_action,
            name=tf.fill([self._batch_size], self._name_code),
        )
        next_action, hidden_state = sample(
            state_action, hidden_state, needs_reset)
        actions.append(next_action)
        prev_action = next_action

      return actions, hidden_state

    if run_on_cpu:
      sample = tf_utils.run_on_cpu(sample)
      multi_sample = tf_utils.run_on_cpu(multi_sample)
    if compile:
      sample = tf.function(sample)
      multi_sample = tf.function(multi_sample)

    self._sample = sample
    self._multi_sample = multi_sample

    self.hidden_state = self._policy.initial_state(batch_size)

  def step(
      self,
      game: embed.Game,
      needs_reset: np.ndarray
  ) -> embed.Action:
    """Doesn't take into account delay."""
    state_action = embed.StateAction(
        state=game,
        action=self._prev_controller,
        name=np.full([self._batch_size], self._name_code),
    )
    # `from_state` discretizes certain components of the action
    state_action = self._policy.embed_state_action.from_state(state_action)

    # Sample an action.
    action: embed.Action
    action, self.hidden_state = self._sample(
        state_action, self.hidden_state, needs_reset)
    action = utils.map_single_structure(lambda t: t.numpy(), action)

    # decode un-discretizes the discretized components (x/y axis and shoulder)
    sampled_controller = self._embed_controller.decode(action)
    self._prev_controller = sampled_controller
    return sampled_controller

  def multi_step(
      self,
      states: list[tuple[embed.Game, np.ndarray]],
  ) -> list[embed.Action]:
    prev_controller = self._policy.controller_embedding.from_state(
        self._prev_controller)

    actions, self.hidden_state = self._multi_sample(
        states, prev_controller, self.hidden_state)
    actions = utils.map_single_structure(lambda t: t.numpy(), actions)

    sampled_controllers = [
        self._policy.controller_embedding.decode(a) for a in actions]
    self._prev_controller = sampled_controllers[-1]
    return sampled_controllers

  def step_unbatched(
      self,
      game: embed.Game,
      needs_reset: bool
  ) -> embed.Action:
    assert self._batch_size == 1
    batched_game = tf.nest.map_structure(
        lambda x: tf.expand_dims(x, 0), game)
    batched_needs_reset = np.array([needs_reset])
    batched_action = self.step(batched_game, batched_needs_reset)
    return tf.nest.map_structure(lambda x: x.item(), batched_action)

class DelayedAgent:
  """Wraps a BasicAgent with delay."""

  def __init__(
      self,
      policy: policies.Policy,
      batch_size: int,
      console_delay: int = 0,
      batch_steps: int = 0,
      **agent_kwargs,
  ):
    if batch_steps != 0:
      raise NotImplementedError('batch_steps not supported for DelayedAgent.')
    del batch_steps

    self._agent = BasicAgent(
        policy=policy,
        batch_size=batch_size,
        **agent_kwargs)
    self._policy = policy
    self._embed_controller = policy.controller_embedding

    self.delay = policy.delay - console_delay
    self._controller_queue = utils.PeekableQueue()
    self.default_controller = self._embed_controller.decode(
        self._embed_controller.dummy([batch_size]))
    for _ in range(self.delay):
      self._controller_queue.put(self.default_controller)

    # Break circular references.
    self.pop = self._controller_queue.get
    self.peek_n = self._controller_queue.peek_n

    # TODO: put this in the BasicAgent?
    self.step_profiler = utils.Profiler(burnin=1)

  @property
  def batch_steps(self) -> int:
    return 1

  def step(
      self,
      game: embed.Game,
      needs_reset: np.ndarray
  ) -> embed.Action:
    """Synchronous agent step."""
    self.push(game, needs_reset)
    # Return the action actually taken, in decoded form.
    delayed_controller = self.pop()
    return delayed_controller

  # Present the same interface as the async agent.
  def push(self, game: embed.Game, needs_reset: np.ndarray):
    with self.step_profiler:
      sampled_controller = self._agent.step(game, needs_reset)
    self._controller_queue.put(sampled_controller)

  @contextlib.contextmanager
  def run(self):
    """For compatibility with the async agent."""
    yield self

  def start(self):
    """For compatibility with the async agent."""

  def stop(self):
    """For compatibility with the async agent."""


def _run_agent(
    agent: BasicAgent,
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
      next_item: Optional[tuple[embed.Game, bool]] = state_queue.get()
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
        next_item: Optional[tuple[embed.Game, bool]] = state_queue.get()
      if next_item is None:
        state_queue.task_done()
        return
      states.append(next_item)

    with step_profiler:
      sampled_controllers = agent.multi_step(states)

    for controller in sampled_controllers:
      controller_queue.put(controller)
      state_queue.task_done()

class AsyncDelayedAgent:
  """Delayed agent that runs inference asynchronously."""

  def __init__(
      self,
      policy: policies.Policy,
      batch_size: int,
      console_delay: int = 0,
      batch_steps: int = 1,
      **agent_kwargs,
  ):
    self._batch_size = batch_size
    self._batch_steps = batch_steps
    self._agent = BasicAgent(
        policy=policy,
        batch_size=batch_size,
        **agent_kwargs)
    self._policy = policy
    self._embed_controller = policy.controller_embedding

    self.delay = policy.delay - console_delay
    self._controller_queue: utils.PeekableQueue[embed.Action] \
      = utils.PeekableQueue()
    self.default_controller = self._embed_controller.decode(
        self._embed_controller.dummy([batch_size]))
    for _ in range(self.delay):
      self._controller_queue.put(self.default_controller)

    self.state_queue_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()
    self._state_queue = queue.Queue()
    self._worker_thread = None

    self.pop = self._controller_queue.get
    self.peek_n = self._controller_queue.peek_n

  @property
  def batch_steps(self) -> int:
    return self._batch_steps or 1

  def start(self):
    if self._worker_thread:
      raise RuntimeError('Already started.')

    self._worker_thread = threading.Thread(
        target=_run_agent, kwargs=dict(
            agent=self._agent,
            state_queue=self._state_queue,
            controller_queue=self._controller_queue,
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
      print('actor run exit')
      self.stop()

  def push(self, game: embed.Game, needs_reset: np.ndarray):
    self._state_queue.put((game, needs_reset))

  def __del__(self):
    print('actor del')
    self.stop()

  def step(
      self,
      game: embed.Game,
      needs_reset: np.ndarray
  ) -> embed.Action:
    self._state_queue.put((game, needs_reset))

    # Return the action actually taken, in decoded form.
    delayed_controller = self._controller_queue.get()
    return delayed_controller

class Agent:
  """Wraps a Policy to interact with Dolphin."""

  def __init__(
      self,
      controller: melee.Controller,
      opponent_port: int,
      config: dict,  # use train.Config instead
      **agent_kwargs,
  ):
    self._controller = controller
    self._port = controller.port
    self._players = (self._port, opponent_port)
    self.config = config

    self._agent = DelayedAgent(batch_size=1, **agent_kwargs)

  def step(self, gamestate: melee.GameState) -> embed.StateAction:
    needs_reset = np.array([gamestate.frame == -123])
    game = get_game(gamestate, ports=self._players)
    game = utils.map_nt(lambda x: np.expand_dims(x, 0), game)

    action = self._agent.step(game, needs_reset)
    action = utils.map_nt(lambda x: x.item(), action)
    send_controller(self._controller, action)
    return action

AGENT_FLAGS = dict(
    path=ff.String(None, 'Local path to pickled agent state.'),
    tag=ff.String(None, 'Tag used to save state in s3.'),
    sample_temperature=ff.Float(1.0, 'Change sampling temperature at run-time.'),
    compile=ff.Boolean(True, 'Compile the sample function.'),
    name=ff.String('Master Player', 'Name of the agent.'),
    # Generally we want to set `run_on_cpu` once for all agents.
    # run_on_cpu=ff.Boolean(False, 'Run the agent on the CPU.'),
)

def load_state(path: Optional[str], tag: Optional[str]) -> dict:
  if path:
    return saving.load_state_from_disk(path)
  elif tag:
    return saving.load_state_from_s3(tag)
  else:
    raise ValueError('Must specify one of "tag" or "path".')

def get_name_code(state: dict, name: str) -> int:
  name_map: dict[str, int] = state['name_map']
  if name not in name_map:
    raise ValueError(f'Nametag must be one of {name_map.keys()}.')
  return name_map[name]

def build_delayed_agent(
    state: dict,
    name: str,
    async_inference: bool = False,
    sample_temperature: float = 1.0,
    console_delay: int = 0,
    **agent_kwargs,
) -> tp.Union[DelayedAgent, AsyncDelayedAgent]:
  policy = saving.load_policy_from_state(state)
  agent_class = AsyncDelayedAgent if async_inference else DelayedAgent
  return agent_class(
      policy=policy,
      name_code=get_name_code(state, name),
      console_delay=console_delay,
      sample_kwargs=dict(temperature=sample_temperature),
      **agent_kwargs,
  )

def build_agent(
    controller: melee.Controller,
    opponent_port: int,
    name: str,
    state: Optional[dict] = None,
    path: Optional[str] = None,
    tag: Optional[str] = None,
    sample_temperature: float = 1.0,
    console_delay: int = 0,
    **agent_kwargs,
) -> Agent:
  if state is None:
    state = load_state(path, tag)
  policy = saving.load_policy_from_state(state)

  return Agent(
      controller=controller,
      opponent_port=opponent_port,
      policy=policy,
      config=state['config'],
      console_delay=console_delay,
      name_code=get_name_code(state, name),
      sample_kwargs=dict(temperature=sample_temperature),
      **agent_kwargs,
  )

# TODO: move the dolphin-related stuff to dolphin.py

@dataclasses.dataclass
class DolphinConfig:
  """Configure dolphin for evaluation."""
  path: Optional[str] = None  # Path to folder containing the dolphin executable
  iso: Optional[str] = None  # Path to melee 1.02 iso.
  stage: melee.Stage = melee.Stage.RANDOM_STAGE  # Which stage to play on.
  online_delay: int = 0  # Simulate online delay.
  blocking_input: bool = True  # Have game wait for AIs to send inputs.
  slippi_port: int = 51441  # Local ip port to communicate with dolphin.
  render: bool = True  # Render frames. Only disable if using vladfi1\'s slippi fork.
  save_replays: bool = False  # Save slippi replays to the usual location.
  replay_dir: Optional[str] = None  # Directory to save replays to.
  headless: bool = True  # Headless configuration: exi + ffw, no graphics or audio.
  infinite_time: bool = True  # Infinite time no stocks.

DOLPHIN_FLAGS = dict(
    path=ff.String(None, 'Path to folder containing the dolphin executable.'),
    iso=ff.String(None, 'Path to melee 1.02 iso.'),
    stage=ff.EnumClass(melee.Stage.RANDOM_STAGE, melee.Stage, 'Which stage to play on.'),
    online_delay=ff.Integer(0, 'Simulate online delay.'),
    blocking_input=ff.Boolean(True, 'Have game wait for AIs to send inputs.'),
    slippi_port=ff.Integer(51441, 'Local ip port to communicate with dolphin.'),
    fullscreen=ff.Boolean(False, 'Run dolphin in full screen mode.'),
    render=ff.Boolean(True, 'Render frames. Only disable if using vladfi1\'s slippi fork.'),
    save_replays=ff.Boolean(False, 'Save slippi replays to the usual location.'),
    replay_dir=ff.String(None, 'Directory to save replays to.'),
    headless=ff.Boolean(
        False, 'Headless configuration: exi + ffw, no graphics or audio.'),
    infinite_time=ff.Boolean(False, 'Infinite time no stocks.'),
)

PLAYER_FLAGS = dict(
    type=ff.Enum('ai', ('ai', 'human', 'cpu'), 'Player type.'),
    character=ff.EnumClass(
        melee.Character.FOX, melee.Character,
        'Character selected by agent or CPU.'),
    level=ff.Integer(9, 'CPU level.'),
    ai=AGENT_FLAGS,
)

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


def update_character(player: dolphin.AI, config: dict):
  allowed_characters = config['dataset']['allowed_characters']
  character_list = data.chars_from_string(allowed_characters)
  if character_list is None or player.character in character_list:
    return

  if len(character_list) == 1:
    # If there's only one option, then go with that.
    player.character = character_list[0]
    print('Setting character to', player.character)
  else:
    # Could use character_list[0] here, but that might lead to silently never
    # picking the other options.
    raise ValueError(f"Character must be one of {character_list}")
