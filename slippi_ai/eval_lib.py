import contextlib
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
from slippi_ai.controller_heads import SampleOutputs
from slippi_db.parse_libmelee import get_game

def disable_gpus():
  tf.config.set_visible_devices([], 'GPU')


Sample = Callable[
    [embed.StateAction, policies.RecurrentState],
    Tuple[embed.Action, policies.RecurrentState]]

def dummy_sample_outputs(
    controller_embedding: embed.Embedding[tp.Any, embed.Action],
    shape: tp.Sequence[int],
):
  return SampleOutputs(
      controller_state=controller_embedding.dummy(shape),
      logits=controller_embedding.dummy_embedding(shape),
  )


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
    self._name_code = embed.NAME_DTYPE(name_code)
    self._embed_controller = policy.controller_embedding
    self._batch_size = batch_size

    # The controller_head may discretize certain components of the action.
    # Agents only work with the discretized action space; you will need
    # to call `decode` on the action before sending it to Dolphin.
    default_controller = self._embed_controller.dummy([batch_size])
    self._prev_controller = default_controller

    initial_state = policy.initial_state(batch_size)

    def sample(
        state_action: embed.StateAction,
        prev_state: policies.RecurrentState,
        needs_reset: tf.Tensor,
    ) -> tuple[SampleOutputs, policies.RecurrentState]:
      prev_state = tf.nest.map_structure(
          lambda x, y: tf_utils.where(needs_reset, x, y),
          initial_state, prev_state)
      return policy.sample(state_action, prev_state, **sample_kwargs)

    def multi_sample(
        states: list[tuple[embed.Game, tf.Tensor]],  # time-indexed
        prev_action: embed.Action,  # only for first step
        initial_state: policies.RecurrentState,
    ) -> Tuple[list[SampleOutputs], policies.RecurrentState]:
      actions: list[SampleOutputs] = []
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
        prev_action = next_action.controller_state

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
  ) -> SampleOutputs:
    """Doesn't take into account delay."""
    state_action = embed.StateAction(
        state=game,
        action=self._prev_controller,
        name=np.full([self._batch_size], self._name_code),
    )
    # TODO: technically we should be calling `from_state` on the game here.
    # We don't call `policy.embed_state_action.from_state` on the entire
    # `state_action` tuple because the action is already encoded. This isn't
    # an issue for now because the game encoding is a no-op, but this might
    # not be the case in the future.

    # Sample an action.
    sample_outputs: SampleOutputs
    sample_outputs, self.hidden_state = self._sample(
        state_action, self.hidden_state, needs_reset)
    sample_outputs = utils.map_single_structure(
        lambda t: t.numpy(), sample_outputs)

    self._prev_controller = sample_outputs.controller_state
    return sample_outputs

  def multi_step(
      self,
      states: list[tuple[embed.Game, np.ndarray]],
  ) -> list[SampleOutputs]:
    sample_outputs: list[SampleOutputs]
    sample_outputs, self.hidden_state = self._multi_sample(
        states, self._prev_controller, self.hidden_state)
    sample_outputs = utils.map_single_structure(
        lambda t: t.numpy(), sample_outputs)

    self._prev_controller = sample_outputs[-1].controller_state
    return sample_outputs

  def step_unbatched(
      self,
      game: embed.Game,
      needs_reset: bool
  ) -> SampleOutputs:
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
    self.embed_controller = policy.controller_embedding

    self.delay = policy.delay - console_delay
    self._output_queue: utils.PeekableQueue[SampleOutputs] \
      = utils.PeekableQueue()

    self.dummy_sample_outputs = dummy_sample_outputs(
        self.embed_controller, [batch_size])
    for _ in range(self.delay):
      self._output_queue.put(self.dummy_sample_outputs)

    # Break circular references.
    self.pop = self._output_queue.get
    self.peek_n = self._output_queue.peek_n

    # TODO: put this in the BasicAgent?
    self.step_profiler = utils.Profiler(burnin=1)

  @property
  def batch_steps(self) -> int:
    return 1

  def step(
      self,
      game: embed.Game,
      needs_reset: np.ndarray
  ) -> SampleOutputs:
    """Synchronous agent step."""
    self.push(game, needs_reset)
    delayed_controller = self.pop()
    return delayed_controller

  # Present the same interface as the async agent.
  def push(self, game: embed.Game, needs_reset: np.ndarray):
    with self.step_profiler:
      sampled_controller = self._agent.step(game, needs_reset)
    self._output_queue.put(sampled_controller)

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
      batch_steps: int = 0,
      **agent_kwargs,
  ):
    self._batch_size = batch_size
    self._batch_steps = batch_steps
    self._agent = BasicAgent(
        policy=policy,
        batch_size=batch_size,
        **agent_kwargs)
    self._policy = policy
    self.embed_controller = policy.controller_embedding

    self.delay = policy.delay - console_delay
    self._output_queue: utils.PeekableQueue[SampleOutputs] \
      = utils.PeekableQueue()

    self.dummy_sample_outputs = dummy_sample_outputs(
        self.embed_controller, [batch_size])
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
  ) -> SampleOutputs:
    self._state_queue.put((game, needs_reset))
    delayed_controller = self._output_queue.get()
    return delayed_controller

def load_state(path: Optional[str] = None, tag: Optional[str] = None) -> dict:
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
    self.players = (self._port, opponent_port)
    self.config = config

    self._agent = build_delayed_agent(batch_size=1, **agent_kwargs)
    # Forward async interface
    self.run = self._agent.run
    self.start = self._agent.start
    self.stop = self._agent.stop

  def step(self, gamestate: melee.GameState) -> embed.StateAction:
    needs_reset = np.array([gamestate.frame == -123])
    game = get_game(gamestate, ports=self.players)
    game = utils.map_nt(lambda x: np.expand_dims(x, 0), game)
    # TODO: use the policy's game embedding
    game = embed.default_embed_game.from_state(game)

    sample_outputs = self._agent.step(game, needs_reset)
    action = sample_outputs.controller_state
    # Note: x.item() can return the wrong dtype, e.g. int instead of uint8.
    action = utils.map_nt(lambda x: x[0], action)
    action = self._agent.embed_controller.decode(action)
    send_controller(self._controller, action)
    return sample_outputs


def build_agent(
    controller: melee.Controller,
    opponent_port: int,
    name: str,
    state: Optional[dict] = None,
    path: Optional[str] = None,
    tag: Optional[str] = None,
    **agent_kwargs,
) -> Agent:
  if state is None:
    state = load_state(path, tag)

  return Agent(
      controller=controller,
      opponent_port=opponent_port,
      config=state['config'],
      # The rest are passed through to build_delayed_agent
      state=state,
      name=name,
      **agent_kwargs,
  )

AGENT_FLAGS = dict(
    path=ff.String(None, 'Local path to pickled agent state.'),
    tag=ff.String(None, 'Tag used to save state in s3.'),
    sample_temperature=ff.Float(1.0, 'Change sampling temperature at run-time.'),
    compile=ff.Boolean(True, 'Compile the sample function.'),
    name=ff.String('Master Player', 'Name of the agent.'),
    # arg to build_delayed_agent
    async_inference=ff.Boolean(False, 'run agent asynchronously'),
    # Generally we want to set `run_on_cpu` once for all agents.
    # run_on_cpu=ff.Boolean(False, 'Run the agent on the CPU.'),
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
