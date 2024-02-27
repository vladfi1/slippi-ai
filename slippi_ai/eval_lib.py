from collections import deque
from typing import Callable, Optional, Tuple

import numpy as np
import fancyflags as ff
import tensorflow as tf

import melee

from slippi_ai import embed, policies, dolphin, saving, data, utils
from slippi_ai.types import Controller
from slippi_db.parse_libmelee import get_game

def disable_gpus():
  tf.config.set_visible_devices([], 'GPU')


Sample = Callable[
    [embed.StateAction, policies.RecurrentState],
    Tuple[embed.Action, policies.RecurrentState]]

def send_controller(controller: melee.Controller, controller_state: Controller):
  for b in embed.LEGAL_BUTTONS:
    if getattr(controller_state.buttons, b.value):
      controller.press_button(b)
    else:
      controller.release_button(b)
  main_stick = controller_state.main_stick
  controller.tilt_analog(melee.Button.BUTTON_MAIN, main_stick.x, main_stick.y)
  c_stick = controller_state.c_stick
  controller.tilt_analog(melee.Button.BUTTON_C, c_stick.x, c_stick.y)
  controller.press_shoulder(melee.Button.BUTTON_L, controller_state.shoulder)


class RawAgent:
  """Wraps a Policy to track hidden state."""

  def __init__(
      self,
      policy: policies.Policy,
      batch_size: int,
      console_delay: int = 0,
      sample_kwargs: dict = {},
      compile: bool = True,
  ):
    self._policy = policy
    self._embed_controller = policy.controller_embedding
    self._batch_size = batch_size

    delay = policy.delay - console_delay
    self._controller_queue = deque(maxlen=delay+1)
    default_controller = self._embed_controller.decode(
        self._embed_controller.dummy([batch_size]))
    self._controller_queue.extend([default_controller] * delay)
    self._prev_controller = default_controller

    initial_state = policy.initial_state(batch_size)

    def sample(
        state_action: embed.StateAction,
        prev_state: policies.RecurrentState,
        needs_reset: tf.Tensor,
    ) -> tuple[embed.Action, policies.RecurrentState]:
      prev_state = tf.nest.map_structure(
          lambda x, y: utils.where(needs_reset, x, y),
          initial_state, prev_state)
      return policy.sample(state_action, prev_state, **sample_kwargs)

    if compile:
      sample = tf.function(sample)
    self._sample = sample

    self._hidden_state = self._policy.initial_state(batch_size)

  def step(
      self,
      game: embed.Game,
      needs_reset: np.ndarray
  ) -> embed.StateAction:
    state_action = embed.StateAction(
        state=game,
        action=self._prev_controller,
    )
    # `from_state` discretizes certain components of the action
    state_action = self._policy.embed_state_action.from_state(state_action)

    # Sample an action.
    action: embed.Action
    action, self._hidden_state = self._sample(
        state_action, self._hidden_state, needs_reset)
    action = tf.nest.map_structure(lambda t: t.numpy(), action)

    # decode un-discretizes the discretized components (x/y axis and shoulder)
    sampled_controller = self._embed_controller.decode(action)
    self._prev_controller = sampled_controller

    # Push the action into the queue and pop the current action.
    self._controller_queue.appendleft(sampled_controller)
    delayed_controller = self._controller_queue.pop()

    # Return the action actually taken, in decoded form.
    return delayed_controller

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

    self._agent = RawAgent(batch_size=1, **agent_kwargs)

  def step(self, gamestate: melee.GameState) -> embed.StateAction:
    needs_reset = np.array([gamestate.frame == -123])
    game = get_game(gamestate, ports=self._players)
    game = utils.map_nt(lambda x: np.expand_dims(x, 0), game)

    action = self._agent.step(game, needs_reset)
    action = utils.map_nt(lambda x: x.item(), action)
    send_controller(self._controller, action)
    return embed.StateAction(state=game, action=action)

AGENT_FLAGS = dict(
    path=ff.String(None, 'Local path to pickled agent state.'),
    tag=ff.String(None, 'Tag used to save state in s3.'),
    sample_temperature=ff.Float(1.0, 'Change sampling temperature at run-time.'),
    compile=ff.Boolean(True, 'Compile the sample function.'),
)

def load_state(path: Optional[str], tag: Optional[str]) -> dict:
  if path:
    return saving.load_state_from_disk(path)
  elif tag:
    return saving.load_state_from_s3(tag)
  else:
    raise ValueError('Must specify one of "tag" or "path".')

def build_raw_agent(
    state: dict,
    sample_temperature: float = 1.0,
    console_delay: int = 0,
    **agent_kwargs,
) -> Agent:
  policy = saving.load_policy_from_state(state)
  return RawAgent(
      policy=policy,
      console_delay=console_delay,
      sample_kwargs=dict(temperature=sample_temperature),
      **agent_kwargs,
  )

def build_agent(
    controller: melee.Controller,
    opponent_port: int,
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
      sample_kwargs=dict(temperature=sample_temperature),
      **agent_kwargs,
  )

DOLPHIN_FLAGS = dict(
    path=ff.String(None, 'Path to folder containing the dolphin executable.'),
    iso=ff.String(None, 'Path to melee 1.02 iso.'),
    stage=ff.EnumClass(melee.Stage.FINAL_DESTINATION, melee.Stage, 'Which stage to play on.'),
    online_delay=ff.Integer(0, 'Simulate online delay.'),
    blocking_input=ff.Boolean(True, 'Have game wait for AIs to send inputs.'),
    slippi_port=ff.Integer(51441, 'Local ip port to communicate with dolphin.'),
    fullscreen=ff.Boolean(False, 'Run dolphin in full screen mode.'),
    render=ff.Boolean(True, 'Render frames. Only disable if using vladfi1\'s slippi fork.'),
    save_replays=ff.Boolean(False, 'Save slippi replays to the usual location.'),
    replay_dir=ff.String(None, 'Directory to save replays to.'),
    headless=ff.Boolean(
        False, 'Headless configuration: exi + ffw, no graphics or audio.'),
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
