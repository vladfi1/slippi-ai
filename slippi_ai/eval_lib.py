from collections import deque
import multiprocessing as mp
from typing import Callable, Mapping, Optional, Tuple

import fancyflags as ff
import tensorflow as tf

import melee

from slippi_ai import embed, policies, dolphin, saving
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

class Agent:
  """Wraps a Policy to track hidden state."""

  def __init__(
      self,
      controller: melee.Controller,
      opponent_port: int,
      policy: policies.Policy,
      config: dict,  # use train.Config instead
      embed_controller: embed.StructEmbedding[Controller] = embed.embed_controller_discrete,
      console_delay: int = 0,
      sample_kwargs: dict = {},
  ):
    self._controller = controller
    self._port = controller.port
    self._players = (self._port, opponent_port)
    self._policy = policy
    self.config = config
    self._embed_controller = embed_controller

    delay = policy.delay - console_delay
    self._controller_queue = deque(maxlen=delay+1)
    default_controller = embed_controller.decode(embed_controller.dummy([]))
    self._controller_queue.extend([default_controller] * delay)
    self._prev_controller = default_controller

    def sample_unbatched(state_action, prev_state):
      batched_state_action = tf.nest.map_structure(
          lambda x: tf.expand_dims(x, 0), state_action)
      batched_action, next_state = policy.sample(
          batched_state_action, prev_state, **sample_kwargs)
      action = tf.nest.map_structure(
          lambda x: tf.squeeze(x, 0), batched_action)
      return action, next_state

    self._sample = tf.function(sample_unbatched)
    # self._sample = sample_unbatched
    self.reset_state()

  def reset_state(self):
    self._hidden_state = self._policy.initial_state(1)

  def step(self, gamestate: melee.GameState) -> embed.StateAction:
    if gamestate.frame == -123:
      self.reset_state()

    game = get_game(gamestate, ports=self._players)

    state_action = embed.StateAction(
        state=game,
        action=self._prev_controller,
    )
    # `from_state` discretizes certain components of the action
    state_action = self._policy.embed_state_action.from_state(state_action)

    # Sample an action.
    action: embed.Action
    action, self._hidden_state = self._sample(
        state_action, self._hidden_state)
    action = tf.nest.map_structure(lambda t: t.numpy(), action)

    # decode un-discretizes the discretized components (x/y axis and shoulder)
    sampled_controller = self._embed_controller.decode(action)
    self._prev_controller = sampled_controller

    # Push the action into the queue and pop the current action.
    self._controller_queue.appendleft(sampled_controller)
    delayed_controller = self._controller_queue.pop()
    send_controller(self._controller, delayed_controller)

    # Return the action actually taken, in decoded form.
    return embed.StateAction(game, delayed_controller)


AGENT_FLAGS = dict(
    path=ff.String(None, 'Local path to pickled agent state.'),
    tag=ff.String(None, 'Tag used to save state in s3.'),
    sample_temperature=ff.Float(1.0, 'Change sampling temperature at run-time.'),
)

def build_agent(
    controller: melee.Controller,
    opponent_port: int,
    path: Optional[str] = None,
    tag: Optional[str] = None,
    sample_temperature: float = 1.0,
    console_delay: int = 0,
) -> Agent:
  if path:
    state = saving.load_state_from_disk(path)
  elif tag:
    state = saving.load_state_from_s3(tag)
  else:
    raise ValueError('Must specify one of "tag" or "path".')

  policy = saving.load_policy_from_state(state)

  return Agent(
      controller=controller,
      opponent_port=opponent_port,
      policy=policy,
      config=state['config'],
      console_delay=console_delay,
      sample_kwargs=dict(temperature=sample_temperature),
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

name_to_character = {
    char.name.lower(): char for char in melee.Character
}
