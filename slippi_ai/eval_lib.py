from collections import deque
import multiprocessing as mp
from typing import Callable, Mapping, Optional, Tuple

import fancyflags as ff
import tensorflow as tf

import melee

from slippi_ai import embed, policies, dolphin, saving
from slippi_ai.types import Controller, Game
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
      embed_controller: embed.StructEmbedding[Controller] = embed.embed_controller_discrete,
      sample_kwargs: dict = {},
  ):
    self._controller = controller
    self._port = controller.port
    self._players = (self._port, opponent_port)
    self._policy = policy
    self._embed_controller = embed_controller

    self._action_queue = deque(maxlen=policy.delay+1)
    default_action = self._embed_controller.dummy([])
    self._action_queue.extend([default_action] * policy.delay)
    self._prev_controller = embed_controller.decode(default_action)

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
    self._hidden_state = policy.initial_state(1)

  def step(self, gamestate: melee.GameState):
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

    # Push the action into the queue and pop the current action.
    self._action_queue.append(action)
    action = self._action_queue.popleft()

    # decode un-discretizes the discretized components (x/y axis and shoulder)
    sampled_controller = self._embed_controller.decode(action)
    send_controller(self._controller, sampled_controller)
    self._prev_controller = sampled_controller

    return action


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
) -> Agent:
  if path:
    policy = saving.load_policy_from_disk(path)
  elif tag:
    policy = saving.load_policy_from_s3(tag)
  else:
    raise ValueError('Must specify one of "tag" or "path".')

  return Agent(
      controller=controller,
      opponent_port=opponent_port,
      policy=policy,
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


class Environment(dolphin.Dolphin):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    assert len(self._players) == 2

    self._player_to_ports = {}
    ports = list(self._players)

    for port, opponent_port in zip(ports, reversed(ports)):
      if isinstance(self._players[port], dolphin.AI):
        self._player_to_ports[port] = (port, opponent_port)

  def step(self, controllers: Mapping[int, dict]) -> Mapping[int, Game]:
    for port, controller in controllers.items():
      send_controller(self.controllers[port], controller)

    gamestate = super().step()

    return {
        p: get_game(gamestate, self._player_to_ports[p])
        for p in self._players
    }


def run_env(init_kwargs, conn):
  env = Environment(**init_kwargs)

  while True:
    controllers = conn.recv()
    if controllers is None:
      break

    conn.send(env.step(controllers))

  env.stop()
  conn.close()

class AsyncEnv:

  def __init__(self, **kwargs):
    self._parent_conn, child_conn = mp.Pipe()
    self._process = mp.Process(target=run_env, args=(kwargs, child_conn))
    self._process.start()

  def stop(self):
    self._parent_conn.send(None)
    self._process.join()
    self._parent_conn.close()

  def send(self, controllers):
    self._parent_conn.send(controllers)

  def recv(self):
    return self._parent_conn.recv()