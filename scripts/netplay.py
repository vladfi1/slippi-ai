"""Test a trained model."""

import collections
import json

from absl import app
from absl import flags
import fancyflags as ff

from slippi_ai import eval_lib, types, utils
from slippi_ai import dolphin as dolphin_lib
from slippi_db.parse_libmelee import get_controller

PLAYER = ff.DEFINE_dict('player', **eval_lib.PLAYER_FLAGS)

dolphin_flags = dolphin_lib.DOLPHIN_FLAGS.copy()
dolphin_flags.update(
    online_delay=ff.Integer(15),
    connect_code=ff.String(None, required=True),
    user_json_path=ff.String(None, required=True),
    # blocking_input=ff.Boolean(False),
)
DOLPHIN = ff.DEFINE_dict('dolphin', **dolphin_flags)

CHECK_INPUTS = flags.DEFINE_boolean('check_inputs', False, 'Check inputs.')

FLAGS = flags.FLAGS

def main(_):
  eval_lib.disable_gpus()

  port = 1

  player = eval_lib.get_player(**PLAYER.value)

  dolphin = dolphin_lib.Dolphin(
      players={port: player},
      **DOLPHIN.value,
  )

  agents: list[eval_lib.Agent] = []

  if isinstance(player, dolphin_lib.AI):
    agent = eval_lib.build_agent(
        controller=dolphin.controllers[port],
        opponent_port=None,  # will be set later
        console_delay=DOLPHIN.value['online_delay'],
        run_on_cpu=True,
        **PLAYER.value['ai'],
    )
    agents.append(agent)

    eval_lib.update_character(player, agent.config)

  # Start game
  gamestate = dolphin.step()

  with open(DOLPHIN.value['user_json_path']) as f:
    user_json = json.load(f)
  display_name = user_json['displayName']

  name_to_port = {
      player.displayName: port for port, player in gamestate.players.items()
  }

  actual_port = name_to_port[display_name]
  ports = list(gamestate.players)
  ports.remove(actual_port)
  opponent_port = ports[0]
  agent.players = (actual_port, opponent_port)

  # Main loop
  agent.start()
  try:
    while True:
      if gamestate.frame == -123:
        action_queue = collections.deque(
            [None] * (1 + dolphin.console.online_delay))

      # "step" to the next frame
      prev_frame = gamestate.frame
      gamestate = dolphin.step()
      if CHECK_INPUTS.value:
        assert gamestate.frame == prev_frame + 1

      for agent in agents:
        action: types.Controller = agent.step(gamestate).controller_state
        action = utils.map_nt(lambda x: x[0], action)
        action_queue.appendleft(action)

        expected: types.Controller = action_queue.pop()
        if expected is None:
          continue

        if gamestate.frame < 0:
          continue

        observed = agent._agent._policy.controller_embedding.from_state(
            get_controller(gamestate.players[actual_port].controller_state))

        # deadzone can change observed stick values
        if CHECK_INPUTS.value and observed.buttons != expected.buttons:
          raise ValueError('Wrong controller seen')

  finally:
    agent.stop()
    dolphin.stop()

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
