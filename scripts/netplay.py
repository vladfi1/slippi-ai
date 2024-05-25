"""Test a trained model."""

import json
from typing import Optional

from absl import app
from absl import flags
import fancyflags as ff

from slippi_ai import eval_lib
from slippi_ai import dolphin as dolphin_lib

PLAYER = ff.DEFINE_dict('player', **eval_lib.PLAYER_FLAGS)

dolphin_flags = dolphin_lib.DOLPHIN_FLAGS.copy()
dolphin_flags.update(
    online_delay=ff.Integer(2),
    connect_code=ff.String(None, required=True),
    user_json_path=ff.String(None, required=True),
)
DOLPHIN = ff.DEFINE_dict('dolphin', **dolphin_flags)

flags.DEFINE_integer('runtime', 300, 'Running time, in seconds.')

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
    # character = player.character

  total_frames = 60 * FLAGS.runtime

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
  for _ in range(total_frames):
    # "step" to the next frame
    gamestate = dolphin.step()

    # if gamestate.frame == -123: # initial frame
    #   controller.release_all()

    for agent in agents:
      agent.step(gamestate)

  dolphin.stop()

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
