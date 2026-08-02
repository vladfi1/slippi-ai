"""Test a trained model."""

import collections
import json
import unicodedata
import logging

from absl import app
from absl import flags
import fancyflags as ff

import melee
from slippi_ai import eval_lib, types, utils, saving
from slippi_ai import dolphin as dolphin_lib
from slippi_db.parse_libmelee import get_controller

agent_flags = eval_lib.AGENT_FLAGS.copy()
agent_flags['async_inference'] = ff.Boolean(True)

AGENT = ff.DEFINE_dict('agent', **agent_flags)
CHAR = flags.DEFINE_enum_class('char', melee.Character.FOX, melee.Character, 'Character to use for AI player')
COSTUME = flags.DEFINE_integer('costume', None, 'Costume index to use for AI player. Only works online.')

dolphin_flags = dolphin_lib.DOLPHIN_FLAGS.copy()
dolphin_flags.update(
    online_delay=ff.Integer(15),
    connect_code=ff.String(None, required=True),
)
DOLPHIN = ff.DEFINE_dict('dolphin', **dolphin_flags)

RUNTIME = flags.DEFINE_integer('runtime', None, 'Runtime in seconds.')

FLAGS = flags.FLAGS

def main(_):
  eval_lib.disable_gpus()

  port = 1

  agent_state = saving.load_state_from_disk(AGENT.value['path'])

  player = dolphin_lib.AI(
      character=CHAR.value,
      costume=COSTUME.value
  )
  eval_lib.update_character(player, agent_state['config'])

  dolphin = dolphin_lib.Dolphin(
      players={port: player},
      **DOLPHIN.value,
  )

  # Warm up agent before starting game to prevent initial hiccup.
  agent = eval_lib.build_agent(
      controller=dolphin.controllers[port],
      opponent_port=None,  # will be set later
      console_delay=DOLPHIN.value['online_delay'],
      run_on_cpu=True,
      state=agent_state,
      **AGENT.value,
  )

  try:
    # Start game
    gamestate = dolphin.step()

    with open(DOLPHIN.value['user_json_path']) as f:
      user_json = json.load(f)
    display_name = user_json['displayName']

    name_to_port = {
        # player.displayName: port for port, player in gamestate.players.items()
        unicodedata.normalize('NFKC', player.displayName): port for port, player in gamestate.players.items()
    }

    actual_port = name_to_port[display_name]
    ports = list(gamestate.players)
    ports.remove(actual_port)
    opponent_port = ports[0]
    agent.players = (actual_port, opponent_port)

    # Main loop
    agent.start()
    agent.step(gamestate)

    num_frames = 1
    # Marker line consumed by the launcher's Discord bot to cancel its
    # "opponent never joined" timeout once the match is actually live.
    print('[NETPLAY_MATCH_STARTED]', flush=True)

    while True:
      gamestate = dolphin.step()
      agent.step(gamestate)

      num_frames += 1

      if RUNTIME.value is not None and num_frames >= RUNTIME.value * 60:
        break

  finally:
    agent.stop()
    dolphin.stop()

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
