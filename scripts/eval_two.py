"""Run a game between two trained agents, or vs a human player.

To run two agents against each other:

```shell
python scripts/eval_two.py \
  --dolphin.path=/path/to/slippi-dolphin \
  --dolphin.iso=/path/to/SSBM.iso \
  --p1.ai.path=/path/to/agent1 \
  --p2.ai.path=/path/to/agent2
```

To run an agent against a human player in port 1:

```shell
python scripts/eval_two.py \
  --dolphin.path=/path/to/slippi-dolphin \
  --dolphin.iso=/path/to/SSBM.iso \
  --p1.type=human \
  --p2.ai.path=/path/to/agent
```

"""

import logging
import os

from absl import app
from absl import flags
import fancyflags as ff

from slippi_ai import eval_lib, flag_utils, utils
from slippi_ai import dolphin as dolphin_lib

PORTS = (1, 2)

player_flags = utils.map_nt(lambda x: x, eval_lib.PLAYER_FLAGS)
player_flags['ai']['async_inference'] = ff.Boolean(True)

PLAYERS = {p: ff.DEFINE_dict(f"p{p}", **player_flags) for p in PORTS}

dolphin_config = dolphin_lib.DolphinConfig(
    headless=False,
    infinite_time=False,
    path=os.environ.get('DOLPHIN_PATH'),
    iso=os.environ.get('ISO_PATH'),
)
DOLPHIN = ff.DEFINE_dict(
    'dolphin', **flag_utils.get_flags_from_default(dolphin_config))

NUM_GAMES = flags.DEFINE_integer('num_games', None, 'Number of games to play')

FLAGS = flags.FLAGS

def main(_):
  eval_lib.disable_gpus()

  players = {
      port: eval_lib.get_player(**player.value)
      for port, player in PLAYERS.items()
  }

  agents: list[eval_lib.Agent] = []

  for port, opponent_port in zip(PORTS, reversed(PORTS)):
    player = players[port]
    if isinstance(player, dolphin_lib.AI):
      agent = eval_lib.build_agent(
          port=port,
          opponent_port=opponent_port,
          console_delay=DOLPHIN.value['online_delay'],
          **PLAYERS[port].value['ai'],
      )
      agent.start()
      agents.append(agent)

      eval_lib.update_character(player, agent.config)

  dolphin = dolphin_lib.Dolphin(
      players=players,
      **dolphin_lib.DolphinConfig.kwargs_from_flags(DOLPHIN.value),
  )

  for agent in agents:
    agent.set_controller(dolphin.controllers[agent._port])

  step_timer = utils.Profiler()

  num_games = 0

  # Main loop
  try:
    for gamestate in dolphin.iter_gamestates(skip_menu_frames=False):
      if dolphin_lib.is_menu_state(gamestate):
        if num_games == NUM_GAMES.value:
          break
        continue

      if gamestate.frame == -123: # initial frame
        num_games += 1
        logging.info(f'Game {num_games}')

      with step_timer:
        for agent in agents:
          agent.step(gamestate)

      if gamestate.frame > 0 and gamestate.frame % (15 * 60) == 0:
        logging.info(f'step_time: {step_timer.mean_time():.3f}')
  finally:
    for agent in agents:
      agent.stop()
    dolphin.stop()

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
