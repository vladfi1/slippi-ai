"""Test a trained model."""

import logging
import os

from absl import app
from absl import flags
import fancyflags as ff

from slippi_ai import eval_lib, flag_utils, utils
from slippi_ai import dolphin as dolphin_lib

PORTS = (1, 2)

PLAYERS = {p: ff.DEFINE_dict(f"p{p}", **eval_lib.PLAYER_FLAGS) for p in PORTS}

dolphin_config = dolphin_lib.DolphinConfig(
    headless=False,
    infinite_time=False,
    path=os.environ.get('DOLPHIN_PATH'),
    iso=os.environ.get('ISO_PATH'),
)
DOLPHIN = ff.DEFINE_dict(
    'dolphin', **flag_utils.get_flags_from_default(dolphin_config))

flags.DEFINE_integer('runtime', 300, 'Running time, in seconds.')

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

  total_frames = 60 * FLAGS.runtime

  step_timer = utils.Profiler()

  # Main loop
  try:
    for i in range(total_frames):
      # "step" to the next frame
      gamestate = dolphin.step()

      # if gamestate.frame == -123: # initial frame
      #   controller.release_all()

      with step_timer:
        for agent in agents:
          agent.step(gamestate)

      if i > 0 and i % (15 * 60) == 0:
        logging.info(f'step_time: {step_timer.mean_time():.3f}')
  finally:
    for agent in agents:
      agent.stop()
    dolphin.stop()

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
