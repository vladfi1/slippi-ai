"""Test that controller inputs match observed outputs."""

from absl import app
from absl import flags
import fancyflags as ff

from melee.enums import Button
from slippi_ai import dolphin as dolphin_lib

DOLPHIN = ff.DEFINE_dict('dolphin', **dolphin_lib.DOLPHIN_FLAGS)

FLAGS = flags.FLAGS

def main(_):
  PORT = 1
  players = {
    PORT: dolphin_lib.AI(),
    2: dolphin_lib.AI(),
  }

  dolphin = dolphin_lib.Dolphin(
      players=players,
      **DOLPHIN.value,
  )

  controller = dolphin.controllers[PORT]

  for _ in range(123):
    gamestate = dolphin.step()

  for raw in range(141):
    if raw < 43:
      raw = 0

    x = raw / 140
    controller.press_shoulder(Button.BUTTON_L, x)
    gamestate = dolphin.step()
    output = gamestate.players[PORT].controller_state.l_shoulder

    assert round(output * 140) == raw, f"{raw} {output}"
  print("all triggers match")

  dolphin.stop()

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
