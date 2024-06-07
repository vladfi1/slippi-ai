"""Test that controller inputs match observed outputs."""

from absl import app
from absl import flags
import fancyflags as ff
import tqdm

from melee.enums import Button, Stage
from slippi_ai import dolphin as dolphin_lib
from slippi_ai import flag_utils

dolphin_config = dolphin_lib.DolphinConfig(
    headless=True,
    infinite_time=True,
    stage=Stage.FINAL_DESTINATION,
)

DOLPHIN = ff.DEFINE_dict(
    'dolphin', **flag_utils.get_flags_from_default(dolphin_config))

FLAGS = flags.FLAGS

# TODO: use methods from controller_lib

# TODO: test combination of digital and analog presses
def test_triggers(dolphin: dolphin_lib.Dolphin, port: int):
  controller = dolphin.controllers[port]

  for raw in range(141):
    x = raw / 140
    controller.press_shoulder(Button.BUTTON_L, x)
    gamestate = dolphin.step()
    output = gamestate.players[port].controller_state.l_shoulder

    expected_raw = 0 if raw < 43 else raw

    assert round(output * 140) == expected_raw, f"{raw} {output}"
  print("all triggers match")

Raw = int

def naive_raw_to_float(x: Raw) -> float:
  return x / 160 + 0.5

def output_to_raw(x: float) -> Raw:
  return round(x * 160 - 80)

def is_deadzone(x: Raw) -> bool:
  return x != 0 and abs(x) < 23

def is_valid_raw(xy) -> bool:
  x, y = xy

  if x * x + y * y > 80 * 80:
    return False

  if is_deadzone(x) or is_deadzone(y):
    return False

  return True

all_raw = [(x, y) for x in range(-80, 81) for y in range(-80, 81)]
valid_raw = list(filter(is_valid_raw, all_raw))

def test_sticks(dolphin: dolphin_lib.Dolphin, port: int):
  controller = dolphin.controllers[port]

  def test_xy(raw_xy: tuple[Raw, Raw]):
    raw_x, raw_y = raw_xy
    x = naive_raw_to_float(raw_x)
    y = naive_raw_to_float(raw_y)
    controller.tilt_analog(Button.BUTTON_MAIN, x, y)
    gamestate = dolphin.step()
    outputs = gamestate.players[port].controller_state.main_stick
    out_x, out_y = outputs
    assert output_to_raw(out_x) == raw_x, (raw_xy, outputs)
    assert output_to_raw(out_y) == raw_y, (raw_xy, outputs)

  for raw_xy in tqdm.tqdm(valid_raw):
    test_xy(raw_xy)

  print("all sticks match")

valid_buttons = [
    Button.BUTTON_A, Button.BUTTON_B, Button.BUTTON_X, Button.BUTTON_Y,
    Button.BUTTON_Z, Button.BUTTON_L, Button.BUTTON_R, Button.BUTTON_D_UP,
    Button.BUTTON_D_DOWN, Button.BUTTON_D_LEFT, Button.BUTTON_D_RIGHT
]

def test_buttons(dolphin: dolphin_lib.Dolphin, port: int):
  controller = dolphin.controllers[port]

  def test_button(button: Button):
    controller.press_button(button)
    gamestate = dolphin.step()
    assert gamestate.players[port].controller_state.button[button], button

    controller.release_button(button)
    gamestate = dolphin.step()
    assert not gamestate.players[port].controller_state.button[button], button

  for button in valid_buttons:
    test_button(button)

  print("all buttons match")

def main(_):
  PORT = 1
  players = {
    PORT: dolphin_lib.AI(),
    2: dolphin_lib.AI(),
  }

  dolphin_kwargs = dolphin_lib.DolphinConfig.kwargs_from_flags(DOLPHIN.value)

  dolphin = dolphin_lib.Dolphin(
      players=players,
      **dolphin_kwargs,
  )

  controller = dolphin.controllers[PORT]

  for _ in range(123):
    dolphin.step()

  print('Starting tests')

  # Run in order of least to most time-consuming.

  test_buttons(dolphin, PORT)
  controller.release_all()
  dolphin.step()

  test_triggers(dolphin, PORT)
  controller.release_all()
  dolphin.step()

  test_sticks(dolphin, PORT)

  dolphin.stop()

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
