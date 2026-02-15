"""Test that controller inputs match observed outputs."""

from absl import app
from absl import flags
import fancyflags as ff
import tqdm

from melee.enums import Button, Stage
from slippi_ai import dolphin as dolphin_lib
from slippi_ai import flag_utils, controller_lib

dolphin_config = dolphin_lib.DolphinConfig(
    headless=True,
    infinite_time=True,
    stage=Stage.FINAL_DESTINATION,
    emulation_speed=0,
)

DOLPHIN = ff.DEFINE_dict(
    'dolphin', **flag_utils.get_flags_from_default(dolphin_config))

FLAGS = flags.FLAGS

# TODO: test combination of digital and analog presses
def test_triggers(dolphin: dolphin_lib.Dolphin, port: int):
  controller = dolphin.controllers[port]

  for raw in range(controller_lib.TRIGGER_SPACING + 1):
    x = controller_lib.from_raw_trigger(raw)
    controller.press_shoulder(Button.BUTTON_L, x)
    gamestate = dolphin.step()
    output = gamestate.players[port].controller_state.l_shoulder

    expected_raw = 0 if raw < controller_lib.TRIGGER_DEADZONE else raw

    assert round(output * controller_lib.TRIGGER_SPACING) == expected_raw, f"{raw} {output}"
  print("all triggers match")

Raw = controller_lib.RawAxis

def test_sticks(
    dolphin: dolphin_lib.Dolphin,
    port: int,
    stick: Button = Button.BUTTON_MAIN,
):
  assert stick in [Button.BUTTON_MAIN, Button.BUTTON_C]

  controller = dolphin.controllers[port]

  def test_xy(raw_xy: tuple[Raw, Raw]):
    raw_x, raw_y = raw_xy
    x = controller_lib.from_raw_axis(raw_x)
    y = controller_lib.from_raw_axis(raw_y)
    controller.tilt_analog(stick, x, y)
    gamestate = dolphin.step()
    if stick == Button.BUTTON_MAIN:
      outputs = gamestate.players[port].controller_state.main_stick
    else:
      outputs = gamestate.players[port].controller_state.c_stick
    out_x, out_y = outputs
    assert controller_lib.to_raw_axis(out_x) == raw_x, (raw_xy, outputs)
    assert controller_lib.to_raw_axis(out_y) == raw_y, (raw_xy, outputs)

  for raw_xy in tqdm.tqdm(controller_lib.VALID_RAW_STICKS):
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

  for stick in [Button.BUTTON_MAIN, Button.BUTTON_C]:
    print(f'Testing {stick} stick')
    test_sticks(dolphin, PORT, stick=stick)

  # TODO: test combinations of inputs

  dolphin.stop()

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
