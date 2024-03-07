
import melee

from slippi_ai.types import Controller

# this will be the autoregressive order in embed.py
LEGAL_BUTTONS = [
    melee.Button.BUTTON_A,
    melee.Button.BUTTON_B,
    melee.Button.BUTTON_X,
    melee.Button.BUTTON_Y,
    melee.Button.BUTTON_Z,
    melee.Button.BUTTON_L,
    melee.Button.BUTTON_R,
    melee.Button.BUTTON_D_UP,
]

def send_controller(controller: melee.Controller, controller_state: Controller):
  for b in LEGAL_BUTTONS:
    if getattr(controller_state.buttons, b.value):
      controller.press_button(b)
    else:
      controller.release_button(b)
  main_stick = controller_state.main_stick
  controller.tilt_analog(melee.Button.BUTTON_MAIN, main_stick.x, main_stick.y)
  c_stick = controller_state.c_stick
  controller.tilt_analog(melee.Button.BUTTON_C, c_stick.x, c_stick.y)
  controller.press_shoulder(melee.Button.BUTTON_L, controller_state.shoulder)
  # flush the controller?
