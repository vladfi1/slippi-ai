import random

import numpy as np

import melee

from slippi_ai.types import Controller
from slippi_ai import types

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

RawTrigger = int
TRIGGER_DEADZONE = 43
TRIGGER_SPACING = 140

RAW_TRIGGERS = [0] + list(range(TRIGGER_DEADZONE, TRIGGER_SPACING + 1))

def from_raw_trigger(x: RawTrigger) -> float:
  return x / TRIGGER_SPACING

def to_raw_trigger(x: float) -> RawTrigger:
  return np.round(x * TRIGGER_SPACING).astype(np.int32)

RawAxis = int
AXIS_SPACING = 160
AXIS_RADIUS = AXIS_SPACING // 2
AXIS_DEADZONE = 23

def from_raw_axis(x: RawAxis) -> float:
  return x / AXIS_SPACING + 0.5

def to_raw_axis(x: float) -> RawAxis:
  return np.round((x - 0.5) * AXIS_SPACING).astype(np.int32)

def is_deadzone(x: RawAxis) -> bool:
  return x != 0 and abs(x) < AXIS_DEADZONE

def is_valid_raw_stick(xy: tuple[RawAxis, RawAxis]) -> bool:
  x, y = xy

  if x * x + y * y > AXIS_RADIUS * AXIS_RADIUS:
    return False

  if is_deadzone(x) or is_deadzone(y):
    return False

  return True

all_raw = [(x, y) for x in range(-AXIS_RADIUS, AXIS_RADIUS + 1) for y in range(-AXIS_RADIUS, AXIS_RADIUS + 1)]
VALID_RAW_STICKS = list(filter(is_valid_raw_stick, all_raw))

def to_raw_controller(c: Controller):
  return c._replace(
      main_stick=types.Stick(*map(to_raw_axis, c.main_stick)),
      shoulder=to_raw_trigger(c.shoulder),
  )

def controllers_equivalent(c1: Controller, c2: Controller):
  return to_raw_controller(c1) == to_raw_controller(c2)

# Pressing L/R/Z messes up the trigger values.
NON_SHOULDER_BUTTONS = [
    melee.Button.BUTTON_A,
    melee.Button.BUTTON_B,
    melee.Button.BUTTON_X,
    melee.Button.BUTTON_Y,
    # melee.Button.BUTTON_Z,
    # melee.Button.BUTTON_L,
    # melee.Button.BUTTON_R,
    # melee.Button.BUTTON_D_UP,
]

def random_valid_controller() -> Controller:
  main_xy = map(from_raw_axis, random.choice(VALID_RAW_STICKS))
  main_stick = types.Stick(*main_xy)
  c_stick = types.Stick(0.5, 0.5)  # TODO: what are valid C-sticks?

  shoulder = from_raw_trigger(random.choice(RAW_TRIGGERS))

  button = random.choice(NON_SHOULDER_BUTTONS + [None])
  # button = None
  is_pressed = {b.value: (b == button) for b in LEGAL_BUTTONS}
  buttons = types.Buttons(**is_pressed)

  return types.Controller(
      main_stick=main_stick,
      c_stick=c_stick,
      shoulder=shoulder,
      buttons=buttons,
  )
