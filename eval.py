"""Test a trained model."""

import os
import time
import signal

from sacred import Experiment

import numpy as np
import sonnet as snt
import tensorflow as tf

import melee

import embed

LOG_INTERVAL = 10
SAVE_INTERVAL = 300

ex = Experiment('eval')

@ex.config
def config():
  saved_model_path = None
  dolphin_path = None
  iso_path = None

@ex.automain
def main(saved_model_path, dolphin_path, iso_path, _log):
  embed_game = embed.make_game_embedding()
  policy = tf.saved_model.load(saved_model_path)
  sample = lambda *structs: policy.sample(*tf.nest.flatten(structs))
  hidden_state = policy.initial_state(1)

  console = melee.Console(
      path=dolphin_path,
      online_delay=0,
      blocking_input=True,
  )

  # This isn't necessary, but makes it so that Dolphin will get killed when you ^C
  def signal_handler(sig, frame):
      console.stop()
      print("Shutting down cleanly...")
      # sys.exit(0)

  signal.signal(signal.SIGINT, signal_handler)

  controller = melee.Controller(
      console=console,
      port=1,
      type=melee.ControllerType.STANDARD)
  cpu_controller = melee.Controller(
      console=console,
      port=2,
      type=melee.ControllerType.STANDARD)

  # Run the console
  console.run(iso_path=iso_path)

  # Connect to the console
  _log.info("Connecting to console...")
  if not console.connect():
    _log.error("Failed to connect to the console.")
    return
  _log.info("Console connected")

  for c in [controller, cpu_controller]:
    print("Connecting controller to console...")
    if not c.connect():
        print("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    print("Controller connected")

  action_repeat = 0
  repeats_left = 0

  # Main loop
  while True:
    # "step" to the next frame
    gamestate = console.step()
    if gamestate is None:
        continue

    if gamestate.frame == -123: # initial frame
      controller.release_all()

    # The console object keeps track of how long your bot is taking to process frames
    #   And can warn you if it's taking too long
    if console.processingtime * 1000 > 12:
        print("WARNING: Last frame took " + str(console.processingtime*1000) + "ms to process.")

    # What menu are we in?
    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
      if repeats_left > 0:
        repeats_left -= 1
        continue

      embedded_game = embed_game.from_state(gamestate), action_repeat, 0.
      batched_game = tf.nest.map_structure(
          lambda a: np.expand_dims(a, 0), embedded_game)
      sampled_controller_with_repeat, hidden_state = sample(
          batched_game, hidden_state)
      sampled_controller_with_repeat = tf.nest.map_structure(
          lambda t: np.squeeze(t.numpy(), 0), sampled_controller_with_repeat)
      sampled_controller = sampled_controller_with_repeat['controller']
      action_repeat = sampled_controller_with_repeat['action_repeat']
      repeats_left = action_repeat

      for b in embed.LEGAL_BUTTONS:
        if sampled_controller['button'][b.value]:
          controller.press_button(b)
        else:
          controller.release_button(b)
      main_stick = sampled_controller["main_stick"]
      controller.tilt_analog(melee.Button.BUTTON_MAIN, *main_stick)
      c_stick = sampled_controller["c_stick"]
      controller.tilt_analog(melee.Button.BUTTON_C, *c_stick)
      controller.press_shoulder(melee.Button.BUTTON_L, sampled_controller["l_shoulder"])
      controller.press_shoulder(melee.Button.BUTTON_R, sampled_controller["r_shoulder"])
    else:
      melee.MenuHelper.menu_helper_simple(gamestate,
                                          controller,
                                          melee.Character.FOX,
                                          melee.Stage.YOSHIS_STORY,
                                          connect_code=None,
                                          autostart=False,
                                          swag=False)
      melee.MenuHelper.menu_helper_simple(gamestate,
                                          cpu_controller,
                                          melee.Character.FOX,
                                          melee.Stage.YOSHIS_STORY,
                                          connect_code=None,
                                          cpu_level=9,
                                          autostart=True,
                                          swag=False)
