"""Test a trained model."""

import os
import time
import signal

from sacred import Experiment

import melee
import eval_lib

ex = Experiment('eval')

@ex.config
def config():
  saved_model_path = None
  dolphin_path = None
  iso_path = None
  runtime = 300

@ex.automain
def main(saved_model_path, dolphin_path, iso_path, _log, _config):
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

  ports = [1, 2]
  controllers = []

  for port in ports:
    controller = melee.Controller(
        console=console,
        port=port,
        type=melee.ControllerType.STANDARD)
    controllers.append(controller)

  # Run the console
  console.run(iso_path=iso_path)

  # Connect to the console
  _log.info("Connecting to console...")
  if not console.connect():
    _log.error("Failed to connect to the console.")
    return
  _log.info("Console connected")

  for c in controllers:
    print("Connecting controller to console...")
    if not c.connect():
        print("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    print("Controller connected")

  policies = []

  for controller, opponent_port in zip(controllers, reversed(ports)):
    policy = eval_lib.SavedModelPolicy(
        controller=controller,
        opponent_port=opponent_port,
        path=saved_model_path,
    )
    policies.append(policy)

  total_frames = 60 * _config["runtime"]
  num_frames = 0

  # Main loop
  while num_frames < total_frames:
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
      for policy in policies:
        policy.step(gamestate)
      num_frames += 1
    else:
      for i, controller in enumerate(controllers):
        melee.MenuHelper.menu_helper_simple(
            gamestate,
            controller,
            melee.Character.FOX,
            melee.Stage.YOSHIS_STORY,
            costume=i,
            autostart=i == 0,
            swag=False)

  console.stop()
