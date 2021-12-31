"""Test a trained model."""

import sys
import signal

from sacred import Experiment

import melee
from slippi_ai import eval_lib

ex = Experiment('eval')

@ex.config
def config():
  saved_model_path = None
  tag = None
  dolphin_path = None
  iso_path = None
  cpu_level = 9
  runtime = 300
  sample_temperature = 1.0

@ex.automain
def main(saved_model_path, tag, dolphin_path, iso_path, _log, _config):
  players = {
      1: eval_lib.AI(),
      2: eval_lib.CPU(),
  }

  dolphin = eval_lib.Dolphin(dolphin_path, iso_path, players)

  if saved_model_path:
    policy = eval_lib.Policy.from_saved_model(saved_model_path)
  elif tag:
    policy = eval_lib.Policy.from_experiment(
      tag, sample_kwargs=dict(temperature=_config["sample_temperature"]))
  else:
    assert False

  agent = eval_lib.Agent(
      controller=dolphin.controllers[1],
      opponent_port=2,
      policy=policy,
  )

  total_frames = 60 * _config["runtime"]

  # Main loop
  for _ in range(total_frames):
    # "step" to the next frame
    gamestate, _ = dolphin.step()

    # if gamestate.frame == -123: # initial frame
    #   controller.release_all()

    agent.step(gamestate)

  dolphin.stop()
