"""Test a trained model."""

from sacred import Experiment

from slippi_ai import eval_lib
from slippi_ai import dolphin as dolphin_lib

ex = Experiment('eval_cpu')

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
def main(saved_model_path, tag, dolphin_path, iso_path, _config):
  players = {
      1: dolphin_lib.AI(),
      2: dolphin_lib.CPU(),
  }

  dolphin = dolphin_lib.Dolphin(dolphin_path, iso_path, players)

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
    gamestate = dolphin.step()

    # if gamestate.frame == -123: # initial frame
    #   controller.release_all()

    agent.step(gamestate)

  dolphin.stop()
