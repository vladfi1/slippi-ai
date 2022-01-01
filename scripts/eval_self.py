"""Test a trained model."""

from sacred import Experiment

from slippi_ai import eval_lib
from slippi_ai import dolphin as dolphin_lib

ex = Experiment('eval_self')

@ex.config
def config():
  saved_model_path = None
  tag = None
  dolphin_path = None
  iso_path = None
  runtime = 300
  sample_temperature = 1.0

@ex.automain
def main(saved_model_path, tag, dolphin_path, iso_path, _config):
  ports = [1, 2]
  players = {p: dolphin_lib.AI() for p in ports}

  dolphin = dolphin_lib.Dolphin(dolphin_path, iso_path, players)

  if saved_model_path:
    policy = eval_lib.Policy.from_saved_model(saved_model_path)
  elif tag:
    policy = eval_lib.Policy.from_experiment(
      tag, sample_kwargs=dict(temperature=_config["sample_temperature"]))
  else:
    assert False

  agents = []

  for port, opponent_port in zip(ports, reversed(ports)):
    agent = eval_lib.Agent(
        controller=dolphin.controllers[port],
        opponent_port=opponent_port,
        policy=policy
    )
    agents.append(agent)

  total_frames = 60 * _config["runtime"]

  # Main loop
  for _ in range(total_frames):
    # "step" to the next frame
    gamestate, _ = dolphin.step()

    # if gamestate.frame == -123: # initial frame
    #   controller.release_all()

    for agent in agents:
      agent.step(gamestate)

  dolphin.stop()
