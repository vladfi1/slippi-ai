"""Test a trained model."""

from typing import Optional
from absl import app
from absl import flags
import fancyflags as ff
from melee.enums import Character, Stage

from slippi_ai import eval_lib
from slippi_ai import dolphin as dolphin_lib

PORTS = (1, 2)

def build_policy(
    saved_model_path: Optional[str] = None,
    tag: Optional[str] = None,
    sample_temperature: float = 1.0,
) -> eval_lib.Policy:
  if saved_model_path:
    return eval_lib.Policy.from_saved_model(saved_model_path)
  elif tag:
    return eval_lib.Policy.from_experiment(
      tag, sample_kwargs=dict(temperature=sample_temperature))
  else:
    raise ValueError("Must specify one of 'tag' and 'saved_model_path'.")

_POLICY_FLAGS = ff.auto(build_policy)

_PLAYER_FLAGS = dict(
    type=ff.Enum('ai', ('ai', 'human', 'cpu')),
    character=ff.EnumClass(Character.FOX, Character),
    level=ff.Integer(9),
    ai=_POLICY_FLAGS,
)

# POLICY_FLAGS = {p: ff.DEFINE_dict(f"p{p}", **_POLICY_FLAGS) for p in PORTS}
PLAYER_FLAGS = {p: ff.DEFINE_dict(f"p{p}", **_PLAYER_FLAGS) for p in PORTS}

flags.DEFINE_string('iso_path', None, 'Path to SSBM iso.', required=True)
flags.DEFINE_string('dolphin_path', None, 'Path to dolphin binary dir.', required=True)
flags.DEFINE_boolean('save_replays', False, 'Save slippi replays.')
flags.DEFINE_enum_class('stage', Stage.YOSHIS_STORY, Stage, 'stage')
flags.DEFINE_integer('runtime', 300, 'Running time, in seconds.')

FLAGS = flags.FLAGS


def get_player(port: int) -> dolphin_lib.Player:
  player_flags = PLAYER_FLAGS[port].value
  player_type = player_flags['type']
  if player_type == 'ai':
    return dolphin_lib.AI(player_flags['character'])
  elif player_type == 'human':
    return dolphin_lib.Human()
  elif player_type == 'cpu':
    return dolphin_lib.CPU(player_flags['character'], player_flags['level'])

def main(_):
  players = {p: get_player(p) for p in PORTS}

  dolphin = dolphin_lib.Dolphin(
      FLAGS.dolphin_path,
      FLAGS.iso_path,
      players,
      stage=FLAGS.stage,
      save_replays=FLAGS.save_replays,
  )

  agents = []

  for port, opponent_port in zip(PORTS, reversed(PORTS)):
    if isinstance(players[port], dolphin_lib.AI):
      policy = build_policy(**PLAYER_FLAGS[port].value['ai'])

      agent = eval_lib.Agent(
          controller=dolphin.controllers[port],
          opponent_port=opponent_port,
          policy=policy,
      )
      agents.append(agent)

  total_frames = 60 * FLAGS.runtime

  # Main loop
  for _ in range(total_frames):
    # "step" to the next frame
    gamestate = dolphin.step()

    # if gamestate.frame == -123: # initial frame
    #   controller.release_all()

    for agent in agents:
      agent.step(gamestate)

  dolphin.stop()

if __name__ == '__main__':
  app.run(main)
