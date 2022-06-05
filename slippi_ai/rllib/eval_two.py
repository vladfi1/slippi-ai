"""Test a trained model."""

from absl import app
from absl import flags
import fancyflags as ff

from ray.rllib.agents import ppo
from ray.rllib.policy.policy import PolicySpec

import melee

from slippi_ai import eval_lib, embed
from slippi_ai import dolphin as dolphin_lib

PORTS = (1, 2)

PLAYER_FLAGS = dict(
    type=ff.Enum('rllib', ('rllib', 'imitation', 'human', 'cpu'), 'Player type.'),
    character=ff.EnumClass(
        melee.Character.FOX, melee.Character,
        'Character selected by agent or CPU.'),
    level=ff.Integer(9, 'CPU level.'),
    imitation=eval_lib.AGENT_FLAGS,
    rllib=dict(
        ckpt=ff.String(None, 'Path to checkpoint'),
    ),
)

def get_player(config: dict) -> dolphin_lib.Player:
  type_ = config['type']
  if type_ in ('imitation', 'rllib'):
    return dolphin_lib.AI(config['character'])
  elif type_ == 'human':
    return dolphin_lib.Human()
  elif type_ == 'cpu':
    return dolphin_lib.CPU(config['character'], config['level'])
  raise ValueError(f'Unknown player type "{type_}".')

PLAYERS = {p: ff.DEFINE_dict(f"p{p}", **PLAYER_FLAGS) for p in PORTS}

DOLPHIN = ff.DEFINE_dict('dolphin', **eval_lib.DOLPHIN_FLAGS)

flags.DEFINE_integer('runtime', 300, 'Running time, in seconds.')

FLAGS = flags.FLAGS

class RLLibAgent:

  def __init__(
      self,
      ckpt: str,
      controller: melee.Controller,
      opponent_port: int,
      embed_game=embed.default_embed_game,
      embed_controller=embed.embed_controller_discrete,
  ) -> None:
    self.ckpt = ckpt
    self._controller = controller
    self._port = controller.port
    self._players = (self._port, opponent_port)

    self.trainer = ppo.PPOTrainer(
        config={
            "framework": "tf2",
            "eager_tracing": True,
            "num_workers": 0,
            "multiagent": {
                "policies": {
                    "default_policy": PolicySpec(
                        observation_space=embed_game.space(),
                        action_space=embed_controller.space(),
                    )
                }
            },
        },
    )
    self.trainer.restore(ckpt)

  def step(self, gamestate: melee.GameState):
    obs = eval_lib.get_game(gamestate, ports=self._players)
    obs = self.embed_game.from_state(obs)
    obs = self.embed_game.to_nest(obs)

    action = self.trainer.compute_single_action(obs)
    print(action)
    import ipdb; ipdb.set_trace()


def main(_):
  eval_lib.disable_gpus()

  players = {
      port: get_player(player.value)
      for port, player in PLAYERS.items()
  }

  dolphin = dolphin_lib.Dolphin(
      players=players,
      **DOLPHIN.value,
  )

  agents = []

  for port, opponent_port in zip(PORTS, reversed(PORTS)):
    player_config = PLAYERS[port].value
    player_type = player_config['type']
    if player_type == 'imitation':
      agent = eval_lib.build_agent(
          controller=dolphin.controllers[port],
          opponent_port=opponent_port,
          **player_config['ai'],
      )
      agents.append(agent)
    elif player_type == 'rllib':
      agent = RLLibAgent(
          controller=dolphin.controllers[port],
          opponent_port=opponent_port,
          **player_config['rllib'],
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
