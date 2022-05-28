
from absl import app
from absl import flags
import fancyflags as ff

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents import pg

from slippi_ai import eval_lib, embed
from slippi_ai import dolphin as dolphin_lib
from slippi_ai.rllib.env import MeleeEnv

DOLPHIN = ff.DEFINE_dict('dolphin', **eval_lib.DOLPHIN_FLAGS)

class AdaptorEnv(MeleeEnv):
  def __init__(self, config):
    dolphin = dolphin_lib.Dolphin(**config)
    super().__init__(dolphin)

def main(_):

  players = {
      1: dolphin_lib.AI(),
      2: dolphin_lib.CPU(),
  }

  config = {
      "env": AdaptorEnv,
      "env_config": dict(
          players=players,
          **DOLPHIN.value,
      ),
      # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
      # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
      "model": {
          # "custom_model": "my_model",
          "vf_share_layers": True,
      },
      "multiagent": {
          "policies": {
              "default_policy": PolicySpec(
                  observation_space=embed.default_embed_game.space(),
                  action_space=embed.embed_controller_discrete.space(),
              )
          }
      },
      "num_workers": 0,  # parallelism
      "framework": "tf2",
  }

  pg_config = pg.DEFAULT_CONFIG.copy()
  pg_config.update(config)
  trainer = pg.PGTrainer(pg_config)

  for _ in range(100):
    trainer.train()

if __name__ == '__main__':
  app.run(main)
