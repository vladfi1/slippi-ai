import typing as tp

from absl import app
from absl import flags
import fancyflags as ff

from ray import tune
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents import ppo

from slippi_ai import eval_lib, embed
from slippi_ai import dolphin as dolphin_lib
from slippi_ai.rllib.env import MeleeEnv

DOLPHIN = ff.DEFINE_dict('dolphin', **eval_lib.DOLPHIN_FLAGS)

_DEFAULT_CONFIG = ppo.PPOConfig().to_dict()

_OVERRIDES = {
  "rollout_fragment_length": 2 * 60,
  "horizon": 60 * 60,
  "soft_horizon": True,
  "no_done_at_end": False,

  # "num_workers": 0,  # parallelism
  "framework": "tf2",
  "eager_tracing": True,
}

def _update_dicts(original, updates):
  if isinstance(original, dict):
    assert isinstance(updates, dict)
    for k, v in updates.items():
      if k in original:
        original[k] = _update_dicts(original[k], v)
      else:
        original[k] = v
    return original
  else:
    return updates

_update_dicts(_DEFAULT_CONFIG, _OVERRIDES)


def _get_flags_from_default(default) -> tp.Optional[ff.Item]:
  if isinstance(default, dict):
    result = {}
    for k, v in default.items():
      flag = _get_flags_from_default(v)
      if flag is not None:
        result[k] = flag
    return result
  elif isinstance(default, bool):
    return ff.Boolean(default)
  elif isinstance(default, int):
    return ff.Integer(default)
  elif isinstance(default, str):
    return ff.String(default)
  # elif isinstance(default, list):
  #   if default:
  #     elem = default[0]
  #     if isinstance(elem, int):
  #       return ff.Sequence()
  return None

CONFIG = ff.DEFINE_dict(
    'config', **_get_flags_from_default(_DEFAULT_CONFIG))

class AdaptorEnv(MeleeEnv):
  def __init__(self, config):
    dolphin = dolphin_lib.Dolphin(**config)
    super().__init__(dolphin)

players = {
    1: dolphin_lib.AI(),
    2: dolphin_lib.CPU(),
}

def main(_):

  config = CONFIG.value.copy()

  # non-flags
  updates = {
      "env": AdaptorEnv,
      "env_config": dict(
          players=players,
          **DOLPHIN.value,
      ),

      "multiagent": {
          "policies": {
              "default_policy": PolicySpec(
                  observation_space=embed.default_embed_game.space(),
                  action_space=embed.embed_controller_discrete.space(),
              )
          }
      },
  }
  _update_dicts(config, updates)

  tune.run(
      "PPO",
      stop={"episode_reward_mean": 1},
      config=config,
  )

if __name__ == '__main__':
  app.run(main)
