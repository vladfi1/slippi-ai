import functools
import logging
import typing as tp

from absl import app
from absl import flags
import fancyflags as ff

from ray import tune
import ray
from ray.air.callbacks.wandb import WandbLoggerCallback
# from ray.tune.integration.wandb import WandbLoggerCallback
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents import ppo

from slippi_ai import eval_lib, embed, utils
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

CONFIG = ff.DEFINE_dict(
    'config', **utils.get_flags_from_default(_DEFAULT_CONFIG))

TUNE = ff.DEFINE_dict(
    'tune',
    checkpoint_freq=ff.Integer(20),
    keep_checkpoints_num=ff.Integer(3),
    checkpoint_at_end=ff.Boolean(False),
    restore=ff.String(None, 'path to checkpoint to restore from'),
    resume=ff.Enum(None, ["LOCAL", "REMOTE", "PROMPT", "ERRORED_ONLY", "AUTO"]),
    sync_config=dict(
        upload_dir=ff.String(None, 'Path to local or remote folder.'),
        syncer=ff.String('auto'),
        sync_on_checkpoint=ff.Boolean(True),
        sync_period=ff.Integer(300),
    ),
    verbose=ff.Integer(3),
)

WANDB = ff.DEFINE_dict(
    'wandb',
    use=ff.Boolean(False),
    project=ff.String('slippi-ai'),
    api_key_file=ff.String("~/.wandb"),
    log_config=ff.Boolean(False),
    save_checkpoints=ff.Boolean(False),
)

ENV = ff.DEFINE_dict(
    'env',
    num_envs=ff.Integer(1),
)

RAY_INIT = flags.DEFINE_boolean('ray_init', False, 'init ray')

class AdaptorEnv(MeleeEnv):
  def __init__(self, config):
    # tune converts integer keys to strings when checkpointing
    # we convert them back here in case we're resuming
    dolphin_cfg = config['dolphin']
    dolphin_cfg['players'] = {
        int(k): v for k, v in dolphin_cfg['players'].items()}

    dolphin_fn = functools.partial(
        dolphin_lib.Dolphin, **dolphin_cfg)
    super().__init__(dolphin_fn, **config['env'])

players = {
    1: dolphin_lib.AI(),
    2: dolphin_lib.CPU(),
}

def main(_):
  if RAY_INIT.value:
    ray.init('auto')

  config = CONFIG.value.copy()

  # non-flags
  updates = {
      "env": AdaptorEnv,
      "env_config": dict(
          dolphin=dict(
              players=players,
              **DOLPHIN.value),
          env=ENV.value,
      ),

      "multiagent": {
          "policies": {
              "default_policy": PolicySpec(
                  observation_space=embed.default_embed_game.space(),
                  action_space=embed.embed_controller_discrete.space(),
              )
          }
      },

      # force
      "custom_resources_per_worker": {"worker": 1},
  }
  _update_dicts(config, updates)

  tune_config = TUNE.value
  tune_config['sync_config'] = tune.SyncConfig(**tune_config['sync_config'])

  callbacks = []

  wandb_config = WANDB.value.copy()
  if wandb_config.pop('use'):
    wandb_callback = WandbLoggerCallback(**wandb_config)
    callbacks.append(wandb_callback)

  tune.run(
      "PPO",
      stop={"episode_reward_mean": 1},
      config=config,
      callbacks=callbacks,
      **tune_config,
  )

if __name__ == '__main__':
  app.run(main)
