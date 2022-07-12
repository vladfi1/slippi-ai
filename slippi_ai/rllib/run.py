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
from ray.rllib.algorithms import registry

from slippi_ai import eval_lib, embed, utils
from slippi_ai import networks, controller_heads
from slippi_ai import dolphin as dolphin_lib
from slippi_ai.rllib import model
from slippi_ai.rllib.env import MeleeEnv

DOLPHIN = ff.DEFINE_dict('dolphin', **eval_lib.DOLPHIN_FLAGS)

ALGORITHM_NAMES = "ppo", "a2c", "impala"

ALGORITHMS = {
    name: registry.get_algorithm_class(
        name.upper(), return_config=True)
    for name in ALGORITHM_NAMES
}

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

def _get_flags_for_algo(name: str) -> dict:
  _, default_config = ALGORITHMS[name]
  _update_dicts(default_config, _OVERRIDES)
  return utils.get_flags_from_default(default_config)

_ALGO = flags.DEFINE_enum('algo', 'ppo', ALGORITHM_NAMES, 'RL algorithm')

_ALGO_CONFIGS = {
    name: ff.DEFINE_dict(name, **_get_flags_for_algo(name))
    for name in ALGORITHM_NAMES
}

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

NETWORK = ff.DEFINE_dict(
    'network',
    **utils.get_flags_from_default(networks.DEFAULT_CONFIG)
)

CONTROLLER_HEAD = ff.DEFINE_dict(
    'controller_head',
    **utils.get_flags_from_default(controller_heads.DEFAULT_CONFIG)
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

  model.register()

  algo = _ALGO.value
  config = _ALGO_CONFIGS[algo].value.copy()

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

      "model": {
          "custom_model": "slippi",
          "custom_model_config": {
              "network": NETWORK.value,
              "controller_head": CONTROLLER_HEAD.value,
          },
          "_disable_preprocessor_api": True,
      },
      "_disable_preprocessor_api": True,

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
      algo.upper(),
      stop={"episode_reward_mean": 1},
      config=config,
      callbacks=callbacks,
      **tune_config,
  )

if __name__ == '__main__':
  app.run(main)
