import pickle
import typing as tp

from slippi_ai import policies
from slippi_ai.policies import Platform

PLATFORM_KEY = 'platform'

def get_platform(
    config: dict[str, tp.Any],
    expected_platform: tp.Optional[Platform] = None,
) -> Platform:
  if expected_platform is not None:
    if PLATFORM_KEY in config:
      if config[PLATFORM_KEY] != expected_platform.value:
        raise ValueError(f'Config has platform {config[PLATFORM_KEY]}, but expected {expected_platform.value}')
    else:
      config[PLATFORM_KEY] = expected_platform.value

    return expected_platform

  # Why do we save as string instead of enum? We use pickle after all.
  platform_str: str = config.get(PLATFORM_KEY, Platform.TF.value)
  return Platform(platform_str)

def upgrade_config(
    config: dict[str, tp.Any],
    platform: tp.Optional[Platform] = None,
) -> dict[str, tp.Any]:
  platform = get_platform(config, platform)

  match platform:
    case Platform.JAX:
      from slippi_ai.jax import saving as jax_saving
      return jax_saving.upgrade_config(config)
    case Platform.TF:
      from slippi_ai.tf import saving as tf_saving
      return tf_saving.upgrade_config(config)

def load_policy_from_state(state: dict) -> policies.Policy:
  config: dict = state['config']
  platform = get_platform(config)

  match platform:
    case Platform.JAX:
      from slippi_ai.jax import saving as jax_saving
      return jax_saving.load_policy_from_state(state)
    case Platform.TF:
      from slippi_ai.tf import saving as tf_saving
      return tf_saving.load_policy_from_state(state)


class CustomUnpickler(pickle.Unpickler):
  def find_class(self, module, name):
    # The embed config has an Enum; everything else is saved as a dict.
    if module == 'slippi_ai.embed' and name == 'ItemsType':
      module = 'slippi_ai.tf.embed'
    return super().find_class(module, name)


def load_state_from_disk(path: str) -> dict:
  with open(path, 'rb') as f:
    return CustomUnpickler(f).load()

def load_policy_from_disk(path: str) -> policies.Policy:
  state = load_state_from_disk(path)
  return load_policy_from_state(state)
