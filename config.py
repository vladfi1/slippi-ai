import copy
import json

from absl import flags

FLAGS = flags.FLAGS

def update_config_from_path(path, value, config):
  path = list(reversed(path))

  while len(path) > 1:
    key = path.pop()
    config = config[key]

  key = path[0]
  type_ = type(config[key])
  config[key] = type_(value)

def update_config_from_dict(config, dict_):
  for k, v in dict_.items():
    if isinstance(v, dict):
      update_config_from_dict(config[k], v)
    else:
      assert type(config[k]) == type(v)
      config[k] = v

class ConfigParser:

  def __init__(self, name, default_config):
    self.name = name
    self.default = default_config
    flags.DEFINE_multi_string(
        name, [],
        f'Overrides for {name}. Usage: --{name} some.path.in.config=value')
    self.path_name = name + '_path'
    flags.DEFINE_string(self.path_name, None, f'Path to json file for {name}.')

  def parse(self):
    config = copy.deepcopy(self.default)

    path = getattr(FLAGS, self.path_name)
    if path:
      with open(path, 'r') as f:
        new_config = json.load(f)
      update_config_from_dict(config, new_config)

    for setting in getattr(FLAGS, self.name):
      path, value = setting.split('=')
      update_config_from_path(path.split('.'), value, config)

    return config
