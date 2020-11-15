import copy

from absl import flags

FLAGS = flags.FLAGS

def update_config_from_path(path, value, config):
  path = list(reversed(path.split('.')))

  while len(path) > 1:
    key = path.pop()
    config = config[key]

  key = path[0]
  type_ = type(config[key])
  config[key] = type_(value)

class ConfigParser:

  def __init__(self, name, default_config):
    self.name = name
    self.default = default_config
    flags.DEFINE_multi_string(
        name, [],
        f'Overrides for {name}. Usage: --{name} some.path.in.config=value')

  def parse(self):
    config = copy.deepcopy(self.default)

    for setting in getattr(FLAGS, self.name):
      path, value = setting.split('=')
      update_config_from_path(path, value, config)

    return config
