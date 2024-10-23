# Make sure not to import things unless we're the main module.
# This allows child processes to avoid importing tensorflow,
# which uses a lot of memory.

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  from absl import app
  import fancyflags as ff

  import wandb

  from slippi_ai import flag_utils
  from slippi_ai.nash import train_lib

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(train_lib.DEFAULT_CONFIG))

  # passed to wandb.init
  WANDB = ff.DEFINE_dict(
      'wandb',
      project=ff.String('slippi-ai'),
      mode=ff.Enum('disabled', ['online', 'offline', 'disabled']),
      group=ff.String('nash'),
      name=ff.String(None),
      notes=ff.String(None),
      dir=ff.String(None, 'directory to save logs'),
  )

  def main(_):
    config = flag_utils.dataclass_from_dict(train_lib.Config, CONFIG.value)

    wandb_kwargs = dict(WANDB.value)
    if config.tag:
      wandb_kwargs['name'] = config.tag
    wandb.init(
        config=CONFIG.value,
        **wandb_kwargs,
    )
    train_lib.train(config)

  app.run(main)
