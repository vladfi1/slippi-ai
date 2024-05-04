# Make sure not to import things unless we're the main module.
# This allows child processes to avoid importing tensorflow,
# which uses a lot of memory.
if __name__ == '__main__':
  __spec__ = None  # https://github.com/python/cpython/issues/87115

  from absl import app
  import fancyflags as ff

  from slippi_ai import flag_utils
  from slippi_ai.rl import run_lib


  CONFIG = ff.DEFINE_dict(
      'config',
      **flag_utils.get_flags_from_dataclass(run_lib.Config))

  def main(_):
    config = flag_utils.dataclass_from_dict(run_lib.Config, CONFIG.value)
    run_lib.run(config)

  app.run(main)
