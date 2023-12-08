from absl import app
from absl import flags

from slippi_db import upload_lib

ENV = flags.DEFINE_string('env', None, 'env', required=True)
DRY_RUN = flags.DEFINE_bool('dry_run', False, 'dry run')

def main(_):
  upload_lib.remove_processed(ENV.value, DRY_RUN.value)

if __name__ == '__main__':
  app.run(main)
