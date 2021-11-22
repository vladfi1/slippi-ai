import os
from typing import Iterable

from absl import app
from absl import flags

import upload_lib

def upload_file(db: upload_lib.ReplayDB, path: str, desc: str) -> str:
  filename = os.path.basename(path)

  with open(path, 'rb') as f:
    return db.upload_raw(
        name=filename,
        f=f,
        check_max_size=False,
        description=desc,
    )

def upload_multiple(
    env: str,
    desc: str,
    paths: Iterable[str],
):
  db = upload_lib.ReplayDB(env)

  for path in paths:
    result = upload_file(db, path, desc)
    print(result)

flags.DEFINE_string('env', None, 'production environment')
flags.DEFINE_string('desc', '', 'description')

flags.DEFINE_multi_string('path', [], 'path of file to upload')

FLAGS = flags.FLAGS

def main(_):
  if FLAGS.env is None:
    raise ValueError('Must specify the env.')
  upload_multiple(FLAGS.env, FLAGS.desc, FLAGS.path)

if __name__ == '__main__':
  app.run(main)
