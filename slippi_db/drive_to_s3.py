import os
from contextlib import contextmanager
import tempfile
from typing import Iterable

from absl import app
from absl import flags
import gdown

import upload_lib

DRIVE_URL = 'drive_url'
URL_PREFIX = "https://drive.google.com/uc?id="

def _tmp_dir(in_memory: bool):
  return '/dev/shm' if in_memory else None

def normalize_drive_url(url: str):
  return url if url.startswith(URL_PREFIX) else URL_PREFIX + url

@contextmanager
def get_drive_file(url: str, in_memory: bool):
  tmpdir = tempfile.TemporaryDirectory(dir=_tmp_dir(in_memory))
  os.chdir(tmpdir.name)
  filename = gdown.download(normalize_drive_url(url))

  try:
    with open(os.path.join(tmpdir.name, filename), 'rb') as f:
      yield filename, f
  finally:
    tmpdir.cleanup()

def upload_file(db: upload_lib.ReplayDB, desc: str, url: str, in_memory: bool):
  with get_drive_file(url, in_memory) as (name, f):
    db.upload_raw(
        name=name,
        f=f,
        check_max_size=False,
        drive_url=url,
        description=desc,
    )

def upload_multiple(
    env: str,
    desc: str,
    urls: Iterable[str],
    in_memory: bool,
    avoid_duplicate_urls=True,
):
  db = upload_lib.ReplayDB(env)

  if avoid_duplicate_urls:
    drive_uploads = db.raw.find({DRIVE_URL: {'$exists': True}}, [DRIVE_URL])
    drive_urls = set(doc[DRIVE_URL] for doc in drive_uploads)
  else:
    drive_urls = set()

  for url in urls:
    if url in drive_urls:
      print('Skipping duplicate url/id:', url)
      continue

    upload_file(db, desc, url, in_memory)
    drive_urls.add(url)

flags.DEFINE_string('env', None, 'production environment')
flags.DEFINE_string('desc', '', 'description')
flags.DEFINE_boolean('in_memory', False, 'Use ram for temporary memory.')

flags.DEFINE_multi_string('id', [], 'id of file to upload')
flags.DEFINE_string('urls_file', None, 'path to text file')

FLAGS = flags.FLAGS

def main(_):
  if FLAGS.id:
    upload_multiple(FLAGS.env, FLAGS.desc, FLAGS.id)
  elif FLAGS.urls_file:
    with open(FLAGS.urls_file) as f:
      lines = f.readlines()
    urls = [l for l in lines if l]
    upload_multiple(FLAGS.env, FLAGS.desc, urls)

if __name__ == '__main__':
  app.run(main)
