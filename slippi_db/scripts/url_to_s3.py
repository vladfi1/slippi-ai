import os
from contextlib import contextmanager
import tempfile
from typing import Iterable
import urllib.request

from absl import app
from absl import flags

from slippi_db import upload_lib

URL_KEY = 'url'

def _tmp_dir(in_memory: bool):
  return '/dev/shm' if in_memory else None

@contextmanager
def get_file(url: str, in_memory: bool):
  f = tempfile.NamedTemporaryFile(dir=_tmp_dir(in_memory))
  with upload_lib.Timer(f"download {url}"):
    filename, response = urllib.request.urlretrieve(url, filename=f.name)
  del response

  print(f'Downloaded {url} to {filename}.')

  try:
    yield os.path.basename(filename), f
  finally:
    f.close()

def upload_file(db: upload_lib.ReplayDB, desc: str, url: str, in_memory: bool):
  print(f"Uploading {url}")
  with get_file(url, in_memory) as (name, f):
    db.upload_raw(
        name=name,
        f=f,
        check_max_size=False,
        url=url,
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
    uploads = db.raw.find({URL_KEY: {'$exists': True}}, [URL_KEY])
    existing_urls = set(doc[URL_KEY] for doc in uploads)
  else:
    existing_urls = set()

  for url in urls:
    if url in existing_urls:
      print('Skipping duplicate url:', url)
      continue

    upload_file(db, desc, url, in_memory)
    existing_urls.add(url)

flags.DEFINE_string('env', None, 'production environment')
flags.DEFINE_string('desc', '', 'description')
flags.DEFINE_boolean('in_memory', False, 'Use ram for temporary memory.')

flags.DEFINE_multi_string('url', [], 'url of file to upload')
flags.DEFINE_string('urls_file', None, 'path to text file')

FLAGS = flags.FLAGS

def main(_):
  if FLAGS.url:
    upload_multiple(FLAGS.env, FLAGS.desc, FLAGS.url, FLAGS.in_memory)
  elif FLAGS.urls_file:
    with open(FLAGS.urls_file) as f:
      text = f.read()
    lines = text.split('\n')
    urls = [l for l in lines if l]
    upload_multiple(FLAGS.env, FLAGS.desc, urls, FLAGS.in_memory)

if __name__ == '__main__':
  app.run(main)
