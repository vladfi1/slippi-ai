import itertools

from absl import app
from absl import flags
import ray

from slippi_db import decompress, upload_lib

_ENV = flags.DEFINE_string('env', 'test', 'production environment')

ray.init(address='auto')

process_raw = ray.remote(num_cpus=1)(decompress.process_raw)

def process_all(env: str):
  raw_db = upload_lib.get_db(env, 'raw')
  raw_info = raw_db.find({})
  summaries = []
  for doc in raw_info:
    if doc['type'] not in decompress.SUPPORTED_TYPES:
      print(f'Skipping {doc["filename"]} ({doc["description"]}).')
      continue
    summaries.append(process_raw.remote(env, doc['key']))

  summaries = ray.get(summaries)
  summaries = list(itertools.chain(*summaries))

  num_uploaded = 0
  if summaries:
    filenames, was_uploaded = zip(*summaries)
    num_uploaded = sum(was_uploaded)

  print(f'Uploaded {num_uploaded} slippi replays.')
  return num_uploaded

def main(_):
  process_all(_ENV.value)

if __name__ == '__main__':
  app.run(main)
