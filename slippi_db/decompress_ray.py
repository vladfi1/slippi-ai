import time
from typing import List, Set

from absl import app
from absl import flags
import ray

from slippi_db import decompress, upload_lib
from slippi_db.decompress import Files, UploadResults, upload_files

FLAGS = flags.FLAGS
flags.DEFINE_boolean('local', False, 'run locally')
# flags.DEFINE_string('ray_address', 'auto', 'ray address')

# using mp for distributed uploading of slps within a raw
@ray.remote(resources=dict(host=1))
def process_raw_mp(*args, **kwargs):
  return decompress.process_raw(
    *args, uploader=decompress.upload_files_mp, **kwargs)

def process_all(env: str, in_memory: bool, skip_processed: bool = True):
  start_time = time.perf_counter()

  raw_db = upload_lib.get_db(env, 'raw')
  raw_info = raw_db.find({})
  results = []
  for doc in raw_info:
    skip = False
    reason = None

    if doc['type'] not in decompress.SUPPORTED_TYPES:
      skip = True
      reason = f'{doc["type"]} not supported'
    elif skip_processed and doc.get('processed', False):
      skip = True
      reason = 'already processed'
    
    if skip:
      print(f'Skipping {doc["filename"]} ({doc["description"]}): {reason}')
      continue

    print(f'Processing {doc["filename"]} ({doc["description"]}).')
    results.append(
      process_raw_mp.remote(
        env, doc['key'],
        in_memory=in_memory,
        skip_processed=skip_processed))

  results: List[decompress.UploadResults] = ray.get(results)
  uploads = []
  duplicates = []
  for result in results:
    uploads.extend(result.uploads)
    duplicates.extend(result.duplicates)
  num_uploaded = len(uploads)
  num_duplicates = len(duplicates)
  num_processed = num_uploaded + num_duplicates

  run_time = time.perf_counter() - start_time
  fps = num_processed / run_time
  print(f'Total stats for {len(results)} raw files:')
  print(f'Processed {num_processed} slippi replays in {run_time:.1f} seconds.')
  print(f'Uploaded {num_uploaded} with {num_duplicates} duplicates.')
  print(f'Files per second: {fps:.2f}')
  decompress.print_timings(results)
  return num_uploaded

def main(_):
  ray.init(address=None if FLAGS.local else 'auto')

  # flags defined in decompress.py
  process_all(
    FLAGS.env,
    in_memory=FLAGS.in_memory,
    skip_processed=not FLAGS.processed)

if __name__ == '__main__':
  app.run(main)
