"""Generate metadata for all slp files.

Run remotely with

ray submit --start slippi_db/parsing_cluster.yaml \
  slippi_db/scripts/generate_metadata.py --env test
"""

import concurrent.futures
import os
import time
from typing import Dict, List, Iterator

from absl import app
from absl import flags
import ray

from slippi_db import preprocessing
from slippi_db import upload_lib
from slippi_db.utils import monitor

flags.DEFINE_string('env', 'test', 'production environment')
flags.DEFINE_bool('wipe', False, 'Wipe existing metadata')
flags.DEFINE_bool('local', False, 'Run locally instead of ray')

FLAGS = flags.FLAGS

def generate_one(env: str, key: str) -> dict:
  """Generate metadata for a single slp file."""
  slp_file = None
  try:
    slp_file = upload_lib.download_slp_to_file(env, key)
  except MemoryError as e:
    return dict(
        failed=True,
        reason='MemoryError',
        key=key,
    )

  metadata = preprocessing.get_metadata_safe(slp_file.name)
  os.remove(slp_file.name)

  metadata.update(key=key)
  return metadata

def generate_many(env: str, keys: List[str]) -> List[dict]:
  return [generate_one(env, key) for key in keys]

generate_one_ray = ray.remote(
    scheduling_strategy="SPREAD",
    resources=dict(workers=1),
)(generate_one)

def generate_many_ray(env: str, keys: List[str]) -> Iterator[dict]:
  object_refs: List[ray.ObjectRef] = []
  futures: Dict[concurrent.futures.Future, str] = {}
  for key in keys:
    obj_ref = generate_one_ray.remote(env, key)
    object_refs.append(obj_ref)
    futures[obj_ref.future()] = key

  yield from monitor(futures, log_interval=10)

def generate_all(env: str, wipe=False, local=False):
  start_time = time.perf_counter()

  slp_db = upload_lib.get_db(env, upload_lib.SLP)
  meta_db = upload_lib.get_db(env, upload_lib.META)

  if wipe:
    meta_db.delete_many({})  # maybe drop collection instead?

  all_keys = slp_db.find({}, ['key'])
  all_keys = [doc['key'] for doc in all_keys]

  existing = meta_db.find({}, ['key'])
  existing = set(doc['key'] for doc in existing)

  to_generate = [key for key in all_keys if key not in existing]
  skipped = len(all_keys) - len(to_generate)

  print(f'todo: {len(to_generate)}, skip: {skipped}')

  if local:
    metadata = generate_many(env, to_generate)
  else:
    metadata = list(generate_many_ray(env, to_generate))

  with upload_lib.Timer('meta_db.insert_many'):
    meta_db.insert_many(metadata)

  run_time = time.perf_counter() - start_time

  num_generated = len(metadata)
  gps = num_generated / run_time
  print(
      f'gen: {num_generated}, skip: {skipped}, '
      f'time: {run_time:.0f}, rate: {gps:.1f}')

def main(_):
  if not FLAGS.local:
    ray.init('auto')

  generate_all(FLAGS.env, FLAGS.wipe, FLAGS.local)

if __name__ == '__main__':
  app.run(main)
