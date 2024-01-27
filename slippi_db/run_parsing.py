"""Create a Parquet dataset for machine learning.

ray submit --start slippi_db/parsing_cluster.yaml \
  slippi_db/run_parsing.py --env test
"""

import concurrent.futures
import enum
import itertools
import math
import os
import time
from typing import Dict, Iterator, List, Optional, Tuple

from absl import app
from absl import flags
import pyarrow as pa
import pyarrow.parquet as pq
import ray

from slippi_db import (
    upload_lib,
    parse_libmelee,
    parse_peppi,
    parsing_utils,
)
from slippi_db.test_peppi import get_singles_info
from slippi_db.utils import monitor


class Parser(enum.Enum):
  LIBMELEE = 'libmelee'
  PEPPI = 'peppi'

  def get_slp(self, path: str) -> pa.StructArray:
    if self == Parser.LIBMELEE:
      return parse_libmelee.get_slp(path)
    elif self == Parser.PEPPI:
      return parse_peppi.get_slp(path)

def process_slp(
    env: str,
    key: str,
    parser: Parser,
    dataset: str,
    compression: parsing_utils.CompressionType,
    **kwargs,
) -> Tuple[dict, dict]:
  timers = {
      k: upload_lib.Timer(k, verbose=False)
      for k in ['download_slp', 'parse_slp', 'put_object']
  }

  with timers['download_slp']:
    slp_file = upload_lib.download_slp_to_file(env, key)
  with timers['parse_slp']:
    try:
      game = parser.get_slp(slp_file.name)
      pq_bytes = parsing_utils.convert_game(
        game=game,
        compression=compression,
        **kwargs)
    finally:
      os.remove(slp_file.name)

  with timers['put_object']:
    upload_lib.bucket.put_object(
        Key=upload_lib.s3_path(env, dataset, key),
        Body=pq_bytes,
    )

  result = dict(
      key=key,
      size=len(pq_bytes),
      compression=compression.value,
  )

  timing = {k: t.duration for k, t in timers.items()}
  return result, timing

ProcessResult = Tuple[dict, Optional[dict]]

def process_slp_safe(
    env: str,
    key: str,
    **kwargs,
) -> ProcessResult:
  reason = None
  try:
    result, timings = process_slp(env, key, **kwargs)
  except Exception as e:
    # catches AssertionError, MemoryError, InvalidGameError
    reason = f'{type(e)}: {e}'

  if reason is not None:
    result = dict(
        key=key,
        failed=True,
        reason=reason,
    )
    timings = None
  else:
    result.update(failed=False)

  return result, timings


def process_many(env: str, keys: List[str], **kwargs) -> Iterator[ProcessResult]:
  return (process_slp_safe(env, key, **kwargs) for key in keys)

process_slp_ray = ray.remote(
    scheduling_strategy="SPREAD",
    resources=dict(workers=1),
  )(process_slp_safe)

def process_many_ray(
    env: str, keys: List[str], **kwargs
) -> Iterator[ProcessResult]:
  object_refs: List[ray.ObjectRef] = []
  futures: Dict[concurrent.futures.Future, str] = {}
  for key in keys:
    obj_ref = process_slp_ray.remote(env, key, **kwargs)
    object_refs.append(obj_ref)
    futures[obj_ref.future()] = key

  yield from monitor(futures, log_interval=10)

  # possibly a lot of memory, may want to chunk?
  # but that might not actually avoid going through the object store
  # could break into chunks and send off with a ray.remote


def print_timings(timings: List[Optional[dict]]):
  timings = list(filter(None, timings))

  if not timings:
    return

  for key in timings[0]:
    values = [t[key] for t in timings]
    total = sum(values)
    mean = total / len(values)
    print(f'{key}: total={total:.1f} mean={mean:.2f}')


def process_all(
    env: str,
    dataset: str = upload_lib.PQ,
    wipe: bool = False,
    serial: bool = False,
    **process_kwargs,
):
  """Process all slippi files.

  Parquet files are written to env/dataset/key.
  The env-dataset db has per-file info such as parse failures.
  The env-dataset_meta db has metadata about this processing run.
  """
  start_time = time.perf_counter()

  dataset_meta_db = upload_lib.get_db(env, upload_lib.DATASET_META)
  parsed_db = upload_lib.get_db(env, dataset)

  if wipe:
    dataset_meta_db.delete_many({'name': dataset})
    parsed_db.delete_many({})

  # Check compression types.
  prev_dataset_meta = dataset_meta_db.find_one({'name': dataset})
  if prev_dataset_meta is not None:
    new = process_kwargs['compression'].value
    old = prev_dataset_meta['compression']
    if new != old:
      raise ValueError(f'Mixing compression types {new} and {old}')

  # only parse valid training replays
  metas = get_singles_info(env)

  all_keys = [meta['key'] for meta in metas]
  existing = set(doc['key'] for doc in parsed_db.find({}, ['key']))

  to_process = [key for key in all_keys if key not in existing]
  skipped = len(all_keys) - len(to_process)

  if not to_process:
    print('No files to process.')
    return

  print(f'Processing {len(to_process)} slp files.')

  process_fn = process_many if serial else process_many_ray
  results_iter = process_fn(
    env, to_process, dataset=dataset, **process_kwargs)

  results_and_timings = []
  with upload_lib.Timer('parsed_db.insert_many'):
    while True:
      num_remaining = len(to_process) - len(results_and_timings)
      if not num_remaining:
        break

      chunk_size = math.ceil(num_remaining / 2)
      chunk = list(itertools.islice(results_iter, chunk_size))
      results_and_timings.extend(chunk)

      results_chunk, _ = zip(*chunk)
      parsed_db.insert_many(results_chunk)

  _, timings = zip(*results_and_timings)

  meta_data = dict(
      name=dataset,
      parser=process_kwargs['parser'].value,
      compression=process_kwargs['compression'].value,
      compression_level=process_kwargs['compression_level'],
  )
  dataset_meta_db.insert_one(meta_data)

  # done with real work
  run_time = time.perf_counter() - start_time

  print_timings(timings)
  num_processd = len(to_process)
  rate = num_processd / run_time
  print(
      f'gen: {num_processd}, skip: {skipped}, '
      f'time: {run_time:.0f}, rate: {rate:.1f}')


def main(_):
  if CLUSTER.value:
    ray.init('auto')

  process_all(
      env=ENV.value,
      wipe=WIPE.value,
      serial=SERIAL.value,
      dataset=DATASET.value,
      parser=PARSER.value,
      compression=COMPRESSION.value,
      compression_level=COMPRESSION_LEVEL.value,
  )

if __name__ == '__main__':
  ENV = flags.DEFINE_string('env', 'test', 'production environment')
  WIPE = flags.DEFINE_bool('wipe', False, 'Wipe existing metadata')
  SERIAL = flags.DEFINE_bool('local', False, 'Run serially instead of in parallel with ray.')
  CLUSTER = flags.DEFINE_bool('cluster', True, 'Run in a ray cluster.')
  DATASET = flags.DEFINE_string('dataset', upload_lib.PQ, 'Dataset name.')
  PARSER = flags.DEFINE_enum_class('parser', Parser.LIBMELEE, Parser, 'Which parser to use.')
  COMPRESSION = flags.DEFINE_enum_class(
      name='compression',
      default=CompressionType.ZLIB,  # best one
      enum_class=CompressionType,
      help='Type of compression to use.')
  COMPRESSION_LEVEL = flags.DEFINE_integer('compression_level', None, 'Compression level.')

  app.run(main)
