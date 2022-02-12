import enum
import io
import os
import time
from typing import List, Optional, Tuple
import zlib

from absl import app
from absl import flags
import pyarrow as pa
import pyarrow.parquet as pq
import ray

from slippi_db import (
    upload_lib,
    parse_libmelee,
    parse_peppi,
)

class Parser(enum.Enum):
  LIBMELEE = 'libmelee'
  PEPPI = 'peppi'

  def get_slp(self, path: str) -> pa.StructArray:
    if self == Parser.LIBMELEE:
      return parse_libmelee.get_slp(path)
    elif self == Parser.PEPPI:
      return parse_peppi.get_slp(path)

class CompressionType(enum.Enum):
  ZLIB = 'zlib'
  SNAPPY = 'snappy'
  GZIP = 'gzip'
  BROTLI = 'brotli'
  LZ4 = 'lz4'
  ZSTD = 'zstd'
  NONE = 'none'

  def for_parquet(self) -> str:
    if self is CompressionType.ZLIB:
      return CompressionType.NONE.value
    return self.value

def convert_slp(
    path: str,
    parser: Parser,
    pq_version: str = '2.4',
    compression: CompressionType = CompressionType.NONE,
    compression_level: Optional[int] = None,
) -> bytes:
  game = parser.get_slp(path)
  table = pa.Table.from_arrays([game], names=['root'])
  pq_file = io.BytesIO()

  if compression == CompressionType.ZLIB:
    pq_compression_level = None
  else:
    pq_compression_level = compression_level

  pq.write_table(
      table, pq_file,
      version=pq_version,
      compression=compression.for_parquet(),
      compression_level=pq_compression_level,
      use_dictionary=False,
  )
  pq_bytes = pq_file.getvalue()

  if compression == CompressionType.ZLIB:
    level = -1 if compression_level is None else compression_level
    pq_bytes = zlib.compress(pq_bytes, level=level)
  return pq_bytes

def process_slp(
    env: str,
    key: str,
    parser: Parser,
    dataset: str,
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
      pq_bytes = convert_slp(slp_file.name, parser, **kwargs)
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


def process_many(env: str, keys: List[str], **kwargs) -> List[ProcessResult]:
  return [process_slp_safe(env, key, **kwargs) for key in keys]

process_slp_ray = ray.remote(num_cpus=1)(process_slp_safe)

def process_many_ray(env: str, keys: List[str], **kwargs) -> List[ProcessResult]:
  futures = []
  for key in keys:
    futures.append(process_slp_ray.remote(env, key, **kwargs))
  # possibly a lot of memory, may want to chunk?
  # but that might not actually avoid going through the object store
  # could break into chunks and send off with a ray.remote
  return ray.get(futures)


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

  slp_db = upload_lib.get_db(env, upload_lib.SLP)
  meta_db = upload_lib.get_db(env, upload_lib.DATASET_META)
  parsed_db = upload_lib.get_db(env, dataset)

  if wipe:
    parsed_db.delete_many({})

  all_keys = [doc['key'] for doc in slp_db.find({}, ['key'])]
  existing = set(doc['key'] for doc in parsed_db.find({}, ['key']))

  to_process = [key for key in all_keys if key not in existing]
  skipped = len(all_keys) - len(to_process)

  if not to_process:
    print('No files to process.')
    return

  process_fn = process_many if serial else process_many_ray
  results = process_fn(env, to_process, dataset=dataset, **process_kwargs)

  results, timings = zip(*results)

  with upload_lib.Timer('parsed_db.insert_many'):
    parsed_db.insert_many(results)

  meta_db.delete_many({'name': dataset})
  meta_data = dict(
      name=dataset,
      parser=process_kwargs['parser'].value,
      compression=process_kwargs['compression'].value,
      compression_level=process_kwargs['compression_level'],
  )
  meta_db.insert_one(meta_data)

  # done with real work
  run_time = time.perf_counter() - start_time

  print_timings(timings)
  num_processd = len(to_process)
  rate = num_processd / run_time
  print(
      f'gen: {num_processd}, skip: {skipped}, '
      f'time: {run_time:.0f}, rate: {rate:.1f}')

ENV = flags.DEFINE_string('env', 'test', 'production environment')
WIPE = flags.DEFINE_bool('wipe', False, 'Wipe existing metadata')
SERIAL = flags.DEFINE_bool('local', False, 'Run serially instead of in parallel with ray.')
CLUSTER = flags.DEFINE_bool('cluster', False, 'Run in a ray cluster.')
DATASET = flags.DEFINE_string('dataset', upload_lib.PQ, 'Dataset name.')
PARSER = flags.DEFINE_enum_class('parser', Parser.LIBMELEE, Parser, 'Which parser to use.')
COMPRESSION = flags.DEFINE_enum_class(
    name='compression',
    default=CompressionType.NONE,
    enum_class=CompressionType,
    help='Type of compression to use.')
COMPRESSION_LEVEL = flags.DEFINE_integer('compression_level', None, 'Compression level.')


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
  app.run(main)
