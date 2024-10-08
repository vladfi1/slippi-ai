"""Decompress raw uploads and upload individually compressed slp files."""

from concurrent import futures
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
from typing import Callable, Iterator, List, NamedTuple, Set, Tuple
import zlib

from absl import app
from absl import flags

from slippi_db import upload_lib, fix_zip, utils
from slippi_db.utils import monitor


flags.DEFINE_string('env', 'test', 'production environment')
flags.DEFINE_bool('mp', False, 'Run in parallel with multiprocessing.')
flags.DEFINE_bool('in_memory', True, 'Use ram instead of disk.')
flags.DEFINE_bool(
    'processed', False, 'Decompress already-processed uploads.')
DRY_RUN = flags.DEFINE_bool('dry_run', False, 'Don\'t upload anything.')

FLAGS = flags.FLAGS

bucket = upload_lib.s3.bucket

SUPPORTED_TYPES = ('zip', '7z')

def upload_slp(
    env: str,
    file: utils.LocalFile,
    slp_key: str,
) -> Tuple[dict, dict]:
  timers = {
      k: upload_lib.Timer(k, verbose=False)
      for k in ['read_slp', 'zlib.compress', 'put_object']
  }

  with timers['read_slp']:
    slp_bytes = file.read()

  with timers['zlib.compress']:
    compressed_slp_bytes = zlib.compress(slp_bytes)

  slp_s3_path = f'{env}/slp/{slp_key}'

  with timers['put_object']:
    bucket.put_object(
        Key=slp_s3_path,
        Body=compressed_slp_bytes)

  result = dict(
      # filenames may have bogus unicode characters
      filename=file.name.encode('utf-8', 'ignore'),
      compression='zlib',
      key=slp_key,
      original_size=len(slp_bytes),
      stored_size=len(compressed_slp_bytes),
  )

  timing = {k: t.duration for k, t in timers.items()}
  return result, timing

def upload_slp_mp(kwargs: dict):
  return upload_slp(**kwargs)

class UploadResults(NamedTuple):
  uploads: List[dict]
  duplicates: List[str]  # slp keys
  timings: List[dict]

  @staticmethod
  def empty() -> 'UploadResults':
    return UploadResults([], [], [])

def filter_duplicate_slp(
    files: list[utils.LocalFile],
    slp_keys: Set[str],
    duplicates: List[str],
) -> Iterator[Tuple[utils.LocalFile, str]]:
  for file in files:
    slp_key = file.md5()

    if slp_key in slp_keys:
      # print('Duplicate slp with key', slp_key)
      duplicates.append(slp_key)
      continue

    slp_keys.add(slp_key)
    yield file, slp_key

def _file_md5(file: utils.LocalFile) -> str:
  return file.md5()

def filter_duplicate_slp_mp(
    files: list[utils.LocalFile],
    slp_keys: Set[str],
    duplicates: List[str],
) -> List[Tuple[utils.LocalFile, str]]:
  context = mp.get_context('forkserver')
  with futures.ProcessPoolExecutor(mp_context=context) as pool:
    md5_futures = [pool.submit(_file_md5, file) for file in files]
    futures_dict = {future: file.name for future, file in zip(md5_futures, files)}
    for _ in monitor(futures_dict, log_interval=30):
      pass
    local_slp_keys = [future.result() for future in md5_futures]

  files_and_keys = []
  for file, slp_key in zip(files, local_slp_keys):
    if slp_key in slp_keys:
      # print('Duplicate slp with key', slp_key)
      duplicates.append(slp_key)
      continue

    slp_keys.add(slp_key)
    files_and_keys.append((file, slp_key))

  return files_and_keys

def upload_files(
  env: str,
  files: list[utils.LocalFile],
  slp_keys: Set[str],
) -> UploadResults:
  results = UploadResults.empty()

  for file, slp_key in filter_duplicate_slp(
      files, slp_keys, results.duplicates):

    result, timing = upload_slp(
        env=env,
        file=file,
        slp_key=slp_key,
    )
    results.uploads.append(result)
    results.timings.append(timing)

  return results

def upload_files_mp(
  env: str,
  files: list[utils.LocalFile],
  slp_keys: Set[str],
) -> UploadResults:
  duplicates = []

  with upload_lib.Timer('filter_duplicate_slp_mp'):
    files_and_keys = filter_duplicate_slp_mp(files, slp_keys, duplicates)

  context = mp.get_context('forkserver')
  with futures.ProcessPoolExecutor(mp_context=context) as pool:
    upload_futures = [
        pool.submit(upload_slp, env=env, file=file, slp_key=slp_key)
        for file, slp_key in files_and_keys
    ]
    futures_dict = {
        future: file.name for future, (file, _) in
        zip(upload_futures, files_and_keys)
    }
    uploads = list(monitor(futures_dict, log_interval=30))

  if uploads:
    results, timings = zip(*uploads)
  else:
    results, timings = [], []

  return UploadResults(results, duplicates, timings)

# TODO: use a Protocol
Uploader = Callable[[str, list[utils.LocalFile], Set[str]], UploadResults]

# TODO: handle duplicates when calling multiple process_raw in parallel
def process_raw(
    env: str,
    raw_key: str,
    uploader: Uploader = upload_files_mp,
    skip_processed: bool = True,
    in_memory: bool = False,
    dry_run: bool = False,
) -> UploadResults:
  start_time = time.perf_counter()

  raw_db = upload_lib.get_db(env, 'raw')
  raw_info = raw_db.find_one({'key': raw_key})
  obj_type = raw_info['type']
  if obj_type not in ('zip', '7z'):
    raise ValueError('Unsupported obj_type={obj_type}.')

  raw_name = raw_info["filename"]
  if skip_processed and raw_info.get('processed', False):
    print(f'Skipping already-processed raw upload: {raw_name}')
    return UploadResults.empty()
  else:
    print(f'Processing {raw_name}')

  if dry_run:
    return UploadResults.empty()

  slp_db = upload_lib.get_db(env, 'slp')
  slp_keys = set(doc["key"] for doc in slp_db.find({}, ["key"]))

  tmp_dir = utils.get_tmp_dir(in_memory)

  # download raw file
  raw_file = tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False)
  raw_s3_path = f'{env}/raw/{raw_key}'
  with upload_lib.Timer(f"download {raw_name}"):
    bucket.download_fileobj(raw_s3_path, raw_file)
  raw_file.close()

  if obj_type == 'zip':
    unzip_dir = tempfile.TemporaryDirectory(dir=tmp_dir)

    try:
      if obj_type == 'zip':
        if raw_info['stored_size'] >= 2 ** 32:
          # Fix zip64 files generated on Windows.
          fix_zip.fix_zip(raw_file.name)
        utils.extract_zip(raw_file.name, unzip_dir.name)
      elif obj_type == '7z':
        utils.extract_7z(raw_file.name, unzip_dir.name)
    except subprocess.CalledProcessError as e:
      raise RuntimeError(f'Extracting {raw_name} ({raw_key}) failed') from e
    finally:
      os.remove(raw_file.name)

    files = utils.traverse_slp_files(unzip_dir.name)
    cleanup = unzip_dir.cleanup
  elif obj_type == '7z':
    # Don't extract the 7z file to save on space.
    files = utils.traverse_slp_files_7z(raw_file.name)
    cleanup = lambda: os.remove(raw_file.name)

  try:
    results = uploader(env, files, slp_keys)
  finally:
    cleanup()

  for d in results.uploads:
    d.update(raw_key=raw_key)
  if results.uploads:
    try:
      slp_db.insert_many(results.uploads)
    except UnicodeError as e:
      raise RuntimeError(f'DB upload failed for {raw_name} ({raw_key})') from e

  # set processed flag in raw_db
  duplicates = ','.join(results.duplicates)
  raw_db.update_one({'key': raw_key}, {'$set': {'processed': True, 'duplicates': duplicates}})
  # TODO: consider deleting raw file from s3 after it has been processed

  num_uploaded = len(results.uploads)
  num_duplicates = len(results.duplicates)
  num_processed = num_uploaded + num_duplicates

  run_time = time.perf_counter() - start_time
  fps = num_processed / run_time
  print(f'Stats for {raw_name}:')
  print(f'Processed {num_processed} slippi replays in {run_time:.1f} seconds.')
  print(f'Files per second: {fps:.2f}')
  print(f'Uploaded {num_uploaded} with {num_duplicates} duplicates.')

  return results

def print_timings(results: List[UploadResults]):
  timings = []
  for result in results:
    timings.extend(result.timings)

  if not timings:
    return

  for key in timings[0]:
    values = [t[key] for t in timings]
    total = sum(values)
    mean = total / len(values)
    print(f'{key}: total={total:.1f} mean={mean:.2f}')

def process_all(
  env: str,
  uploader: Uploader,
  in_memory: bool,
  skip_processed: bool = True,
  dry_run: bool = False,
) -> List[UploadResults]:
  results = []

  raw_db = upload_lib.get_db(env, 'raw')
  # TODO: skip processed here?
  raw_info = raw_db.find({})
  for doc in raw_info:
    if doc['type'] not in SUPPORTED_TYPES:
      print(f'Skipping {doc["filename"]} ({doc["description"]}).')
      continue
    result = process_raw(
        env=env,
        raw_key=doc['key'],
        uploader=uploader,
        in_memory=in_memory,
        skip_processed=skip_processed,
        dry_run=dry_run,
      )
    results.append(result)

  print_timings(results)
  return results

def main(_):
  process_all(
      env=FLAGS.env,
      uploader=upload_files_mp if FLAGS.mp else upload_files,
      in_memory=FLAGS.in_memory,
      skip_processed=not FLAGS.processed,
      dry_run=DRY_RUN.value,
  )

if __name__ == '__main__':
  app.run(main)
