"""Decompress raw uploads and upload individually compressed slp files."""

import hashlib
import multiprocessing
import os
import subprocess
import tempfile
import time
from typing import Callable, Iterator, List, NamedTuple, Set, Tuple
import zlib

from absl import app
from absl import flags

from slippi_db import upload_lib

flags.DEFINE_string('env', 'test', 'production environment')
flags.DEFINE_bool('mp', False, 'Run in parallel with multiprocessing.')
flags.DEFINE_bool('in_memory', True, 'Use ram instead of disk.')
flags.DEFINE_bool(
    'processed', False, 'Decompress already-processed uploads.')

FLAGS = flags.FLAGS

bucket = upload_lib.s3.bucket

SUPPORTED_TYPES = ('zip', '7z')

def _md5(b: bytes) -> str:
  return hashlib.md5(b).hexdigest()

def _tmp_dir(in_memory: bool):
  return '/dev/shm' if in_memory else None

Files = List[Tuple[str, str]]

def extract_zip(src: str, dst_dir: str):
  with upload_lib.Timer("unzip"):
    subprocess.check_call(
        ['unzip', src],
        cwd=dst_dir,
        stdout=subprocess.DEVNULL)

def extract_7z(src: str, dst_dir: str):
  with upload_lib.Timer("7z x"):
    subprocess.check_call(
        ['7z', 'x', src, '-o' + dst_dir],
        stdout=subprocess.DEVNULL)

def traverse_slp_files(dir: str) -> Files:
  files = []
  for dirpath, _, filenames in os.walk(dir):
    for name in filenames:
      if name.endswith('.slp'):
        files.append((name, os.path.join(dirpath, name)))
  return files

def upload_slp(
    env: str,
    path: str,
    slp_key: str,
    filename: str,
) -> Tuple[dict, dict]:
  timers = {
      k: upload_lib.Timer(k, verbose=False)
      for k in ['read_slp', 'zlib.compress', 'put_object']
  }

  with timers['read_slp']:
    with open(path, 'rb') as f:
      slp_bytes = f.read()

  with timers['zlib.compress']:
    compressed_slp_bytes = zlib.compress(slp_bytes)

  slp_s3_path = f'{env}/slp/{slp_key}'

  with timers['put_object']:
    bucket.put_object(
        Key=slp_s3_path,
        Body=compressed_slp_bytes)

  result = dict(
      filename=filename,
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
  duplicates: List[str]
  timings: List[dict]

  @staticmethod
  def empty() -> 'UploadResults':
    return UploadResults([], [], [])

def _md5_from_file(path: str) -> str:
  with open(path, 'rb') as f:
    slp_bytes = f.read()
  return _md5(slp_bytes)

def filter_duplicate_slp(
    files: Files,
    slp_keys: Set[str],
    duplicates: List[str],
) -> Iterator[Tuple[str, str, str]]:
  for filename, path in files:
    slp_key = _md5_from_file(path)

    if slp_key in slp_keys:
      # print('Duplicate slp with key', slp_key)
      duplicates.append(filename)
      continue

    slp_keys.add(slp_key)
    yield filename, path, slp_key

def filter_duplicate_slp_mp(
    files: Files,
    slp_keys: Set[str],
    duplicates: List[str],
) -> Iterator[Tuple[str, str, str]]:
  names, paths = zip(*files)

  pool = multiprocessing.Pool()
  local_slp_keys = pool.map(_md5_from_file, paths, chunksize=16)

  for filename, path, slp_key in zip(
      names, paths, local_slp_keys):
    if slp_key in slp_keys:
      # print('Duplicate slp with key', slp_key)
      duplicates.append(filename)
      continue

    slp_keys.add(slp_key)
    yield filename, path, slp_key

def upload_files(
  env: str,
  files: Files,
  slp_keys: Set[str],
) -> UploadResults:
  results = UploadResults.empty()

  for filename, path, slp_key in filter_duplicate_slp(
      files, slp_keys, results.duplicates):

    result, timing = upload_slp(
        env=env,
        path=path,
        slp_key=slp_key,
        filename=filename,
    )
    results.uploads.append(result)
    results.timings.append(timing)

  return results

def upload_files_mp(
  env: str,
  files: Files,
  slp_keys: Set[str],
) -> UploadResults:
  pool = multiprocessing.Pool()
  duplicates = []

  files = filter_duplicate_slp_mp(files, slp_keys, duplicates)

  def to_kwargs(args):
    filename, path, slp_key = args
    return dict(
        env=env,
        path=path,
        slp_key=slp_key,
        filename=filename,
    )

  upload_slp_kwargs = map(to_kwargs, files)
  uploads = pool.imap_unordered(
      upload_slp_mp, upload_slp_kwargs, chunksize=32)
  uploads = list(uploads)
  if uploads:
    results, timings = zip(*uploads)
  else:
    results, timings = [], []

  return UploadResults(results, duplicates, timings)

# TODO: use a Protocol
Uploader = Callable[[str, Files, Set[str]], UploadResults]

def process_raw(
    env: str,
    raw_key: str,
    uploader: Uploader = upload_files_mp,
    skip_processed: bool = True,
    in_memory: bool = False,
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
    return []
  else:
    print(f'Processing {raw_name}')

  slp_db = upload_lib.get_db(env, 'slp')
  slp_keys = set(doc["key"] for doc in slp_db.find({}, ["key"]))

  tmp_dir = _tmp_dir(in_memory)

  # download raw file
  raw_file = tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False)
  raw_s3_path = f'{env}/raw/{raw_key}'
  with upload_lib.Timer(f"download {raw_name}"):
    bucket.download_fileobj(raw_s3_path, raw_file)
  raw_file.close()

  unzip_dir = tempfile.TemporaryDirectory(dir=tmp_dir)

  try:
    if obj_type == 'zip':
      extract_zip(raw_file.name, unzip_dir.name)
    elif obj_type == '7z':
      extract_7z(raw_file.name, unzip_dir.name)
  except subprocess.CalledProcessError as e:
    raise RuntimeError(f'Extracting {raw_name} ({raw_key}) failed') from e
  finally:
    os.remove(raw_file.name)

  files = traverse_slp_files(unzip_dir.name)
  results = uploader(env, files, slp_keys)
  unzip_dir.cleanup()

  for d in results.uploads:
    d.update(raw_key=raw_key)
  if results.uploads:
    slp_db.insert_many(results.uploads)

  # set processed flag in raw_db
  raw_db.update_one({'key': raw_key}, {'$set': {'processed': True}})

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
        skip_processed=skip_processed)
    results.append(result)

  print_timings(results)
  return results

def main(_):
  process_all(
      env=FLAGS.env,
      uploader=upload_files_mp if FLAGS.mp else upload_files,
      in_memory=FLAGS.in_memory,
      skip_processed=not FLAGS.processed)

if __name__ == '__main__':
  app.run(main)
