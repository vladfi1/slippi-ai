"""Decompress raw uploads."""

import hashlib
import os
import subprocess
import tempfile
from typing import Iterator, Set, Tuple
import zipfile
import zlib

import pathlib

from slippi_db import upload_lib

bucket = upload_lib.s3.bucket

SUPPORTED_TYPES = ('zip', '7z')

def _md5(b: bytes) -> str:
  return hashlib.md5(b).hexdigest()

def iter_zip(f) -> Iterator[Tuple[str, bytes]]:
  zf = zipfile.ZipFile(f)

  for info in zf.infolist():
    if not info.filename.endswith('.slp'):
      continue

    with upload_lib.Timer("zf.read"):
      data = zf.read(info)
    
    yield info.filename, data

def iter_7z(path: str) -> Iterator[Tuple[str, bytes]]:
  # py7zr is a bit broken so we instead extract to the local filesystem

  tmpdir = tempfile.TemporaryDirectory()

  with upload_lib.Timer("7z x"):
    subprocess.run(['7z', 'x', path, '-o' + tmpdir.name])

  for dirpath, dirnames, filenames in os.walk(tmpdir.name):
    dir = pathlib.Path(dirpath)
    for name in filenames:
      if name.endswith('.slp'):
        with open(dir / name, 'rb') as f:
          yield name, f.read()

  tmpdir.cleanup()

def process_raw(env: str, raw_key: str):
  with upload_lib.Timer("raw_db"):
    raw_db = upload_lib.get_db(env, 'raw')
    raw_info = raw_db.find_one({'key': raw_key})
    obj_type = raw_info['type']
    if obj_type not in ('zip', '7z'):
      raise ValueError('Unsupported obj_type={obj_type}.')

  slp_db = upload_lib.get_db(env, 'slp')
  slp_keys = set(doc["key"] for doc in slp_db.find({}, ["key"]))

  tmp = tempfile.NamedTemporaryFile()
  raw_s3_path = f'{env}/raw/{raw_key}'
  with upload_lib.Timer("download_fileobj"):
    bucket.download_fileobj(raw_s3_path, tmp)

  tmp.seek(0)

  if obj_type == 'zip':
    data = iter_zip(tmp)
  elif obj_type == '7z':
    data = iter_7z(tmp.name)

  for filename, slp_bytes in data:
    upload_slp(
        env=env,
        raw_key=raw_key,
        filename=filename,
        slp_bytes=slp_bytes, 
        slp_keys=slp_keys,
    )

  # TODO: set processed flag in raw_db

def upload_slp(
  env: str,
  raw_key: str,
  filename: str,
  slp_bytes: bytes,
  slp_keys: Set[str],
) -> bool:
  slp_key = _md5(slp_bytes)

  if slp_key in slp_keys:
    print('Duplicate slp with key', slp_key)
    return False
  slp_keys.add(slp_key)

  with upload_lib.Timer("zlib.compress"):
    compressed_slp_bytes = zlib.compress(slp_bytes)

  slp_s3_path = f'{env}/slp/{slp_key}'

  with upload_lib.Timer("put_object"):
    bucket.put_object(
        Key=slp_s3_path,
        Body=compressed_slp_bytes)
  
  slp_db = upload_lib.get_db(env, 'slp')
  slp_db.insert_one(dict(
      filename=filename,
      compression='zlib',
      key=slp_key,
      raw_key=raw_key,
      original_size=len(slp_bytes),
      stored_size=len(compressed_slp_bytes),
  ))

def process_all(env: str):
  raw_db = upload_lib.get_db(env, 'raw')
  raw_info = raw_db.find({})
  for doc in raw_info:
    if doc['type'] not in SUPPORTED_TYPES:
      print(f'Skipping {doc["filename"]} ({doc["description"]}).')
      continue
    process_raw(env, doc['key'])

if __name__ == '__main__':
  process_all('test')
