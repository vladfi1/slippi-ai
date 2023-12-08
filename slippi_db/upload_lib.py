import functools
import hashlib
import io
import os
import tempfile
import time
import zlib
import zipfile
from typing import Any, BinaryIO, NamedTuple, Optional

import boto3
from simplekv.net.boto3store import Boto3Store
from pymongo import MongoClient

from slippi_db.secrets import SECRETS

MB = 10 ** 6

DEFAULTS = dict(
  max_size_per_file=10 * MB,
  min_size_per_file=1 * MB,
  max_files=100,
  max_total_size=250 * MB,
)

# controls where stuff is stored
ENV = os.environ.get('SLIPPI_DB_ENV', 'test')

Stage = str
RAW: Stage = 'raw'  # uploaded zipped directories
SLP: Stage = 'slp'  # individual slippi files
META: Stage = 'meta'  # per-file metadata
DATASET_META: Stage = 'data_meta'  # metadata for various dataset runs
PQ: Stage = 'pq'  # parquet files, one per slp


class S3(NamedTuple):
  session: boto3.Session
  resource: Any
  bucket: Any  # the type here is difficult to specify
  store: Boto3Store

def make_s3() -> S3:
  s3_creds = SECRETS['S3_CREDS']
  access_key, secret_key = s3_creds.split(':')
  session = boto3.Session(access_key, secret_key)
  resource = session.resource('s3')
  bucket = resource.Bucket('slp-replays')
  store = Boto3Store(bucket)
  return S3(session, resource, bucket, store)

s3 = make_s3()

def get_objects(env, stage):
  get_key = lambda path: path.split('/')[2]
  paths = s3.store.iter_keys(prefix=f'{env}/{stage}/')
  return {get_key(path): s3.bucket.Object(path) for path in paths}

def s3_path(env: str, stage: Stage, key: str):
  return '/'.join([env, stage, key])

@functools.lru_cache()
def get_main_db(mongo_uri=None):
  mongo_uri = mongo_uri or SECRETS['MONGO_URI']
  client = MongoClient(mongo_uri)
  return client.slp_replays

@functools.lru_cache()
def get_db(env: str, stage: str):
  # assert stage in (RAW, SLP, META)
  return get_main_db().get_collection(env + '-' + stage)

def get_params(env: str) -> dict:
  params_coll = get_main_db().params
  found = params_coll.find_one({'env': env})
  if found is None:
    # update params collection
    params = dict(env=env, **DEFAULTS)
    params_coll.insert_one(params)
    return params
  # update found with default params
  for k, v in DEFAULTS.items():
    if k not in found:
      found[k] = v
  return found

def create_params(env: str, **kwargs):
  assert get_main_db().params.find_one({'env': env}) is None
  params = dict(env=env, **DEFAULTS)
  params.update(kwargs)
  get_main_db().params.insert_one(params)

class Timer:

  def __init__(self, name: str, verbose=True):
    self.name = name
    self.verbose = verbose

  def __enter__(self):
    self.start = time.perf_counter()

  def __exit__(self, *_):
    self.duration = time.perf_counter() - self.start
    if self.verbose:
      print(f'{self.name}: {self.duration:.1f}')

def iter_bytes(f: BinaryIO, chunk_size=2 ** 16):
  while True:
    chunk = f.read(chunk_size)
    if chunk:
      yield chunk
    else:
      break
  f.seek(0)

class ReplayDB:

  def __init__(self, env: str = ENV):
    self.env = env
    self.params = get_params(env)
    self.raw = get_main_db().get_collection(env + '-raw')

  def raw_size(self) -> int:
    total_size = 0
    for doc in self.raw.find():
      total_size += doc['stored_size']
    return total_size

  @property
  def max_file_size(self):
    return self.params['max_size_per_file']

  @property
  def min_file_size(self):
    return self.params['min_size_per_file']

  @property
  def max_files(self):
    return self.params['max_files']

  def max_db_size(self):
    return self.params['max_total_size']

  def upload_slp(self, name: str, content: bytes) -> Optional[str]:
    # max_files = params['max_files']
    # if coll.count_documents({}) >= max_files:
    #   return f'DB full, already have {max_files} uploads.'
    if not name.endswith('.slp'):
      return f'{name}: not a .slp'

    max_size = self.params['max_size_per_file']
    if len(content) > max_size:
      return f'{name}: exceeds {max_size} bytes.'
    min_size = self.params['min_size_per_file']
    if len(content) < min_size:
      return f'{name}: must have {min_size} bytes.'

    digest = hashlib.sha256()
    digest.update(content)
    key = digest.hexdigest()

    found = self.raw.find_one({'key': key})
    if found is not None:
      return f'{name}: duplicate file'

    # TODO: validate that file conforms to .slp spec

    # store file in S3
    compressed_bytes = zlib.compress(content)
    s3.store.put(self.name + '.' + key, compressed_bytes)

    # update DB
    self.raw.insert_one(dict(
      key=key,
      name=name,
      type='slp',
      compressed=True,
      original_size=len(content),
      stored_size=len(compressed_bytes),
    ))

    return None

  def upload_zip(self, uploaded):
    errors = []
    with zipfile.ZipFile(uploaded) as zip:
      names = zip.namelist()
      names = [n for n in names if n.endswith('.slp')]
      print(names)

      max_files = self.params['max_files']
      num_uploaded = self.raw.count_documents({})
      if num_uploaded + len(names) > max_files:
        return f'Can\'t upload {len(names)} files, would exceed limit of {max_files}.'

      for name in names:
        with zip.open(name) as f:
          error = self.upload_slp(name, f.read())
          if error:
            errors.append(error)

    uploaded.close()
    if errors:
      return '\n'.join(errors)
    return f'Successfully uploaded {len(names)} files.'

  def upload_raw(
    self,
    name: str,
    f: BinaryIO,
    obj_type: Optional[str] = None,
    check_max_size: bool = True,
    **metadata,
  ):
    if obj_type is None:
      obj_type = name.split('.')[-1]

    size = f.seek(0, 2)
    f.seek(0)

    if check_max_size:
      max_bytes_left = self.max_db_size() - self.raw_size()
      if size > max_bytes_left:
        return f'{name}: exceeds {max_bytes_left} bytes'

    with Timer('md5'):
      digest = hashlib.md5()
      for chunk in iter_bytes(f):
        digest.update(chunk)
      key = digest.hexdigest()

    found = self.raw.find_one({'key': key})
    if found is not None:
      return f'{name}: object with md5={key} already uploaded'

    with Timer('upload_fileobj'):
      s3.bucket.upload_fileobj(
          Fileobj=f,
          Key=s3_path(self.env, RAW, key),
          # ContentLength=size,
          # ContentMD5=str(base64.encodebytes(digest.digest())),
      )
      # store.put_file(self.name + '/raw/' + key, f)

    # update DB
    self.raw.insert_one(dict(
        filename=name,
        key=key,
        hash_method="md5",
        type=obj_type,
        stored_size=size,
        processed=False,
        **metadata,
    ))
    return f'{name}: upload successful'

  def delete(self, key: str):
    s3_key = self.env + '/raw/' + key
    s3.bucket.delete_objects(Delete=dict(Objects=[dict(Key=s3_key)]))
    # store.delete()
    self.raw.delete_one({'key': key})

def delete_keys(keys: list[str]):
  objects = [dict(Key=k) for k in keys]
  if not objects:
    print('No objects to delete.')
    return
  return s3.bucket.delete_objects(Delete=dict(Objects=objects))

def nuke_replays(env: str, stage: str):
  get_main_db().drop_collection(env + '-' + stage)
  get_main_db().params.delete_many({'env': env + '-' + stage})
  keys = s3.store.iter_keys(prefix=f'{env}/{stage}/')
  response = delete_keys(keys)
  print(f'Deleted {len(response["Deleted"])} objects.')

def nuke_stages(env: str):
  for stage in [RAW, SLP, META, DATASET_META, PQ]:
    nuke_replays(env, stage)

def remove_processed(env: str, dry_run: bool = False):
  """Removes processed raw uploads."""
  raw = get_main_db().get_collection(env + '-' + RAW)
  to_remove = list(raw.find({'processed': True}))
  by_key = {info['key']: info for info in to_remove}

  total_size = sum(info['stored_size'] for info in to_remove)
  print(f'Deleting {len(by_key)} raw objects of size {total_size}.')
  if dry_run:
    print('Dry run, quitting.')
    return

  s3_paths = {f'{env}/{RAW}/{key}': key for key in by_key}
  response = delete_keys(s3_paths)

  deleted_keys = [s3_path[obj['Key']] for obj in response['Deleted']]
  deleted_filenames = [by_key[key]['filename'] for key in deleted_keys]
  deleted_total_size = sum(by_key[key]['stored_size'] for key in deleted_keys)

  print(f'Deleted {len(delete_keys)} raw objects from env "{env}".')
  print('Filenames:', deleted_filenames)
  print('Keys:', list(deleted_keys))
  print('Total Size:', deleted_total_size)

shm_dir = '/dev/shm'
bucket = s3.bucket

def download_slp(env: str, key: str) -> bytes:
  compressed_bytes = io.BytesIO()

  slp_s3_path = f'{env}/slp/{key}'
  # with upload_lib.Timer(f"download {raw_name}", verbose=False):
  bucket.download_fileobj(slp_s3_path, compressed_bytes)

  return zlib.decompress(compressed_bytes.getvalue())

def to_file(data: bytes) -> BinaryIO:
  tmp_file = tempfile.NamedTemporaryFile(dir=shm_dir, delete=False)
  tmp_file.write(data)
  tmp_file.close()
  return tmp_file

def download_slp_to_file(env: str, key: str) -> BinaryIO:
  slp_bytes = download_slp(env, key)
  return to_file(slp_bytes)

def download_slp_locally(env: str, key: str, dir='.') -> str:
  slp_bytes = download_slp(env, key)
  slp_db = get_db(env, 'slp')
  doc = slp_db.find_one({'key': key})
  path = os.path.join(dir, doc['filename'].replace('/', '_'))
  with open(path, 'wb') as f:
    f.write(slp_bytes)
  print(f'Downloaded slp to {path}')
  return path
