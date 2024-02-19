import functools
import os
import typing

import boto3
from simplekv.net.boto3store import Boto3Store


class S3Keys(typing.NamedTuple):
  combined: str  # currently used for both config and params
  # saved_model: str  # no longer used

SEP = "."
S3_PREFIX = SEP.join(["slippi-ai", "experiments"])

def get_keys(tag: str) -> S3Keys:
  keys = {key: SEP.join([S3_PREFIX, tag, key]) for key in S3Keys._fields}
  return S3Keys(**keys)

@functools.lru_cache()
def get_store(s3_creds: str = None) -> Boto3Store:
  s3_creds = s3_creds or os.environ['S3_CREDS']
  access_key, secret_key = s3_creds.split(':')

  session = boto3.Session(access_key, secret_key)
  s3 = session.resource('s3')
  bucket = s3.Bucket('slippi-data')
  store = Boto3Store(bucket)
  return store

def delete_params(tag: str = None):
  # TODO: delete using wandb run
  store = get_store()
  for key in get_keys(tag):
    store.delete(key)
