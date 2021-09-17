import os
import typing

import boto3
from simplekv.net.boto3store import Boto3Store

class S3Keys(typing.NamedTuple):
  params: str
  saved_model: str

def get_keys(tag: str):
  return S3Keys(
      params=f"slippi-ai.params.{tag}",
      saved_model=f"slippi-ai.experiments.{tag}.saved_model",
  )

def get_store(s3_creds: str = None):
  s3_creds = s3_creds or os.environ['S3_CREDS']
  access_key, secret_key = s3_creds.split(':')

  session = boto3.Session(access_key, secret_key)
  s3 = session.resource('s3')
  bucket = s3.Bucket('slippi-data')
  store = Boto3Store(bucket)
  return store
