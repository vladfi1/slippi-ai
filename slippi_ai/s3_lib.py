import functools
import os
import typing

import boto3
from simplekv.net.boto3store import Boto3Store

from pymongo import MongoClient
import gridfs

class S3Keys(typing.NamedTuple):
  params: str
  saved_model: str

def get_keys(tag: str):
  return S3Keys(
      params=f"slippi-ai.params.{tag}",
      saved_model=f"slippi-ai.experiments.{tag}.saved_model",
  )

@functools.lru_cache()
def get_store(s3_creds: str = None) -> Boto3Store:
  s3_creds = s3_creds or os.environ['S3_CREDS']
  access_key, secret_key = s3_creds.split(':')

  session = boto3.Session(access_key, secret_key)
  s3 = session.resource('s3')
  bucket = s3.Bucket('slippi-data')
  store = Boto3Store(bucket)
  return store

@functools.lru_cache()
def get_sacred_db(mongo_uri: str = None):
  client = MongoClient(mongo_uri or os.environ['MONGO_URI'])
  return client.sacred

def id_to_tag(_id: int) -> str:
  db = get_sacred_db()
  # TODO: assert that there's only one matching document?
  doc = db.runs.find_one({'_id': _id}, ['config.tag'])
  return doc['config']['tag']

def delete_params(tag: str = None, _id: int = None):
  tag = tag or id_to_tag(_id)
  store = get_store()
  for key in get_keys(tag):
    store.delete(key)

def delete_experiment(experiment_id, db = None):
  db = db or get_sacred_db()
  fs = gridfs.GridFS(db)

  ex = db.runs.find_one(
    dict(_id=experiment_id),
    ['artifacts', 'info.metrics', 'config.tag'])

  # Delete all artifacts (file outcomes)
  for artifact in ex['artifacts']:
    fs.delete(artifact['file_id'])
    if fs.exists(artifact['file_id']):
      raise(RuntimeError('Failed to delete artifact, {}'.format(artifact)))

  # Delete all metrics
  for metric in ex['info']['metrics']:
    db.metrics.delete_one(dict(_id=metric['id']))

  # Delete experiment
  db.runs.delete_one(dict(_id=experiment_id))

  # Delete parameters in s3
  delete_params(ex['config']['tag'])
