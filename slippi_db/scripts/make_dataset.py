r"""Create tar file with parsed parquet files and metadata DataFrame.

ray submit --start slippi_db/submit_cluster.yaml \
  slippi_db/scripts/make_dataset.py --env test
"""

import logging
import io
import tarfile
import tempfile
from typing import Iterable

from absl import app
from absl import flags
import pandas as pd
from tqdm import tqdm

import s3_tar

from slippi_db import upload_lib
from slippi_db.test_peppi import get_singles_info

ENV = flags.DEFINE_string('env', 'test', 'production environment')
DATASET = flags.DEFINE_string('dataset', upload_lib.PQ, 'Dataset name.')
ONLY_META = flags.DEFINE_bool('only_meta', False, 'Only generate metadata.')

def _to_row(game: dict) -> dict:
  row = game.copy()
  del row['players']
  for i, player in enumerate(game['players']):
    for k, v in player.items():
      row[f'p{i}.{k}'] = v
  return row

def make_df(env: str, dataset: str) -> pd.DataFrame:
  # gather keys with valid metadata and successful parses

  metas = get_singles_info(env)
  key_to_meta = {meta['key']: meta for meta in metas}
  keys = set(key_to_meta)

  parses = list(upload_lib.get_db(env, dataset).find({}, ['key', 'failed']))

  parsed_keys = set(info['key'] for info in parses)
  old_size = len(keys)
  keys.intersection_update(parsed_keys)
  dropped = old_size - len(keys)
  if dropped > 0:
    logging.warn(
      f'Dropped {dropped} keys that were never parsed!'
      ' Maybe you forgot to run parsing?')

  failed_keys = set(info['key'] for info in parses if info['failed'])
  old_size = len(keys)
  keys -= failed_keys
  dropped = old_size - len(keys)
  logging.info(f'Dropped {dropped} keys with parse failures.')

  if not keys:
    raise ValueError('No replays left. Maybe all parses failed?')

  valid_metas = [meta for key, meta in key_to_meta.items() if key in keys]
  return pd.DataFrame(map(_to_row, valid_metas))

def make_tar(env: str, dataset: str, keys: Iterable[str]):
  print(f'Creating tar for "{dataset}" out of {len(keys)} games.')

  tmp = tempfile.NamedTemporaryFile()
  tar = tarfile.TarFile(fileobj=tmp.file, mode='w')

  for key in tqdm(keys):
    s3_path = upload_lib.s3_path(env, dataset, key)
    game_bytes = io.BytesIO()
    upload_lib.bucket.download_fileobj(s3_path, game_bytes)
    game_size = game_bytes.tell()
    game_bytes.seek(0)

    tarinfo = tarfile.TarInfo(key)
    tarinfo.size = game_size
    tar.addfile(tarinfo, game_bytes)
  
  tar.close()
  return tmp

def make_tar_and_df(env: str, dataset: str, make_tar: bool = True):
  """Creates the dataset (tar) and metadata (df).
  
  Dataset tar is saved to env/dataset/tar.
  Metadata df is saved to env/dataset/meta.
  """

  with upload_lib.Timer('make_df'):
    df = make_df(env, dataset)

  if make_tar:
    print(f'Creating dataset with {len(df)} replays.')

    with upload_lib.Timer('tar'):
      job = s3_tar.S3Tar(
        source_bucket='slp-replays',
        target_key='/'.join([env, 'datasets', dataset, 'games.tar']),
      )

      for key in df['key']:
        job.add_file(upload_lib.s3_path(env, dataset, key))

      job.tar()

  # write metadata once games.tar is done
  with upload_lib.Timer('write df'):
    df_file = tempfile.NamedTemporaryFile()
    df.to_parquet(df_file.name)

    upload_lib.bucket.upload_file(
      Filename=df_file.name,
      Key='/'.join([env, 'datasets', dataset, 'meta.pq']),
    )

    # probably happens automatically when python object is destroyed?
    df_file.close()


def main(_):
  make_tar_and_df(
      ENV.value, DATASET.value,
      make_tar=not ONLY_META.value)

if __name__ == '__main__':
  app.run(main)
