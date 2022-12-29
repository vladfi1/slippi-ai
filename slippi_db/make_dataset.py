"""Create tar file with parsed parquet files and metadata DataFrame.

ray submit --start slippi_db/submit_cluster slippi_db/make_dataset.py
"""

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

    parses = list(upload_lib.get_db(env, dataset).find())
    for info in parses:
        if info['failed'] and info['key'] in key_to_meta:
            del key_to_meta[info['key']]

    return pd.DataFrame(map(_to_row, key_to_meta.values()))

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

def make_tar_and_df(env: str, dataset: str):
    """Creates the dataset (tar) and metadata (df).
    
    Dataset tar is saved to env/dataset/tar.
    Metadata df is saved to env/dataset/meta.
    """

    df = make_df(env, dataset)

    df_file = tempfile.NamedTemporaryFile()
    df.to_parquet(df_file.name)

    upload_lib.bucket.upload_file(
        Filename=df_file.name,
        Key='/'.join([env, 'datasets', dataset, 'meta.pq']),
    )

    # probably happens automatically when python object is destroyed?
    df_file.close()

    job = s3_tar.S3Tar(
        source_bucket='slp-replays',
        target_key='/'.join([env, 'datasets', dataset, 'games.tar']),
    )

    for key in df['key']:
        job.add_file(upload_lib.s3_path(env, dataset, key))

    job.tar()

def main(_):
    make_tar_and_df(ENV.value, DATASET.value)

if __name__ == '__main__':
    app.run(main)
