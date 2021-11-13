import tempfile

from absl import app
from absl import flags
import dropbox
import dropbox.files

from slippi_db.secrets import SECRETS
from slippi_db import upload_lib

_ENV = flags.DEFINE_string('env', 'test', 'production environment')

def upload_to_raw(env: str):
  db = upload_lib.ReplayDB(env)
  dbox_uploads = db.raw.find({'dropbox_id': {'$exists': True}}, ['dropbox_id'])
  dbox_ids = set(doc['dropbox_id'] for doc in dbox_uploads)

  dbx = dropbox.Dropbox(SECRETS['DBOX_KEY'])
  results: dropbox.files.ListFolderResult
  results = dbx.files_list_folder('/SlippiDump')

  for entry in results.entries:
    entry: dropbox.files.FileMetadata
    name = entry.name

    if entry.id in dbox_ids:
      print(f'{name} already uploaded to s3.')
      continue

    if entry.size > 100 * 10 ** 9:
      print(f'{name} of size {entry.size} too large!')
      continue
      # raise ValueError(f'File {entry.name} of size {entry.size} too large!')

    print(f'Uploading "{name}" of size {entry.size}.')

    tmp = tempfile.NamedTemporaryFile()

    with upload_lib.Timer('dbx.download'):
      dbx.files_download_to_file(tmp.name, entry.path_display)

    with upload_lib.Timer('db.upload_raw'):
      tmp.seek(0)
      result = db.upload_raw(
          name=entry.name,
          f=tmp,
          description=entry.path_display,
          check_max_size=False,
          dropbox_id=entry.id,
          dropbox_hash=entry.content_hash,
      )
      print(result)
    
    tmp.close()

  assert not results.has_more  # max 1000

def main(_):
  upload_to_raw(_ENV.value)

if __name__ == '__main__':
  app.run(main)
