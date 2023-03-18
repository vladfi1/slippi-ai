import io
import tempfile

from absl import app
from absl import flags
import dropbox
import dropbox.files

from slippi_db.secrets import SECRETS
from slippi_db import upload_lib

_ENV = flags.DEFINE_string('env', 'test', 'production environment')
_TO_DISK = flags.DEFINE_boolean('to_disk', False, 'Use disk instead of RAM.')
_DRY_RUN = flags.DEFINE_bool('dry_run', False, 'Don\'t upload anything.')

dbx = dropbox.Dropbox(SECRETS['DBOX_KEY'])

def dbx_to_file(path: str) -> tempfile._TemporaryFileWrapper:
  tmp = tempfile.NamedTemporaryFile()

  with upload_lib.Timer('dbx.download_to_file'):
    dbx.files_download_to_file(tmp.name, path)

  tmp.seek(0)
  return tmp

def dbx_to_ram(path: str) -> io.BytesIO:
  with upload_lib.Timer('dbx.download'):
    metadata, response = dbx.files_download(path)
    return io.BytesIO(response.content)

def is_supported(name: str):
  return name.endswith('.zip') or name.endswith('.7z')

def upload_to_raw(env: str, to_disk: bool = False):
  db = upload_lib.ReplayDB(env)
  dbox_uploads = db.raw.find({'dropbox_id': {'$exists': True}}, ['dropbox_id'])
  dbox_ids = set(doc['dropbox_id'] for doc in dbox_uploads)

  results: dropbox.files.ListFolderResult
  results = dbx.files_list_folder('/SlippiDump')

  for entry in results.entries:
    entry: dropbox.files.FileMetadata
    name = entry.name

    if not is_supported(name):
      print(f'skipping unsupported upload {name}')
      continue

    if entry.id in dbox_ids:
      print(f'Already uploaded: {name}')
      continue

    if entry.size > 100 * 10 ** 9:
      print(f'{name} of size {entry.size} too large!')
      continue
      # raise ValueError(f'File {entry.name} of size {entry.size} too large!')

    print(f'Uploading "{name}" of size {entry.size:.2e}.')
    if _DRY_RUN.value:
      continue

    if to_disk:
      tmp = dbx_to_file(entry.path_display)
      data = tmp
    else:
      data = dbx_to_ram(entry.path_display)

    with upload_lib.Timer('db.upload_raw'):
      result = db.upload_raw(
          name=entry.name,
          f=data,
          description=entry.path_display,
          check_max_size=False,
          dropbox_id=entry.id,
          dropbox_hash=entry.content_hash,
      )
      print(result)

    if to_disk:
      tmp.close()

  # TODO: handle this case
  assert not results.has_more  # max 1000

# For multi-part uploads, see
# https://gist.github.com/barbietunnie/d670d5601151129cbc02fbac3800e399

def main(_):
  upload_to_raw(_ENV.value, to_disk=_TO_DISK.value)

if __name__ == '__main__':
  app.run(main)
