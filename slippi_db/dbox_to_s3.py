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

dbx = dropbox.Dropbox(SECRETS['DBOX_KEY'])

def dbx_to_file(path: str) -> io.BytesIO:
  tmp = tempfile.NamedTemporaryFile()

  with upload_lib.Timer('dbx.download_to_file'):
    dbx.files_download_to_file(tmp.name, path)

  tmp.seek(0)
  return tmp

def dbx_to_ram(path: str) -> io.BytesIO:
  with upload_lib.Timer('dbx.download'):
    metadata, response = dbx.files_download(path)
    return io.BytesIO(response.content)

def upload_to_raw(env: str, to_disk: bool = False):
  db = upload_lib.ReplayDB(env)
  dbox_uploads = db.raw.find({'dropbox_id': {'$exists': True}}, ['dropbox_id'])
  dbox_ids = set(doc['dropbox_id'] for doc in dbox_uploads)

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

    if to_disk:
      data = dbx_to_file(entry.path_display)
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

  # TODO: handle this case
  assert not results.has_more  # max 1000

# from https://gist.github.com/barbietunnie/d670d5601151129cbc02fbac3800e399
def upload_to_dbx(
    access_token,
    file_path,
    target_path,
    timeout=900,
    chunk_size=4 * 1024 * 1024,
):
    dbx = dropbox.Dropbox(access_token, timeout=timeout)
    with open(file_path, "rb") as f:
        file_size = os.path.getsize(file_path)
        chunk_size = 4 * 1024 * 1024
        if file_size <= chunk_size:
            print(dbx.files_upload(f.read(), target_path))
        else:
            with tqdm(total=file_size, desc="Uploaded") as pbar:
                upload_session_start_result = dbx.files_upload_session_start(
                    f.read(chunk_size)
                )
                pbar.update(chunk_size)
                cursor = dropbox.files.UploadSessionCursor(
                    session_id=upload_session_start_result.session_id,
                    offset=f.tell(),
                )
                commit = dropbox.files.CommitInfo(path=target_path)
                while f.tell() < file_size:
                    if (file_size - f.tell()) <= chunk_size:
                        print(
                            dbx.files_upload_session_finish(
                                f.read(chunk_size), cursor, commit
                            )
                        )
                    else:
                        dbx.files_upload_session_append(
                            f.read(chunk_size),
                            cursor.session_id,
                            cursor.offset,
                        )
                        cursor.offset = f.tell()
                    pbar.update(chunk_size)


def main(_):
  upload_to_raw(_ENV.value, to_disk=_TO_DISK.value)

if __name__ == '__main__':
  app.run(main)
