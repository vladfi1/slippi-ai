"""Add missing .gz suffix to files in an archive."""

import zipfile

from absl import app, flags

from slippi_db import utils

flags.DEFINE_string('input', None, 'Input zip file containing .slp files', required=True)

def add_gz_suffix(zip_path: str) -> None:
  """Add .gz suffix to all files in the zip archive that do not have it."""
  to_rename = []

  with zipfile.ZipFile(zip_path, 'a') as zf:
    for zip_info in zf.filelist:
      if zip_info.is_dir():
        continue

      assert zip_info.filename.endswith('.slp'), f'File {zip_info.filename} is not a .slp file'

      new_filename = zip_info.filename + '.gz'
      to_rename.append((zip_info.filename, new_filename))

  utils.rename_within_zip(zip_path, to_rename)

def main(_):
  input_path = flags.FLAGS.input
  if not input_path.endswith('.zip'):
    raise ValueError(f'Input path must be a .zip file: {input_path}')

  add_gz_suffix(input_path)
  print(f'Added .gz suffix to files in {input_path}')

if __name__ == '__main__':
  app.run(main)
