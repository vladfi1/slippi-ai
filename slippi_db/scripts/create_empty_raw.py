"""Create empty raw archives for each slpz archive missing a raw counterpart.

Given a directory of slpz archives and a directory of raw archives, for each
.zip in the slpz directory that has no corresponding .zip in the raw directory,
creates an empty zip archive in the raw directory.

Usage:
  python slippi_db/scripts/create_empty_raw.py --raw Raw/ --slpz Slpz/
"""

import os
import zipfile

from absl import app, flags

RAW = flags.DEFINE_string('raw', None, 'Directory of raw zip archives.', required=True)
SLPZ = flags.DEFINE_string('slpz', None, 'Directory of slpz zip archives.', required=True)
DRY_RUN = flags.DEFINE_bool('dry_run', False, 'Print what would be created without creating.')


def main(_):
  raw_dir = RAW.value
  slpz_dir = SLPZ.value

  # Collect relative paths of .zip files in each directory.
  raw_zips = set()
  for dirpath, _, filenames in os.walk(raw_dir):
    reldir = os.path.relpath(dirpath, raw_dir)
    for name in filenames:
      if name.endswith('.zip'):
        raw_zips.add(os.path.join(reldir, name).removeprefix('./'))

  created = 0
  for dirpath, _, filenames in os.walk(slpz_dir):
    reldir = os.path.relpath(dirpath, slpz_dir)
    for name in filenames:
      if not name.endswith('.zip'):
        continue
      relpath = os.path.join(reldir, name).removeprefix('./')
      if relpath in raw_zips:
        continue

      raw_path = os.path.join(raw_dir, relpath)
      if DRY_RUN.value:
        print(f'Would create: {raw_path}')
      else:
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        zipfile.ZipFile(raw_path, 'w').close()
        print(f'Created: {raw_path}')
      created += 1

  print(f'{"Would create" if DRY_RUN.value else "Created"} {created} empty archive(s).')


if __name__ == '__main__':
  app.run(main)
