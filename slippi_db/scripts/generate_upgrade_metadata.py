"""Populate upgrades.sqlite from existing Raw/Upgraded archive pairs."""

from pathlib import Path

from absl import app, flags

from slippi_db import upgrade_slp

ROOT = flags.DEFINE_string('root', None, 'Dataset root directory (contains Raw/, Upgraded/, upgrades.sqlite).', required=True)
NUM_THREADS = flags.DEFINE_integer('threads', 1, 'Number of threads to use.')


def main(_):
  root = Path(ROOT.value)
  raw_dir = root / 'Raw'
  upgraded_dir = root / 'Upgraded'
  db_path = root / 'upgrades.sqlite'

  if not raw_dir.is_dir():
    raise FileNotFoundError(f'Raw directory does not exist: {raw_dir}')

  db_conn = upgrade_slp.create_upgrades_db(str(db_path))

  input_zips = sorted(raw_dir.rglob('*.zip'))
  print(f'Found {len(input_zips)} zip files in {raw_dir}')

  for input_zip in input_zips:
    rel_path = input_zip.relative_to(raw_dir)
    output_zip = upgraded_dir / rel_path

    print(f'\nProcessing {rel_path}')
    upgrade_slp.check_existing_upgrades(
      input_archive=str(input_zip),
      output_archive=str(output_zip),
      archive_name=str(rel_path),
      db_conn=db_conn,
      num_threads=NUM_THREADS.value,
    )

  db_conn.close()


if __name__ == '__main__':
  app.run(main)
