from pathlib import Path

from absl import app, flags

from slippi_db import upgrade_slp

INPUT = flags.DEFINE_string('input', None, 'Input archive or directory to process.', required=True)
NUM_THREADS = flags.DEFINE_integer('threads', 1, 'Number of threads to use.')
DEBUG = flags.DEFINE_boolean('debug', False, 'Whether to run in debug mode.')
DRY_RUN = flags.DEFINE_boolean('dry_run', False, 'If true, do not delete files, just log what would be done.')


def main(_):
  input_path = Path(INPUT.value)

  if input_path.is_file():
    if not str(input_path).endswith('.zip'):
      raise ValueError(f'Input file must be a .zip file: {input_path}')
    upgrade_slp.delete_non_upgraded(
        archive_path=str(input_path),
        num_threads=NUM_THREADS.value,
        debug=DEBUG.value,
        dry_run=DRY_RUN.value,
    )
  elif input_path.is_dir():
    zip_files = list(input_path.rglob('*.zip'))
    zip_files.sort(key=lambda f: f.stat().st_size)
    print(f'Found {len(zip_files)} zip files to process')

    for zip_file in zip_files:
      print(f'Processing {zip_file}')
      upgrade_slp.delete_non_upgraded(
          archive_path=str(zip_file),
          num_threads=NUM_THREADS.value,
          debug=DEBUG.value,
          dry_run=DRY_RUN.value,
      )
  else:
    raise ValueError(f'Input path does not exist: {input_path}')


if __name__ == '__main__':
  app.run(main)
