import logging
import json
from pathlib import Path
import shutil

from absl import app, flags

from slippi_db import upgrade_slp

DOLPHIN = flags.DEFINE_string('dolphin', None, 'Path to Dolphin executable.', required=True)
SSBM_ISO = flags.DEFINE_string('iso', None, 'Path to SSBM ISO file.', required=True)
VERSION = flags.DEFINE_string('version', '3.18.0', 'Expected slippi version after upgrading.')

ROOT = flags.DEFINE_string('root', None, 'Dataset root directory (contains Raw/, Upgraded/, upgrades.sqlite). Sets --input, --output, and --db.')
INPUT = flags.DEFINE_string('input', None, 'Input archive or directory to convert.')
OUTPUT = flags.DEFINE_string('output', None, 'Output archive or directory to write.')
NUM_THREADS = flags.DEFINE_integer('threads', 1, 'Number of threads to use for conversion.')
CHECK_SAME_PARSE = flags.DEFINE_bool('check_same_parse', True, 'Check if the replay has the same parse as the original.')
WORK_DIR = flags.DEFINE_string('work_dir', None, 'Optional working directory for temporary files.')
IN_MEMORY = flags.DEFINE_bool('in_memory', True, 'Use in-memory temporary files for conversion.')
LOG_INTERVAL = flags.DEFINE_integer('log_interval', 30, 'Interval in seconds to log progress during conversion.')
CHECK_IF_NEEDED = flags.DEFINE_bool('check_if_needed', True, 'Check if the file needs conversion before processing.')
DOLPHIN_TIMEOUT = flags.DEFINE_integer(
    'dolphin_timeout', 60, 'Dolphin timeout in seconds. If the upgrade process takes longer, it is considered a failure.')

SKIP_EXISTING = flags.DEFINE_boolean(
    'skip_existing', True,
    'Whether to skip existing output archives. Files that already exist in the '
    'output archive will not be overwritten.')
REMOVE_INPUT = flags.DEFINE_boolean('remove_input', False, 'Whether to remove the input file after conversion.')
DEBUG = flags.DEFINE_boolean('debug', False, 'Whether to run in debug mode.')
DRY_RUN = flags.DEFINE_boolean('dry_run', False, 'If true, do not perform any conversions, just log what would be done.')

DB = flags.DEFINE_string('db', None, 'Path to upgrades.sqlite database. If provided, uses DB to skip already-processed files and writes results back.')
RETRY_ERRORS = flags.DEFINE_boolean('retry_errors', False, 'If true, re-attempt files that previously failed (only used with --db).')

def process_single_archive(input_path, output_path, dolphin_config, db_conn=None, archive_name=None):
  """Process a single archive file."""
  return upgrade_slp.upgrade_archive(
      input_path=input_path,
      output_path=output_path,
      dolphin_config=dolphin_config,
      expected_version=tuple(map(int, VERSION.value.split('.'))),
      in_memory=IN_MEMORY.value,
      num_threads=NUM_THREADS.value,
      check_same_parse=CHECK_SAME_PARSE.value,
      work_dir=WORK_DIR.value,
      log_interval=LOG_INTERVAL.value,
      check_if_needed=CHECK_IF_NEEDED.value,
      remove_input=REMOVE_INPUT.value,
      dolphin_timeout=DOLPHIN_TIMEOUT.value,
      debug=DEBUG.value,
      db_conn=db_conn,
      archive_name=archive_name,
      retry_errors=RETRY_ERRORS.value,
      dry_run=DRY_RUN.value,
  )


def main(_):
  # Check that slpz and copy_slp_metadata are available
  for command in ['slpz', 'copy_slp_metadata']:
    if shutil.which(command) is None:
      raise RuntimeError(f'Required command not found in PATH: {command}')

  dolphin_config = upgrade_slp.DolphinConfig(
      dolphin_path=DOLPHIN.value,
      ssbm_iso_path=SSBM_ISO.value,
  )

  # Resolve paths from --root or explicit flags
  if ROOT.value is not None:
    root = Path(ROOT.value)
    input_val = INPUT.value or str(root / 'Raw')
    output_val = OUTPUT.value or str(root / 'Upgraded')
    db_val = DB.value or str(root / 'upgrades.sqlite')
  else:
    if INPUT.value is None or OUTPUT.value is None:
      raise ValueError('Either --root or both --input and --output must be specified.')
    input_val = INPUT.value
    output_val = OUTPUT.value
    db_val = DB.value

  db_conn = None
  if db_val is not None:
    db_conn = upgrade_slp.create_upgrades_db(db_val)

  try:
    input_path = Path(input_val)
    output_path = Path(output_val)

    if input_path.is_file():
      # Process single file
      if not str(input_path).endswith('.zip'):
        raise ValueError(f'Input file must be a .zip file: {input_path}')

      archive_name = input_path.name if db_conn is not None else None
      results = process_single_archive(
          str(input_path), str(output_path), dolphin_config,
          db_conn=db_conn, archive_name=archive_name)

      json_results = [
          (result.local_file.name, result.error, result.skipped)
          for result in results
      ]
    elif input_path.is_dir():
      # Process directory recursively
      if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
      elif not output_path.is_dir():
        raise ValueError(f'Output must be a directory when input is a directory: {output_path}')

      # Find all zip files recursively
      zip_files = list(input_path.rglob('*.zip'))
      logging.info(f'Found {len(zip_files)} zip files to process')

      json_results = []
      skipped = []

      for zip_file in zip_files:
        # Calculate relative path from input directory
        rel_path = zip_file.relative_to(input_path)

        # Create output path maintaining directory structure
        output_file = output_path / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if SKIP_EXISTING.value and output_file.exists():
          skipped.append(str(zip_file))
          continue

        print(f'Processing {zip_file} -> {output_file}')

        archive_name = str(rel_path) if db_conn is not None else None
        results = process_single_archive(
            str(zip_file), str(output_file), dolphin_config,
            db_conn=db_conn, archive_name=archive_name)

        for result in results:
          json_results.append((
              str(zip_file), result.local_file.name,
              result.error, result.skipped))

      if skipped:
        print(f'Skipped {len(skipped)} files that already exist in output:')

    else:
      raise ValueError(f'Input path does not exist: {input_path}')

    with open('upgrade_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
  finally:
    if db_conn is not None:
      db_conn.close()

if __name__ == '__main__':
  app.run(main)
