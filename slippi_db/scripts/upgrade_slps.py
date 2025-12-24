import json
from pathlib import Path

from absl import app, flags

from slippi_db import upgrade_slp

DOLPHIN = flags.DEFINE_string('dolphin', None, 'Path to Dolphin executable.', required=True)
SSBM_ISO = flags.DEFINE_string('iso', None, 'Path to SSBM ISO file.', required=True)
VERSION = flags.DEFINE_string('version', '3.18.0', 'Expected slippi version after upgrading.')

INPUT = flags.DEFINE_string('input', None, 'Input archive or directory to convert.', required=True)
OUTPUT = flags.DEFINE_string('output', None, 'Output archive or directory to write.', required=True)
NUM_THREADS = flags.DEFINE_integer('threads', 1, 'Number of threads to use for conversion.')
CHECK_SAME_PARSE = flags.DEFINE_bool('check_same_parse', True, 'Check if the replay has the same parse as the original.')
WORK_DIR = flags.DEFINE_string('work_dir', None, 'Optional working directory for temporary files.')
IN_MEMORY = flags.DEFINE_bool('in_memory', True, 'Use in-memory temporary files for conversion.')
LOG_INTERVAL = flags.DEFINE_integer('log_interval', 30, 'Interval in seconds to log progress during conversion.')
CHECK_IF_NEEDED = flags.DEFINE_bool('check_if_needed', False, 'Check if the file needs conversion before processing.')
DOLPHIN_TIMEOUT = flags.DEFINE_integer('dolphin_timeout', 60, 'Dolphin timeout in seconds.')

SKIP_EXISTING = flags.DEFINE_boolean('skip_existing', False, 'Whether to skip existing output archives.')
REMOVE_INPUT = flags.DEFINE_boolean('remove_input', False, 'Whether to remove the input file after conversion.')
DEBUG = flags.DEFINE_boolean('debug', False, 'Whether to run in debug mode.')

def process_single_archive(input_path, output_path, dolphin_config):
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
  )


def main(_):
  dolphin_config = upgrade_slp.DolphinConfig(
      dolphin_path=DOLPHIN.value,
      ssbm_iso_path=SSBM_ISO.value,
  )

  input_path = Path(INPUT.value)
  output_path = Path(OUTPUT.value)

  if input_path.is_file():
    # Process single file
    if not str(input_path).endswith('.zip'):
      raise ValueError(f'Input file must be a .zip file: {input_path}')
    results = process_single_archive(str(input_path), str(output_path), dolphin_config)

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
    print(f'Found {len(zip_files)} zip files to process')

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
      results = process_single_archive(str(zip_file), str(output_file), dolphin_config)

      for result in results:
        json_results.append((
            str(zip_file), result.local_file.name,
            result.error, result.skipped))

    if skipped:
      print(f'Skipped {len(skipped)} files that already exist in output:')
      for path in skipped:
        print(f'  {path}')

  else:
    raise ValueError(f'Input path does not exist: {input_path}')

  with open('upgrade_results.json', 'w') as f:
      json.dump(json_results, f, indent=2)

if __name__ == '__main__':
  app.run(main)
