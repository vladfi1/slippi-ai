from pathlib import Path

from absl import app, flags

from slippi_db import upgrade_slp

INPUT = flags.DEFINE_string('input', None, 'Input archive or directory containing original .slp files.', required=True)
OUTPUT = flags.DEFINE_string('output', None, 'Output archive or directory containing upgraded .slpz files.', required=True)
UPDATED_OUTPUT_DIR = flags.DEFINE_string('updated_output_dir', None, 'If provided, write updated archives to this directory instead of overwriting originals.')
NUM_THREADS = flags.DEFINE_integer('threads', 1, 'Number of threads to use.')
CHECK_METADATA = flags.DEFINE_boolean('check_metadata', False, 'If true, verify metadata was copied correctly after each copy.')
DEBUG = flags.DEFINE_boolean('debug', False, 'Whether to run in debug mode.')
DRY_RUN = flags.DEFINE_boolean('dry_run', False, 'If true, do not modify files, just log what would be done.')


def main(_):
  input_path = Path(INPUT.value)
  output_path = Path(OUTPUT.value)

  if input_path.is_file() and output_path.is_file():
    if not str(input_path).endswith('.zip'):
      raise ValueError(f'Input file must be a .zip file: {input_path}')
    if not str(output_path).endswith('.zip'):
      raise ValueError(f'Output file must be a .zip file: {output_path}')
    dest_archive = None
    if UPDATED_OUTPUT_DIR.value:
      dest_archive = str(Path(UPDATED_OUTPUT_DIR.value) / output_path.name)
    upgrade_slp.update_slp_metadata_in_archive(
        input_archive=str(input_path),
        output_archive=str(output_path),
        num_threads=NUM_THREADS.value,
        check_metadata=CHECK_METADATA.value,
        dest_archive=dest_archive,
        debug=DEBUG.value,
        dry_run=DRY_RUN.value,
    )
  elif input_path.is_dir() and output_path.is_dir():
    # Find matching archives in both directories by relative path
    input_zips = {f.relative_to(input_path): f for f in input_path.rglob('*.zip')}
    output_zips = {f.relative_to(output_path): f for f in output_path.rglob('*.zip')}

    # Find common archives by relative path
    common_relpaths = set(input_zips.keys()) & set(output_zips.keys())
    if not common_relpaths:
      print('No matching archives found between input and output directories')
      return

    # Sort by output file size (smallest first)
    archive_pairs = [
        (input_zips[relpath], output_zips[relpath], relpath)
        for relpath in common_relpaths
    ]
    archive_pairs.sort(key=lambda p: p[1].stat().st_size)

    print(f'Found {len(archive_pairs)} matching archives to process')

    for input_zip, output_zip, relpath in archive_pairs:
      print(f'Processing {relpath}')
      dest_archive = None
      if UPDATED_OUTPUT_DIR.value:
        dest_archive = str(Path(UPDATED_OUTPUT_DIR.value) / relpath)
      upgrade_slp.update_slp_metadata_in_archive(
          input_archive=str(input_zip),
          output_archive=str(output_zip),
          num_threads=NUM_THREADS.value,
          check_metadata=CHECK_METADATA.value,
          dest_archive=dest_archive,
          debug=DEBUG.value,
          dry_run=DRY_RUN.value,
      )
  else:
    raise ValueError(
        f'Input and output must both be files or both be directories. '
        f'Got input={input_path} (is_file={input_path.is_file()}, is_dir={input_path.is_dir()}), '
        f'output={output_path} (is_file={output_path.is_file()}, is_dir={output_path.is_dir()})'
    )


if __name__ == '__main__':
  app.run(main)
