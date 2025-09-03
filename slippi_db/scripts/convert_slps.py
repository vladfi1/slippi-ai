#!/usr/bin/env python3
"""Convert .slp files in a zip archive or directory to .slpp.gz format using the slp tool.

This script takes a .zip archive or directory containing .slp files and outputs a new .zip
archive or directory with each .slp file converted to .slpp.gz format using the `slp` tool
with gzip compression.

Usage:
  Single archive: python slippi_db/scripts/convert_slps.py --input input.zip --output output.zip [--threads N]
  Directory: python slippi_db/scripts/convert_slps.py --input input_dir/ --output output_dir/ [--threads N]
"""

import concurrent.futures
import enum
import gzip
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Tuple, Optional

from absl import app, flags
import tqdm

from slippi_db import utils

class OutputType(enum.Enum):
    SLPP_GZ = "slpp.gz"
    SLPZ = "slpz"

FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, 'Input zip file or directory containing .slp files', required=True)
flags.DEFINE_string('output', None, 'Output zip file or directory for converted files', required=True)
flags.DEFINE_integer('threads', 1, 'Number of threads to use')
flags.DEFINE_integer('limit', None, 'Limit number of files to process (for testing)')
flags.DEFINE_boolean('remove_input', False, 'Whether to remove successful files from input archive after conversion')
flags.DEFINE_enum_class('output_type', OutputType.SLPZ, OutputType, 'Output type for conversion')

def convert_slp_to_slpp_gz(
    zip_file: utils.ZipFile,
    output_path: str,
) -> Optional[str]:
  """Convert a single .slp file from zip archive to .slpp.gz format.

  Args:
    zip_path: Path to the zip archive containing the .slp file
    slp_filename: Name of the .slp file within the zip archive
    output_path: Path where the .slpp.gz file should be written

  Returns:
    True if conversion was successful, False otherwise
  """
  # Get shared memory tmp dir for intermediate files
  shm_tmpdir = utils.get_tmp_dir(in_memory=True)

  try:
    # Extract .slp file from zip to shared memory
    with tempfile.TemporaryDirectory(dir=shm_tmpdir) as temp_dir:
      slp_path = os.path.join(temp_dir, 'slp')

      # TODO: unzip to shared memory directly
      with open(slp_path, 'wb') as temp_slp:
        temp_slp.write(zip_file.read())

      # Convert .slp to .slpp using the slp tool (also in shared memory)
      slpp_path = os.path.join(temp_dir, 'slpp')
      subprocess.run([
          'slp',
          '-o', slpp_path,
          '-f', 'peppi',
          slp_path
      ], capture_output=True, text=True, check=True)

      # Read the .slpp data and compress with gzip
      with open(slpp_path, 'rb') as f:
        slpp_data = f.read()

      compressed_data = gzip.compress(slpp_data)

      # Write compressed data to final output path (not in shared memory)
      with open(output_path, 'wb') as f:
        f.write(compressed_data)

      return None

  except subprocess.CalledProcessError as e:
    # print(f"Failed to convert {slp_filename}: {e.stderr}")
    return e.stderr.decode().strip()
  except Exception as e:
    # print(f"Error converting {slp_filename}: {e}")
    return str(e)


def convert_slp_to_slpz(
    zip_file: utils.ZipFile,
    output_path: str,
) -> Optional[str]:
  try:
    subprocess.run(
        ['slpz', '-x', '-', '-o', output_path],
        input=zip_file.read(),
        check=True,
        capture_output=True,
    )
  except subprocess.CalledProcessError as e:
    return e.stderr.decode().strip()
  except utils.FileReadException as e:
    return repr(e)

conversion_functions = {
    OutputType.SLPP_GZ: convert_slp_to_slpp_gz,
    OutputType.SLPZ: convert_slp_to_slpz,
}

def process_file(
    zip_file: utils.ZipFile,
    output_path: str,
    output_type: OutputType,
) -> Tuple[str, Optional[str]]:
  """Process a single file for multithreading."""
  error = conversion_functions[output_type](zip_file, output_path)
  return zip_file.path, error

def convert_zip_archive(
    input_zip_path: str,
    output_zip_path: str,
    output_type: OutputType,
    num_threads: int = 1,
    limit: Optional[int] = None,
    remove_input: bool = False,
):
  """Convert all .slp files in a zip archive to .slpp.gz format.

  Args:
    input_zip_path: Path to input zip file containing .slp files
    output_zip_path: Path to output zip file for .slpp.gz files
    num_threads: Number of threads to use for parallel processing
    limit: Maximum number of files to process (for testing)
    remove_input: Whether to remove successful files from input archive
  """
  print(f"Converting {input_zip_path} -> {output_zip_path}")
  print(f"Using {num_threads} thread{'s' if num_threads != 1 else ''}")

  # Check for existing files in output archive
  existing_files = []
  if os.path.exists(output_zip_path):
    existing_files = utils.traverse_slp_files_zip(output_zip_path)
    print(f"Found {len(existing_files)} existing files in output archive")

  existing_names = set(f.name for f in existing_files)

  to_remove: list[str] = []

  # Create temporary directory for output files (not in shared memory)
  with tempfile.TemporaryDirectory() as temp_output_dir:
    todo: list[tuple[utils.ZipFile, str]] = []
    skipped_count = 0
    output_dirs = set()  # Track output directories to create
    files = utils.traverse_slp_files_zip(input_zip_path)
    for f in files:

      # Skip if file already exists in output archive
      if f.name in existing_names:
        skipped_count += 1
        to_remove.append(f.path)
        continue

      output_name = f.base_name + '.' + output_type.value
      output_path = os.path.join(temp_output_dir, output_name)
      output_dirs.add(os.path.dirname(output_path))
      todo.append((f, output_path))

    # Create output directories
    for output_dir in output_dirs:
      os.makedirs(output_dir, exist_ok=True)

    if not todo and skipped_count == 0:
      print("No .slp files found in input archive")
      return

    print(f"Found {len(files)} .slp files in archive")
    if skipped_count > 0:
      print(f"Skipping {skipped_count} files already present in output archive")

    # Apply limit if specified
    if limit is not None and limit > 0:
      todo = todo[:limit]
      print(f"Limited to first {len(todo)} files")

    results: list[tuple[str, Optional[str]]] = []

    # Process files
    if num_threads == 1:
      # Single-threaded processing
      for zip_file, output_path in tqdm.tqdm(todo, desc="Converting", unit="file"):
        result = process_file(zip_file, output_path, output_type)
        results.append(result)
    else:
      # Multi-threaded processing
      with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        try:
          futures = [
              executor.submit(process_file, zip_file, output_path, output_type)
              for zip_file, output_path in todo
          ]

          for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Converting",
            unit="file",
            smoothing=0,
          ):
            result = future.result()
            results.append(result)
        except KeyboardInterrupt:
          print('KeyboardInterrupt, shutting down')
          executor.shutdown(cancel_futures=True)
          raise

    # Count successful conversions
    successful_conversions = sum(1 for _, error in results if error is None)
    failed_conversions = len(results) - successful_conversions

    print(f"Conversion complete: {successful_conversions} successful, {failed_conversions} failed")

    # Create output zip using command line zip (uncompressed)
    if successful_conversions > 0:
      print("Creating output archive...")
      result = subprocess.run([
        'zip', '-r', '-0', output_zip_path, '.'
      ], cwd=temp_output_dir, capture_output=True, text=True)

      if result.returncode != 0:
        raise RuntimeError(f"Failed to create output zip: {result.stderr}")

      print(f"Output saved to: {output_zip_path}")

    # Handle input removal if requested
    if remove_input:
      if failed_conversions == 0:
        print(f"Removing input archive: {input_zip_path}")
        os.remove(input_zip_path)
      else:
        # Remove only successful files from the archive
        to_remove.extend(filename for filename, error in results if error is None)

        if to_remove:
          print(f"Removing {len(to_remove)} successful files from input archive at {input_zip_path}")
          utils.delete_from_zip(input_zip_path, to_remove)

    if failed_conversions > 0:
      print(f"Some files failed to convert: {failed_conversions} errors")
      slp_filename, error = results[0]
      print(f'Example error in {slp_filename}: {error}')

def main(_):
  input_path = Path(FLAGS.input)
  output_path = Path(FLAGS.output)

  # Validate input exists
  if not input_path.exists():
    print(f"Error: Input path '{input_path}' does not exist")
    return 1

  if input_path.is_file():
    # Process single file
    if not str(input_path).endswith('.zip'):
      print(f"Error: Input file must be a .zip file: {input_path}")
      return 1

    # Validate output directory exists for single file
    output_dir = output_path.parent
    if output_dir and not output_dir.exists():
      print(f"Error: Output directory '{output_dir}' does not exist")
      return 1

    try:
      convert_zip_archive(
          input_zip_path=str(input_path),
          output_zip_path=str(output_path),
          output_type=FLAGS.output_type,
          num_threads=FLAGS.threads,
          limit=FLAGS.limit,
          remove_input=FLAGS.remove_input,
      )
      return 0
    except Exception as e:
      print(f"Error: {e}")
      return 1

  elif input_path.is_dir():
    # Process directory recursively
    if not output_path.exists():
      output_path.mkdir(parents=True, exist_ok=True)
    elif not output_path.is_dir():
      print(f"Error: Output must be a directory when input is a directory: {output_path}")
      return 1

    # Find all zip files recursively
    zip_files = list(input_path.rglob('*.zip'))
    print(f"Found {len(zip_files)} zip files to process")

    if not zip_files:
      print("No zip files found in input directory")
      return 0

    for zip_file in zip_files:
      # Calculate relative path from input directory
      rel_path = zip_file.relative_to(input_path)

      # Create output path maintaining directory structure
      output_file = output_path / rel_path
      output_file.parent.mkdir(parents=True, exist_ok=True)

      print(f"\nProcessing {zip_file} -> {output_file}")

      try:
        convert_zip_archive(
            input_zip_path=str(zip_file),
            output_zip_path=str(output_file),
            output_type=FLAGS.output_type,
            num_threads=FLAGS.threads,
            limit=FLAGS.limit,
            remove_input=FLAGS.remove_input,
        )
      except Exception as e:
        print(f"Error processing {zip_file}: {e}")
        continue

    return 0
  else:
    print(f"Error: Input path is neither a file nor a directory: {input_path}")
    return 1


if __name__ == "__main__":
  app.run(main)
