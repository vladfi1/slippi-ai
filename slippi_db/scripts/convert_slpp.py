#!/usr/bin/env python3
"""Convert .slp files in a zip archive to .slpp.gz format using the slp tool.

This script takes a .zip archive containing .slp files and outputs a new .zip
archive with each .slp file converted to .slpp.gz format using the `slp` tool
with gzip compression.

Usage: python slippi_db/scripts/convert_slpp.py --input input.zip --output output.zip [--threads N]
"""

import concurrent.futures
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


FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, 'Input zip file containing .slp files', required=True)
flags.DEFINE_string('output', None, 'Output zip file for .slpp.gz files', required=True)
flags.DEFINE_integer('threads', 1, 'Number of threads to use')
flags.DEFINE_integer('limit', None, 'Limit number of files to process (for testing)')
flags.DEFINE_string('failed_output', None, 'Optional zip file to store failed .slp files')


def convert_slp_to_slpp_gz(
    zip_path: str,
    slp_filename: str,
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

      result = subprocess.run(
          ['unzip', '-p', zip_path, slp_filename],
          check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

      # TODO: unzip to shared memory directly
      with open(slp_path, 'wb') as temp_slp:
        temp_slp.write(result.stdout)

      # Convert .slp to .slpp using the slp tool (also in shared memory)
      slpp_path = os.path.join(temp_dir, 'slpp')
      result = subprocess.run([
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
    return e.stderr
  except Exception as e:
    # print(f"Error converting {slp_filename}: {e}")
    return str(e)


def process_file(args: Tuple[str, str, str]) -> Tuple[str, Optional[str]]:
  """Process a single file for multithreading.

  Args:
    args: Tuple of (zip_path, slp_filename, output_path)

  Returns:
    Tuple of (slp_filename, error_message)
  """
  zip_path, slp_filename, output_path = args
  error = convert_slp_to_slpp_gz(zip_path, slp_filename, output_path)
  return slp_filename, error


def convert_zip_archive(
    input_zip_path: str,
    output_zip_path: str,
    num_threads: int = 1,
    limit: Optional[int] = None,
    failed_output_path: Optional[str] = None,
):
  """Convert all .slp files in a zip archive to .slpp.gz format.

  Args:
    input_zip_path: Path to input zip file containing .slp files
    output_zip_path: Path to output zip file for .slpp.gz files
    num_threads: Number of threads to use for parallel processing
    limit: Maximum number of files to process (for testing)
    failed_output_path: Optional path to store failed .slp files
  """
  print(f"Converting {input_zip_path} -> {output_zip_path}")
  print(f"Using {num_threads} thread{'s' if num_threads != 1 else ''}")

  # Create temporary directory for output files (not in shared memory)
  with tempfile.TemporaryDirectory() as temp_output_dir:

    # Find all .slp files in the zip archive
    slp_files = []
    output_dirs = set()  # Track output directories to create
    with zipfile.ZipFile(input_zip_path, 'r') as zf:
      for file_info in zf.filelist:
        if file_info.filename.lower().endswith('.slp'):
          # Generate output filename
          output_name = Path(file_info.filename).stem + '.slpp.gz'
          output_path = os.path.join(temp_output_dir, output_name)
          output_dirs.add(os.path.dirname(output_path))
          slp_files.append((input_zip_path, file_info.filename, output_path))

    # Create output directories
    for output_dir in output_dirs:
      os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(slp_files)} .slp files in archive")

    # Apply limit if specified
    if limit is not None and limit > 0:
      slp_files = slp_files[:limit]
      print(f"Limited to first {len(slp_files)} files")

    if not slp_files:
      print("No .slp files found in input archive")
      return

    results: list[tuple[str, Optional[str]]] = []

    # Process files
    if num_threads == 1:
      # Single-threaded processing
      for args in tqdm.tqdm(slp_files, desc="Converting", unit="file"):
        result = process_file(args)
        results.append(result)
    else:
      # Multi-threaded processing
      with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        try:
          futures = [executor.submit(process_file, file_args) for file_args in slp_files]

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

    if successful_conversions == 0:
      print("No files were successfully converted")
      return

    # Create output zip using command line zip (uncompressed)
    print("Creating output archive...")
    result = subprocess.run([
      'zip', '-r', '-0', output_zip_path, '.'
    ], cwd=temp_output_dir, capture_output=True, text=True)

    if result.returncode != 0:
      raise RuntimeError(f"Failed to create output zip: {result.stderr}")

    print(f"Output saved to: {output_zip_path}")

    if failed_conversions > 0:
      print(f"Some files failed to convert: {failed_conversions} errors")

      # Create archive of failed files if requested
      if failed_output_path:
        print(f"Creating archive of failed files: {failed_output_path}")
        failed_files = [filename for filename, error in results if error is not None]
        utils.extract_zip_files(input_zip_path, failed_files, failed_output_path)
        print(f"Failed files saved to: {failed_output_path}")

      for slp_filename, error in results:
        if error is not None:
          print(f"Error converting {slp_filename}: {error}")


def main(_):
  # Validate input file exists
  if not os.path.exists(FLAGS.input):
    print(f"Error: Input file '{FLAGS.input}' does not exist")
    return 1

  # Validate output directory exists
  output_dir = os.path.dirname(FLAGS.output)
  if output_dir and not os.path.exists(output_dir):
    print(f"Error: Output directory '{output_dir}' does not exist")
    return 1

  try:
    convert_zip_archive(FLAGS.input, FLAGS.output, FLAGS.threads, FLAGS.limit, FLAGS.failed_output)
    return 0
  except Exception as e:
    print(f"Error: {e}")
    return 1


if __name__ == "__main__":
  app.run(main)
