#!/usr/bin/env python3
"""Test script to compare streaming zip extraction vs per-file extraction.

This script compares two approaches for processing files from zip archives:
1. Streaming method: One process runs `unzip -p` and feeds a multiprocessing Queue
2. Standard method: Each worker process calls unzip on individual files (using ZipFile class)

Usage:
  python test_zip_streaming.py --archive path/to/archive.zip --num_processes 4
"""

import argparse
import concurrent.futures
import hashlib
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from typing import List, Tuple, Dict

# Add parent directories to path to import slippi_db modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from slippi_db import utils


def compute_md5(content: bytes) -> str:
  """Compute MD5 hash of file content."""
  return hashlib.md5(content).hexdigest()


def create_test_archive(num_files: int = 20, file_size_kb: int = 100) -> str:
  """Create a test zip archive with dummy .slp files.

  Args:
    num_files: Number of dummy .slp files to create
    file_size_kb: Approximate size of each file in KB

  Returns:
    Path to the created test archive
  """
  # Create temporary directory for test files
  temp_dir = tempfile.mkdtemp(prefix='slp_test_')
  archive_path = os.path.join(temp_dir, 'test_archive.zip')

  print(f"Creating test archive with {num_files} files (~{file_size_kb}KB each)")

  # Create dummy .slp file content (somewhat realistic looking)
  # SLP files start with specific headers
  slp_header = b'{U\x03raw[$U#l\x00'  # Common SLP file start

  try:
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
      for i in range(num_files):
        filename = f"test_file_{i:03d}.slp"

        # Create dummy content with some variation
        base_content = slp_header + b'\x00' * (file_size_kb * 1024 - len(slp_header))

        # Add some variation to make each file unique
        variation = f"file_{i}_variation".encode() + b'\x00' * 100
        content = base_content[:len(base_content)//2] + variation + base_content[len(base_content)//2 + len(variation):]

        zf.writestr(filename, content)

    print(f"✓ Test archive created: {archive_path}")

    # Print archive info
    with zipfile.ZipFile(archive_path) as zf:
      total_size = sum(info.file_size for info in zf.infolist())
      compressed_size = sum(info.compress_size for info in zf.infolist())
      print(f"  Total uncompressed size: {total_size:,} bytes")
      print(f"  Compressed archive size: {compressed_size:,} bytes")
      print(f"  Compression ratio: {compressed_size/total_size:.1%}")

    return archive_path

  except Exception as e:
    print(f"Error creating test archive: {e}")
    # Clean up on error
    if os.path.exists(archive_path):
      os.remove(archive_path)
    os.rmdir(temp_dir)
    raise


def process_file_content(file_name: str, content: bytes) -> Tuple[str, str, int]:
  """Process file content and return metadata.

  Returns:
    Tuple of (filename, md5_hash, size_bytes)
  """
  md5_hash = compute_md5(content)
  return (file_name, md5_hash, len(content))


def stream_files(
    archive_path: str,
    file_info: List[Tuple[str, int]],
    file_queue: mp.Queue,
):

  proc = None
  try:
    # Start unzip process to stream all files
    proc = subprocess.Popen(
      ['unzip', '-p', archive_path],
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
      # bufsize=0  # Unbuffered for immediate streaming
    )

    files_processed = 0
    total_files = len(file_info)
    start_time = time.perf_counter()

    # Read each file's content based on expected size
    for file_name, expected_size in file_info:
      if expected_size == 0:
        # Handle empty files
        file_queue.put((archive_path, file_name, b''))
        files_processed += 1
        continue

      # Read exactly expected_size bytes using a pre-allocated buffer
      buffer = bytearray(expected_size)
      bytes_read = 0
      while bytes_read < expected_size:
        chunk_size = proc.stdout.readinto(memoryview(buffer)[bytes_read:])
        if chunk_size is None or chunk_size == 0:
          break
        bytes_read += chunk_size

      content = bytes(buffer[:bytes_read])

      if len(content) != expected_size:
        print(f"Warning: Expected {expected_size} bytes for {file_name}, got {len(content)}")

      file_queue.put((archive_path, file_name, content))
      files_processed += 1

      # Progress update with ETA (in place)
      if files_processed % 10 == 0 or files_processed == total_files:
        elapsed = time.perf_counter() - start_time
        if files_processed > 0:
          rate = files_processed / elapsed
          remaining = total_files - files_processed
          eta = remaining / rate if rate > 0 else 0
          print(f"\rStreamed {files_processed}/{total_files} files ({rate:.1f}/sec, ETA: {eta:.1f}s)", end='', flush=True)

    if total_files > 0:  # Ensure we end the progress line
      print()

    # Wait for unzip to finish
    # return_code = proc.wait()
    # if return_code != 0:
    #   print(f"Warning: unzip exited with code {return_code}")

  finally:
    if proc is not None:
      proc.terminate()

def producer_function(
    all_file_info: List[Tuple[str, List[Tuple[str, int]]]],
    file_queue: mp.Queue,
    num_processes: int,
):
  """Producer process that streams zip contents from multiple archives."""

  try:
    for archive_path, file_info in all_file_info:
      print(f"\nStreaming from {os.path.basename(archive_path)}...")
      stream_files(archive_path, file_info, file_queue)

    for _ in range(num_processes):
      file_queue.put(None)
  except Exception as e:
    print(f"Producer error: {e}")
    # Signal error to workers
    for _ in range(num_processes):
      file_queue.put(None)

def worker_function(file_queue: mp.Queue, results_queue: mp.Queue):
  """Worker process that consumes from queue."""
  processed_count = 0
  while True:
    item = file_queue.get()
    if item is None:
      break

    archive_path, file_name, content = item

    # Process the content (applying any decompression if needed)
    try:
      # Handle compressed files using the ZipFile utility
      file_obj = utils.ZipFile(archive_path, file_name)
      processed_content = file_obj.from_raw(content)

      result = process_file_content(file_name, processed_content)
      results_queue.put(result)
      processed_count += 1

    except Exception as e:
      print(f"Error processing {file_name} from {os.path.basename(archive_path)}: {e}")
      # Still add an entry to maintain count
      results_queue.put((file_name, "ERROR", len(content)))
      processed_count += 1


def streaming_method(archive_paths: List[str], num_processes: int, file_limit: int = None) -> Tuple[List[Tuple], float]:
  """Process multiple zip archives using streaming with unzip -p.

  A single producer process streams all files from all archives using `unzip -p` and puts them
  into a queue. Multiple worker processes consume from the queue.
  """
  start_time = time.perf_counter()

  # Collect file info from all archives
  all_file_info = []  # List of (archive_path, [(filename, filesize), ...])
  total_files = 0
  total_size = 0

  for archive_path in archive_paths:
    with zipfile.ZipFile(archive_path) as zf:
      file_info = [(info.filename, info.file_size)
                   for info in zf.infolist()
                   if utils.is_slp_file(info.filename)]

      # Apply file limit per archive if specified
      if file_limit is not None and file_limit > 0:
        file_info = file_info[:file_limit]

      if file_info:
        all_file_info.append((archive_path, file_info))
        total_files += len(file_info)
        total_size += sum(size for _, size in file_info)

  if not all_file_info:
    print(f"No .slp files found in any archives")
    return [], 0

  print(f"Streaming method: Processing {total_files} files ({total_size:,} bytes) from {len(archive_paths)} archive(s) with {num_processes} workers")

  # Create queues for passing data between processes
  # Use a reasonable buffer size to avoid memory issues
  queue_size = min(num_processes * 4, 50)
  file_queue = mp.Queue(maxsize=queue_size)
  results_queue = mp.Queue()

  # Start producer in a separate process
  producer_proc = mp.Process(
      target=producer_function,
      args=(all_file_info, file_queue, num_processes))
  producer_proc.start()

  # Start workers using multiprocessing.Process directly
  worker_processes: list[mp.Process] = []
  for _ in range(num_processes):
    worker_proc = mp.Process(
        target=worker_function,
        args=(file_queue, results_queue))
    worker_proc.start()
    worker_processes.append(worker_proc)

  try:
    results = []
    for _ in range(total_files):
      result = results_queue.get()
      results.append(result)
  except KeyboardInterrupt:
    producer_proc.terminate()
    for worker_proc in worker_processes:
      worker_proc.terminate()
    raise

  # Wait for all workers to finish
  for worker_proc in worker_processes:
    worker_proc.join()

  # Wait for producer to finish
  producer_proc.join()

  elapsed = time.perf_counter() - start_time

  # Sort results by filename for consistent comparison
  results.sort(key=lambda x: x[0])

  print(f"Streaming method completed: {len(results)} files processed")

  return results, elapsed


def process_single_file(file_obj: utils.ZipFile) -> Tuple[str, str, int]:
  """Process a single file from the archive."""
  try:
    content = file_obj.read()
    return process_file_content(file_obj.path, content)
  except Exception as e:
    print(f"Error processing {file_obj.path}: {e}")
    return (file_obj.path, "ERROR", 0)


def standard_method(archive_paths: List[str], num_processes: int, file_limit: int = None) -> Tuple[List[Tuple], float]:
  """Process multiple zip archives using the standard ZipFile approach.

  Each worker process independently calls unzip on individual files from all archives.
  """
  start_time = time.perf_counter()

  # Get list of files from all archives
  all_files = []
  for archive_path in archive_paths:
    files = utils.traverse_slp_files_zip(archive_path)

    # Apply file limit per archive if specified
    if file_limit is not None and file_limit > 0:
      files = files[:file_limit]

    all_files.extend(files)

  if not all_files:
    print(f"No .slp files found in any archives")
    return [], 0

  print(f"Standard method: Processing {len(all_files)} files from {len(archive_paths)} archive(s) with {num_processes} workers")

  results = []

  if num_processes == 1:
    # Single-threaded processing
    start_time = time.perf_counter()
    for i, file_obj in enumerate(all_files):
      results.append(process_single_file(file_obj))
      if (i + 1) % 10 == 0 or (i + 1) == len(all_files):
        elapsed = time.perf_counter() - start_time
        processed = i + 1
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = len(all_files) - processed
        eta = remaining / rate if rate > 0 else 0
        print(f"\rProcessed {processed}/{len(all_files)} files ({rate:.1f}/sec, ETA: {eta:.1f}s)", end='', flush=True)

    if len(all_files) > 0:
      print()  # End the progress line
  else:
    # Multi-process processing
    start_time = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
      try:
        # Submit all tasks
        futures = {executor.submit(process_single_file, file_obj): file_obj
              for file_obj in all_files}

        # Process completed futures
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
          try:
            result = future.result()
            results.append(result)
            completed_count += 1

            if completed_count % 10 == 0 or completed_count == len(all_files):
              elapsed = time.perf_counter() - start_time
              rate = completed_count / elapsed if elapsed > 0 else 0
              remaining = len(all_files) - completed_count
              eta = remaining / rate if rate > 0 else 0
              print(f"\rProcessed {completed_count}/{len(all_files)} files ({rate:.1f}/sec, ETA: {eta:.1f}s)", end='', flush=True)

          except Exception as e:
            file_obj = futures[future]
            print(f"Error processing {file_obj.path}: {e}")
            results.append((file_obj.path, "ERROR", 0))

        if len(all_files) > 0:
          print()  # End the progress line

      except KeyboardInterrupt:
        executor.shutdown(cancel_futures=True)
        raise

  elapsed = time.perf_counter() - start_time

  # Sort results by filename for consistent comparison
  results.sort(key=lambda x: x[0])

  print(f"Standard method completed: {len(results)} files processed")

  return results, elapsed


def verify_results(results1: List[Tuple], results2: List[Tuple], name1: str, name2: str) -> bool:
  """Verify that two sets of results match."""
  if len(results1) != len(results2):
    print(f"ERROR: Different number of files processed: {name1}={len(results1)}, {name2}={len(results2)}")
    return False

  mismatches = []
  hash_mismatches = []

  # Create dictionaries for easier lookup
  dict1 = {filename: (md5_hash, size) for filename, md5_hash, size in results1}
  dict2 = {filename: (md5_hash, size) for filename, md5_hash, size in results2}

  all_files = set(dict1.keys()) | set(dict2.keys())

  for filename in sorted(all_files):
    if filename not in dict1:
      mismatches.append(f"Missing in {name1}: {filename}")
    elif filename not in dict2:
      mismatches.append(f"Missing in {name2}: {filename}")
    else:
      hash1, size1 = dict1[filename]
      hash2, size2 = dict2[filename]

      if hash1 != hash2 or size1 != size2:
        hash_mismatches.append(f"{filename}: {name1}=({hash1}, {size1}) vs {name2}=({hash2}, {size2})")

  if mismatches:
    print(f"ERROR: {len(mismatches)} files missing between {name1} and {name2}")
    for msg in mismatches[:5]:  # Show first 5
      print(f"  {msg}")
    return False

  if hash_mismatches:
    print(f"ERROR: {len(hash_mismatches)} files have different hashes/sizes between {name1} and {name2}")
    for msg in hash_mismatches[:5]:  # Show first 5
      print(f"  {msg}")
    return False

  print(f"✓ Verification passed: {name1} and {name2} produced identical results")
  print(f"  - {len(results1)} files processed")
  print(f"  - All MD5 hashes match")
  print(f"  - All file sizes match")
  return True


def main():
  parser = argparse.ArgumentParser(description='Test zip streaming vs standard extraction')
  parser.add_argument('--archive', nargs='*', help='Path(s) to zip archive(s) (will create test archive if not provided)')
  parser.add_argument('--num_processes', type=int, default=4, help='Number of worker processes')
  parser.add_argument('--method', choices=['all', 'streaming', 'standard'],
            default='all', help='Which method(s) to test')
  parser.add_argument('--skip_verification', action='store_true',
            help='Skip verification of results')
  parser.add_argument('--test_files', type=int, default=50,
            help='Number of files to create in test archive (default: 50)')
  parser.add_argument('--test_file_size', type=int, default=100,
            help='Size of each test file in KB (default: 100)')
  parser.add_argument('--file_limit', type=int, default=None,
            help='Limit the number of files to process from the archive')

  args = parser.parse_args()

  # Create or use provided archives
  created_test_archives = []
  archives_to_process = []

  if not args.archive:  # No archives provided
    print("No archive provided, creating test archive...")
    print()
    test_archive = create_test_archive(args.test_files, args.test_file_size)
    archives_to_process.append(test_archive)
    created_test_archives.append(test_archive)
  else:
    # Validate all provided archives exist
    for archive_path in args.archive:
      if not os.path.exists(archive_path):
        print(f"Error: Archive not found: {archive_path}")
        return 1
      archives_to_process.append(archive_path)

  print("=" * 80)
  print(f"ZIP PROCESSING BENCHMARK")
  print("=" * 80)
  print(f"Archives to process: {len(archives_to_process)}")
  for archive in archives_to_process:
    print(f"  - {archive}")
  print(f"Number of processes: {args.num_processes}")
  print(f"Methods to test: {args.method}")
  if args.file_limit is not None:
    print(f"File limit per archive: {args.file_limit}")
  print()

  results = {}
  times = {}

  # Run standard method
  if args.method in ['all', 'standard']:
    print("=" * 60)
    print("RUNNING STANDARD METHOD")
    print("=" * 60)
    try:
      results['standard'], times['standard'] = standard_method(archives_to_process, args.num_processes, args.file_limit)
      print(f"✓ Standard method completed in {times['standard']:.2f} seconds")
      print(f"  Processed {len(results['standard'])} files")
    except Exception as e:
      print(f"✗ Standard method failed: {e}")
      results['standard'], times['standard'] = [], float('inf')
    print()

  # Run streaming method
  if args.method in ['all', 'streaming']:
    print("=" * 60)
    print("RUNNING STREAMING METHOD")
    print("=" * 60)
    try:
      results['streaming'], times['streaming'] = streaming_method(archives_to_process, args.num_processes, args.file_limit)
      print(f"✓ Streaming method completed in {times['streaming']:.2f} seconds")
      print(f"  Processed {len(results['streaming'])} files")
    except Exception as e:
      print(f"✗ Streaming method failed: {e}")
      results['streaming'], times['streaming'] = [], float('inf')
    print()

  # Verify results match
  if not args.skip_verification and len(results) > 1:
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    if 'standard' in results and 'streaming' in results:
      verify_results(results['standard'], results['streaming'], 'standard', 'streaming')
    print()

  # Print performance comparison
  if len(times) > 1:
    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    # Sort by time
    sorted_times = sorted(times.items(), key=lambda x: x[1])
    baseline_time = sorted_times[0][1]

    print(f"{'Method':<15} {'Time (s)':<10} {'Speedup':<8} {'Files/sec':<12}")
    print("-" * 50)

    for method, elapsed in sorted_times:
      speedup = baseline_time / elapsed if elapsed > 0 else 0
      num_files = len(results[method]) if method in results else 0
      throughput = num_files / elapsed if elapsed > 0 else 0

      print(f"{method:<15} {elapsed:>8.2f}   {speedup:>6.2f}x  {throughput:>10.1f}")

    print()
    print(f"🏆 Fastest method: {sorted_times[0][0]}")
    if len(sorted_times) > 1:
      improvement = ((sorted_times[-1][1] - sorted_times[0][1]) / sorted_times[-1][1]) * 100
      print(f"📈 Improvement over slowest: {improvement:.1f}%")

  # Clean up test archives if we created them
  for test_archive in created_test_archives:
    try:
      temp_dir = os.path.dirname(test_archive)
      os.remove(test_archive)
      os.rmdir(temp_dir)
      print(f"\n🗑️  Cleaned up test archive: {test_archive}")
    except Exception as e:
      print(f"\n⚠️  Could not clean up test archive {test_archive}: {e}")

  return 0


if __name__ == '__main__':
  # Support for multiprocessing on macOS
  mp.set_start_method('spawn', force=True)
  sys.exit(main())
