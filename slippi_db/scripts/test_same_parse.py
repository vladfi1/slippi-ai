from collections import Counter
from concurrent import futures
import os
import random
import time
from typing import Dict, List, Tuple

from absl import app
from absl import flags
import tqdm

from slippi_ai.types import InvalidGameError
from slippi_db import preprocessing
from slippi_db import utils

ROOT = flags.DEFINE_string('root', None, 'root directory', required=True)
MAX_FILES = flags.DEFINE_integer('max_files', None, 'max files to process')
THREADS = flags.DEFINE_integer('threads', 1, 'number of threads')
LOG_INTERVAL = flags.DEFINE_integer('log_interval', 20, 'log interval')

def find_archives_in_subdirs(root: str) -> List[str]:
  """Find all .7z and .zip archives in subdirectories of root."""
  archives = []
  for dirpath, _, filenames in os.walk(root):
    for filename in filenames:
      if filename.endswith(('.7z', '.zip')):
        archives.append(os.path.join(dirpath, filename))
  return archives


def get_files_from_archive(archive_path: str) -> List[utils.LocalFile]:
  """Get all slp files from an archive."""
  if archive_path.endswith('.7z'):
    return utils.traverse_slp_files_7z(archive_path)
  elif archive_path.endswith('.zip'):
    return utils.traverse_slp_files_zip(archive_path)
  else:
    raise ValueError(f'Unknown archive type: {archive_path}')


def distribute_files(archives_files: Dict[str, List[utils.LocalFile]], max_files: int) -> List[utils.LocalFile]:
  """Distribute files according to the specified logic:
  - Half of max_files are distributed evenly across archives
  - The remaining half is distributed proportionally to archive size
  """
  if not archives_files:
    return []

  num_archives = len(archives_files)
  half_max = max_files // 2

  # Calculate even distribution
  even_per_archive = half_max // num_archives
  even_remainder = half_max % num_archives

  # Calculate proportional distribution
  total_files = sum(len(files) for files in archives_files.values())
  proportional_half = max_files - half_max

  selected_files = []

  for i, (archive, files) in enumerate(archives_files.items()):
    if not files:
      continue

    # Even distribution part
    even_count = even_per_archive + (1 if i < even_remainder else 0)

    # Proportional distribution part
    proportion = len(files) / total_files if total_files > 0 else 0
    proportional_count = int(proportional_half * proportion)

    # Total files to select from this archive
    total_to_select = min(even_count + proportional_count, len(files))

    # Randomly sample files
    selected = random.sample(files, total_to_select)
    selected_files.extend(selected)

  return selected_files[:max_files]  # Ensure we don't exceed max_files


def test(file: utils.LocalFile, debug: bool = False) -> dict:
  with file.extract(utils.get_tmp_dir(in_memory=True)) as path:
    meta = preprocessing.get_metadata_safe(path)
    valid, reason = preprocessing.is_training_replay(meta)
    if not valid:
      return {'outcome': 'skipped', 'reason': reason}

    if debug:
      preprocessing.assert_same_parse(path)
    else:
      try:
        preprocessing.assert_same_parse(path)
      except (AssertionError, InvalidGameError) as e:
        return {'outcome': 'failed', 'reason': e.args[0]}

    return {'outcome': 'passed'}

def main(_):
  root = ROOT.value

  # Handle different input types
  if os.path.isdir(root):
    # Check if root is a directory containing archives
    archives = find_archives_in_subdirs(root)
    if archives:
      print(f'Found {len(archives)} archives in subdirectories.')
      # Collect files from all archives
      archives_files = {}
      total_files = 0
      for archive in archives:
        try:
          archive_files = get_files_from_archive(archive)
          if archive_files:
            archives_files[archive] = archive_files
            total_files += len(archive_files)
            print(f'  {archive}: {len(archive_files)} files')
        except Exception as e:
          print(f'  Error processing {archive}: {e}')

      if not archives_files:
        print('No valid archives found.')
        return

      print(f'Total files found: {total_files}')

      # Apply file distribution logic if MAX_FILES is set
      if MAX_FILES.value is not None and total_files > MAX_FILES.value:
        files = distribute_files(archives_files, MAX_FILES.value)
        print(f'Selected {len(files)} files using distribution logic.')
      else:
        # Use all files if no limit or within limit
        files = []
        for archive_files in archives_files.values():
          files.extend(archive_files)
    else:
      # Fall back to directory traversal for .slp files
      files = utils.traverse_slp_files(root)
      if MAX_FILES.value is not None and len(files) > MAX_FILES.value:
        files = random.sample(files, MAX_FILES.value)
  elif root.endswith('.7z'):
    files = utils.traverse_slp_files_7z(root)
    if MAX_FILES.value is not None and len(files) > MAX_FILES.value:
      files = random.sample(files, MAX_FILES.value)
  elif root.endswith('.zip'):
    files = utils.traverse_slp_files_zip(root)
    if MAX_FILES.value is not None and len(files) > MAX_FILES.value:
      files = random.sample(files, MAX_FILES.value)
  else:
    raise ValueError(f'Unknown file type: {root}')

  print(f'Processing {len(files)} files.')

  if THREADS.value == 1:
    results = [test(file, debug=True) for file in tqdm.tqdm(files)]
  else:
    with futures.ProcessPoolExecutor(max_workers=THREADS.value) as executor:
      try:
        results = []
        todo = [executor.submit(test, file) for file in files]
        as_completed = futures.as_completed(todo)
        last_log = time.perf_counter() - LOG_INTERVAL.value + 5
        for f in tqdm.tqdm(as_completed, total=len(files)):
          results.append(f.result())

          now = time.perf_counter()
          if now - last_log > LOG_INTERVAL.value:
            last_log = now
            print(Counter(r['outcome'] for r in results))
            print(Counter(filter(None, [r.get('reason') for r in results])))
      except KeyboardInterrupt:
        executor.shutdown(cancel_futures=True)

  counter = Counter(r['outcome'] for r in results)
  print(counter)
  print(Counter(filter(None, [r.get('reason') for r in results])))

if __name__ == '__main__':
  app.run(main)
