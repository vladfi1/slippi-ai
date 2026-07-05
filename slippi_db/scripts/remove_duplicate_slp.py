"""Remove duplicate .slp/.slpz files from archives in a directory.

Loads the set of slp_md5 hashes already recorded in a parsed.sqlite metadata
database, then scans .zip archives of .slp/.slpz files under a directory. Any
archive member whose (decompressed) content hash matches a hash we've already
seen -- either from the parsed db or from an earlier file in this run -- is
considered a duplicate and removed from its archive.

Files are streamed out of each archive with slippi_db.utils.stream_files_zip
and hashed in parallel worker processes via slippi_ai.datasets.MPMap, which
keeps only a bounded number of files in flight at a time (num_workers *
buffer) instead of queuing the whole archive up front.

Usage:
  python -m slippi_db.scripts.remove_duplicate_slp \
      --parsed_db=/path/to/parsed.sqlite \
      --input_dir=/path/to/Raw \
      --dry_run=True
"""

import os
from pathlib import Path
import sqlite3
import typing as tp
import zipfile

from absl import app
from absl import flags
import tqdm

from slippi_ai import datasets
from slippi_db import utils

PARSED_DB = flags.DEFINE_string(
    'parsed_db', None, 'Path to the parsed.sqlite metadata database.',
    required=True)
INPUT_DIR = flags.DEFINE_string(
    'input_dir', None,
    'Directory (searched recursively) containing .zip archives of .slp/.slpz files.',
    required=True)
NUM_WORKERS = flags.DEFINE_integer(
    'threads', 1, 'Number of worker processes to use for hashing files within an archive.')
BUFFER = flags.DEFINE_integer(
    'buffer', 16, 'Number of files to buffer per worker process.')
DRY_RUN = flags.DEFINE_boolean(
    'dry_run', True, 'If true, only report how many files would be removed.')


def load_known_md5s(parsed_db_path: str) -> set[str]:
  """Loads the set of slp_md5 hashes already recorded in the parsed db."""
  conn = sqlite3.connect(f'file:{parsed_db_path}?mode=ro', uri=True)
  try:
    cursor = conn.execute(
        'SELECT DISTINCT slp_md5 FROM replays WHERE slp_md5 IS NOT NULL')
    return {row[0] for row in cursor}
  finally:
    conn.close()


Entry = tuple[str, str, bytes]  # (archive_path, name, raw bytes)


def _hash_entry(entry: Entry) -> tuple[str, tp.Optional[str]]:
  """Hashes a single archive member. Runs in a worker process."""
  archive_path, name, raw = entry
  try:
    data = utils.SlpZipFile(archive_path, name).from_raw(raw)
    return name, utils.md5(data)
  except Exception as e:
    print(f'Error hashing {name}: {e}')
    return name, None


def find_duplicates_in_archive(
    archive_path: str,
    known_md5s: set[str],
    num_workers: int = 1,
    buffer: int = 16,
) -> list[str]:
  """Finds duplicate .slp/.slpz members within a single archive.

  Updates known_md5s in place with the hashes of every non-duplicate file
  encountered, so that later archives are deduplicated against this one too.

  Returns:
    The list of archive member names to remove.
  """
  with zipfile.ZipFile(archive_path) as zf:
    names = [
        info.filename for info in zf.infolist()
        if not info.is_dir() and utils.is_slp_file(info.filename)
    ]

  if not names:
    return []

  def iter_entries() -> tp.Iterator[Entry]:
    for name, raw in utils.stream_files_zip(archive_path):
      if utils.is_slp_file(name):
        yield archive_path, name, raw

  # MPMap eagerly primes num_workers * buffer items from the dataset before
  # it starts, so both must be capped to fit within a (possibly small)
  # archive.
  num_workers = min(num_workers, len(names))
  buffer = max(1, min(buffer, len(names) // num_workers))

  entry_ds = datasets.IteratorDataset(iter_entries())
  hash_ds = entry_ds.map_mp(_hash_entry, num_workers=num_workers, buffer=buffer)

  duplicates = []
  progress = tqdm.tqdm(
      hash_ds, total=len(names), unit='file',
      desc=os.path.basename(archive_path), smoothing=0)
  try:
    processed = 0
    for name, md5_hash in progress:
      processed += 1
      if md5_hash is None:
        continue  # hashing failed; leave the file alone
      if md5_hash in known_md5s:
        duplicates.append(name)
      else:
        known_md5s.add(md5_hash)

      progress.set_postfix(dupe_ratio=f'{len(duplicates) / processed:.1%}')
  finally:
    hash_ds.stop()

  return duplicates


def main(_):
  parsed_db_path = PARSED_DB.value
  if not os.path.exists(parsed_db_path):
    raise FileNotFoundError(f'Parsed db does not exist: {parsed_db_path}')

  input_dir = Path(INPUT_DIR.value)
  if not input_dir.is_dir():
    raise ValueError(f'Input directory does not exist: {input_dir}')

  print(f'Loading known md5s from {parsed_db_path}')
  known_md5s = load_known_md5s(parsed_db_path)
  print(f'Loaded {len(known_md5s)} known md5 hashes')

  zip_files = sorted(input_dir.rglob('*.zip'))
  print(f'Found {len(zip_files)} zip archives in {input_dir}')

  total_duplicates = 0
  num_processed = 0

  try:
    for zip_path in zip_files:
      duplicates = find_duplicates_in_archive(
          str(zip_path), known_md5s,
          num_workers=NUM_WORKERS.value, buffer=BUFFER.value)
      num_processed += 1

      if duplicates:
        print(f'{zip_path}: {len(duplicates)} duplicate file(s)')
        total_duplicates += len(duplicates)

      if duplicates and not DRY_RUN.value:
        utils.delete_from_zip(str(zip_path), duplicates)
  except KeyboardInterrupt:
    print(f'\nInterrupted after {num_processed}/{len(zip_files)} archive(s).')

  action = 'Would remove' if DRY_RUN.value else 'Removed'
  print(f'{action} {total_duplicates} duplicate file(s) across {num_processed} archive(s)')


if __name__ == '__main__':
  app.run(main)
