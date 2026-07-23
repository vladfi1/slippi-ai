"""Remove duplicate .slp/.slpz files from archives in a directory.

Loads the set of slp_md5 hashes already recorded in a parsed.sqlite metadata
database, then scans .zip archives of .slp/.slpz files under a directory. Any
archive member whose (decompressed) content hash matches a hash we've already
seen -- either from the parsed db or from an earlier file in this run -- is
considered a duplicate and removed from its archive.

Separately, each file's (match_id, match_game, match_tiebreaker) -- read
directly from its start event via peppi_py, the same fields parse_local.py
records as the 'match' dict -- is used to catch replays of the same ranked
match/game recorded more than once (e.g. saved by both clients) whose files
differ byte-for-byte, so they wouldn't be caught by content hashing. As with
md5s, known match keys are seeded from the parsed db and accumulated across
the run, so the first file with a given match key is kept and every later
one (in this run, in archive/name order) is removed.

Files that can't be decoded or parsed for match info at all -- as opposed to
a clean absence of match info -- are called invalid. They're only removed if
--delete_invalid is set; otherwise they're left alone.

Files are streamed out of each archive with slippi_db.utils.stream_files_zip
and hashed/inspected in parallel worker processes via slippi_ai.datasets.
MPMap, which keeps only a bounded number of files in flight at a time
(num_workers * buffer) instead of queuing the whole archive up front.

Usage:
  python -m slippi_db.scripts.remove_duplicate_slp \
      --parsed_db=/path/to/parsed.sqlite \
      --input_dir=/path/to/Raw \
      --dry_run=True
"""

import dataclasses
import os
from pathlib import Path
import sqlite3
import tempfile
import typing as tp
import zipfile

from absl import app
from absl import flags
import peppi_py
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
DELETE_INVALID = flags.DEFINE_boolean(
    'delete_invalid', False,
    'If true, also remove invalid replays -- ones that could not be decoded '
    'or parsed for match info (e.g. corrupt or truncated files). By default '
    'they are left in place and only counted in a per-archive summary.')

MatchKey = tuple[tp.Any, tp.Any, tp.Any]  # (match_id, match_game, match_tiebreaker)


def load_known_md5s(parsed_db_path: str) -> set[str]:
  """Loads the set of slp_md5 hashes already recorded in the parsed db."""
  conn = sqlite3.connect(f'file:{parsed_db_path}?mode=ro', uri=True)
  try:
    cursor = conn.execute(
        'SELECT DISTINCT slp_md5 FROM replays WHERE slp_md5 IS NOT NULL')
    return {row[0] for row in cursor}
  finally:
    conn.close()


def load_known_match_keys(parsed_db_path: str) -> set[MatchKey]:
  """Loads the set of (match_id, match_game, match_tiebreaker) keys already
  recorded in the parsed db."""
  conn = sqlite3.connect(f'file:{parsed_db_path}?mode=ro', uri=True)
  try:
    cursor = conn.execute(
        'SELECT DISTINCT match_id, match_game, match_tiebreaker FROM replays '
        'WHERE match_id IS NOT NULL')
    return {tuple(row) for row in cursor}
  finally:
    conn.close()


Entry = tuple[str, str, bytes]  # (archive_path, name, raw bytes)
# (name, md5_hash, match_key, is_invalid) -- is_invalid is True if the file
# couldn't be decoded or its match info couldn't be read (e.g. corrupt or
# truncated data), as opposed to a clean absence of match info.
ProcessedEntry = tuple[str, tp.Optional[str], tp.Optional[MatchKey], bool]


def _extract_match_key(data: bytes) -> tp.Optional[MatchKey]:
  """Reads the match id/game/tiebreaker out of a decompressed .slp's start
  event via peppi_py. peppi_py only reads from a path, so the bytes are
  spilled to a temp file; skip_frames avoids parsing frame data."""
  with tempfile.NamedTemporaryFile(suffix='.slp') as f:
    f.write(data)
    f.flush()
    game = peppi_py.read_slippi(f.name, skip_frames=True)
  match = game.start.match
  if match is None or match.id is None:
    return None
  return (match.id, match.game, match.tiebreaker)


def _process_entry(entry: Entry) -> ProcessedEntry:
  """Hashes a single archive member and extracts its match key, if any.
  Runs in a worker process."""
  archive_path, name, raw = entry
  try:
    data = utils.SlpZipFile(archive_path, name).from_raw(raw)
  except Exception:
    return name, None, None, True

  md5_hash = utils.md5(data)

  try:
    match_key = _extract_match_key(data)
  except Exception:
    return name, md5_hash, None, True

  return name, md5_hash, match_key, False


@dataclasses.dataclass
class ArchiveResult:
  duplicates: list[str]
  num_invalid: int
  num_processed: int


def find_duplicates_in_archive(
    archive_path: str,
    known_md5s: set[str],
    known_match_keys: set[MatchKey],
    num_workers: int = 1,
    buffer: int = 16,
    delete_invalid: bool = False,
) -> ArchiveResult:
  """Finds duplicate and invalid .slp/.slpz members within a single archive.

  Updates known_md5s and known_match_keys in place with the hash/match key
  of every non-duplicate file encountered, so that later archives are
  deduplicated against this one too.

  Prints a one-line summary of the duplicate/invalid ratios for this archive.
  Invalid files (couldn't be decoded or parsed for match info) are only
  included in the returned duplicates list if delete_invalid is set.
  """
  with zipfile.ZipFile(archive_path) as zf:
    names = [
        info.filename for info in zf.infolist()
        if not info.is_dir() and utils.is_slp_file(info.filename)
    ]

  if not names:
    return ArchiveResult(duplicates=[], num_invalid=0, num_processed=0)

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
  processed_ds = entry_ds.map_mp(_process_entry, num_workers=num_workers, buffer=buffer)

  duplicates = []
  num_invalid = 0
  progress = tqdm.tqdm(
      processed_ds, total=len(names), unit='file',
      desc=os.path.basename(archive_path), smoothing=0)
  try:
    processed = 0
    for name, md5_hash, match_key, is_invalid in progress:
      processed += 1

      is_duplicate = False
      if md5_hash is not None:
        if md5_hash in known_md5s:
          is_duplicate = True
        else:
          known_md5s.add(md5_hash)

      if not is_duplicate and match_key is not None:
        if match_key in known_match_keys:
          is_duplicate = True
        else:
          known_match_keys.add(match_key)

      if is_invalid:
        num_invalid += 1
        if delete_invalid:
          is_duplicate = True

      if is_duplicate:
        duplicates.append(name)

      progress.set_postfix(
          dupe_ratio=f'{len(duplicates) / processed:.1%}',
          invalid_ratio=f'{num_invalid / processed:.1%}')
  finally:
    processed_ds.stop()

  print(
      f'{archive_path}: '
      f'{len(duplicates)}/{processed} duplicate ({len(duplicates) / processed:.1%}), '
      f'{num_invalid}/{processed} invalid ({num_invalid / processed:.1%})')

  return ArchiveResult(
      duplicates=duplicates, num_invalid=num_invalid, num_processed=processed)


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

  print(f'Loading known match keys from {parsed_db_path}')
  known_match_keys = load_known_match_keys(parsed_db_path)
  print(f'Loaded {len(known_match_keys)} known match keys')

  zip_files = sorted(input_dir.rglob('*.zip'))
  print(f'Found {len(zip_files)} zip archives in {input_dir}')

  total_duplicates = 0
  total_invalid = 0
  total_files = 0
  num_processed = 0

  try:
    for zip_path in zip_files:
      result = find_duplicates_in_archive(
          str(zip_path), known_md5s, known_match_keys,
          num_workers=NUM_WORKERS.value, buffer=BUFFER.value,
          delete_invalid=DELETE_INVALID.value)
      num_processed += 1
      total_duplicates += len(result.duplicates)
      total_invalid += result.num_invalid
      total_files += result.num_processed

      if result.duplicates and not DRY_RUN.value:
        utils.delete_from_zip(str(zip_path), result.duplicates)
  except KeyboardInterrupt:
    print(f'\nInterrupted after {num_processed}/{len(zip_files)} archive(s).')

  action = 'Would remove' if DRY_RUN.value else 'Removed'
  invalid_note = (
      '(included above)' if DELETE_INVALID.value else 'left in place')
  print(
      f'{action} {total_duplicates} duplicate file(s) out of {total_files} '
      f'total, across {num_processed} archive(s). '
      f'{total_invalid} were invalid {invalid_note}.')


if __name__ == '__main__':
  app.run(main)
