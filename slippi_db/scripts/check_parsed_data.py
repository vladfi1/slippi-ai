"""Run the data sanity check on already-parsed replays.

Reads the parquet files in Root/Parsed and updates the data_ok/data_error
columns in Root/parsed.sqlite. Unlike parse_local.py --recheck_data, this
doesn't need the raw .slp archives, making it much cheaper to re-run after
adding a new check to preprocessing.check_game_data.

By default only replays that haven't been checked (data_ok IS NULL) are
processed; use --recheck to re-check everything. Replays whose parquet file
is missing from Parsed/ are skipped and counted; regenerate them with
parse_local.py --reparse_missing_parsed.

Note: switches parsed.sqlite to WAL journal mode (persistent), so that
concurrent readers and this script's writes don't block each other.

Usage: python slippi_db/scripts/check_parsed_data.py --root=Root [--threads N]
"""

import concurrent.futures
import itertools
import logging
import multiprocessing
import os
import sqlite3
import zlib
from typing import Optional

from absl import app, flags
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from slippi_db import parse_local
from slippi_db import preprocessing

# (name, raw, path, compression)
CheckSpec = tuple[str, str, str, Optional[str]]
# (name, raw, data_ok, data_error, error)
CheckResult = tuple[str, str, Optional[bool], Optional[str], Optional[str]]


def check_file(spec: CheckSpec) -> CheckResult:
  name, raw, path, compression = spec
  try:
    with open(path, 'rb') as f:
      contents = f.read()
    if compression == 'zlib':
      contents = zlib.decompress(contents)
    table = pq.read_table(pa.BufferReader(contents))
    game = table['root'].combine_chunks()
    data_error = preprocessing.check_game_data(game)
    return name, raw, data_error is None, data_error, None
  except KeyboardInterrupt:
    raise
  except BaseException as e:
    return name, raw, None, None, repr(e)


def run_check(
    root: str,
    num_threads: int = 1,
    recheck: bool = False,
    dry_run: bool = False,
    commit_interval: int = 1000,
):
  parsed_db_path = os.path.join(root, 'parsed.sqlite')
  output_dir = os.path.join(root, 'Parsed')

  conn = sqlite3.connect(parsed_db_path, timeout=60)
  # WAL lets concurrent readers (e.g. notebooks) coexist with our updates
  # instead of causing "database is locked" errors. This is a persistent
  # property of the database file (revert with PRAGMA journal_mode=DELETE).
  journal_mode = conn.execute('PRAGMA journal_mode=WAL').fetchone()[0]
  if journal_mode != 'wal':
    logging.warning('could not enable WAL mode (got %r)', journal_mode)
  parse_local._migrate_parsed_schema(conn)
  conn.commit()

  missing_cond = '' if recheck else 'AND data_ok IS NULL'
  rows = conn.execute(f"""
    SELECT name, raw, slp_md5, compression FROM replays
    WHERE valid = 1 AND is_training = 1 AND slp_md5 IS NOT NULL
    {missing_cond}
  """).fetchall()

  existing = set(os.listdir(output_dir)) if os.path.isdir(output_dir) else set()
  specs: list[CheckSpec] = []
  n_missing = 0
  for name, raw, slp_md5, compression in rows:
    if slp_md5 in existing:
      specs.append((name, raw, os.path.join(output_dir, slp_md5), compression))
    else:
      n_missing += 1

  print(f"Checking {len(specs)} replays ({n_missing} missing from Parsed/).")
  if n_missing:
    print("Regenerate missing files with parse_local.py --reparse_missing_parsed.")

  if dry_run or not specs:
    conn.close()
    return

  n_checked = 0
  n_bad = 0
  n_errors = 0

  def record(result: CheckResult):
    nonlocal n_checked, n_bad, n_errors
    name, raw, data_ok, data_error, error = result
    if error is not None:
      n_errors += 1
      logging.warning('check failed for %s (%s): %s', name, raw, error)
      return
    try:
      conn.execute(
          'UPDATE replays SET data_ok = ?, data_error = ? '
          'WHERE name = ? AND raw = ?',
          (data_ok, data_error, name, raw))
    except sqlite3.OperationalError as e:
      # Skip the record; it stays data_ok NULL and a rerun will pick it up.
      n_errors += 1
      logging.warning('sqlite update failed for %s (%s): %s', name, raw, e)
      return
    n_checked += 1
    if not data_ok:
      n_bad += 1
    if n_checked % commit_interval == 0:
      conn.commit()

  try:
    if num_threads == 1:
      for spec in tqdm.tqdm(specs, desc='checking', unit='replay'):
        record(check_file(spec))
    else:
      # spawn (rather than fork) so workers don't inherit the sqlite fd.
      mp_context = multiprocessing.get_context('spawn')
      with concurrent.futures.ProcessPoolExecutor(
          num_threads, mp_context=mp_context) as pool:
        try:
          # Keep a bounded number of futures in flight to cap memory.
          spec_iter = iter(specs)
          pending: set[concurrent.futures.Future] = set()
          max_pending = 4 * num_threads
          with tqdm.tqdm(
              total=len(specs), desc='checking', unit='replay') as pbar:
            while True:
              for spec in itertools.islice(
                  spec_iter, max_pending - len(pending)):
                pending.add(pool.submit(check_file, spec))
              if not pending:
                break
              done, pending = concurrent.futures.wait(
                  pending, return_when=concurrent.futures.FIRST_COMPLETED)
              for future in done:
                record(future.result())
                pbar.update(1)
        except BaseException:
          # Without this, pool.__exit__ would block until every queued
          # task finishes before the error can even surface.
          print('Shutting down workers.')
          pool.shutdown(cancel_futures=True)
          raise
  finally:
    conn.commit()
    conn.close()

  print(f"Checked {n_checked} replays: "
        f"{n_bad} with bad data, {n_errors} errors.")


if __name__ == '__main__':
  ROOT = flags.DEFINE_string('root', None, 'root directory', required=True)
  THREADS = flags.DEFINE_integer('threads', 1, 'number of threads')
  RECHECK = flags.DEFINE_bool(
      'recheck', False,
      'Re-check replays that already have data_ok set, e.g. after adding a '
      'new check.')
  DRY_RUN = flags.DEFINE_bool('dry_run', False, 'only print counts')

  def main(_):
    run_check(
        ROOT.value,
        num_threads=THREADS.value,
        recheck=RECHECK.value,
        dry_run=DRY_RUN.value,
    )

  app.run(main)
