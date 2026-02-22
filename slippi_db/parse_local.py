"""Run parsing in the local filesystem.

It is assumed that everything is organized under a "root" directory:

Root
  Raw
  raw.json
  Parsed
  parsed.sqlite
  meta.json

Raw contains .zip and .7z archives of .slp files, possibly nested under
subdirectories. The raw.json metadata file contains information about each
raw archive, including whether it has been processed. Once a raw archive has
been processed, it may be removed to save space.

The Parsed directory is populated by this script with a parquet file for each
processed .slp file. These files are named by the MD5 hash of the .slp file,
and are used by imitation learning. The parsed.sqlite database contains
metadata about each processed .slp in Parsed.

The meta.json file is created by scripts/make_local_dataset.py and is used by
imitation learning to know which files to train on.

Usage: python slippi_db/parse_local.py --root=Root [--threads N] [--dry_run]

This will process all unprocessed .zip and .7z files in the Raw directory,
overwriting any existing files in Parsed, and will update parsed.sqlite.
"""

import concurrent.futures
import json
import logging
import os
import sqlite3
import sys
import tempfile
import time
import typing as tp
from typing import Any, Optional

from absl import app, flags
import tqdm

from slippi_db import parse_peppi
from slippi_db import preprocessing
from slippi_db import upgrade_slp
from slippi_db import utils
from slippi_db import parsing_utils
from slippi_db.parsing_utils import CompressionType

## Parsed SQLite DB helpers
# Schema matches slippi_db/scripts/convert_parsed_to_sqlite.py

PARSED_SCHEMA = """
CREATE TABLE IF NOT EXISTS replays (
    name TEXT NOT NULL,
    raw TEXT NOT NULL,
    slp_md5 TEXT,
    slp_size INTEGER,
    start_at TEXT,
    played_on TEXT,
    last_frame INTEGER,
    slippi_version TEXT,
    stage INTEGER,
    timer INTEGER,
    is_teams INTEGER,
    num_players INTEGER,
    winner INTEGER,
    valid INTEGER NOT NULL,
    is_training INTEGER,
    not_training_reason TEXT,
    parse_error TEXT,
    pq_size INTEGER,
    compression TEXT,
    match_id TEXT,
    match_game INTEGER,
    match_tiebreaker INTEGER,
    p0_port INTEGER,
    p0_character INTEGER,
    p0_type INTEGER,
    p0_name_tag TEXT,
    p0_netplay_name TEXT,
    p0_netplay_code TEXT,
    p0_netplay_suid TEXT,
    p0_damage_taken REAL,
    p1_port INTEGER,
    p1_character INTEGER,
    p1_type INTEGER,
    p1_name_tag TEXT,
    p1_netplay_name TEXT,
    p1_netplay_code TEXT,
    p1_netplay_suid TEXT,
    p1_damage_taken REAL,
    PRIMARY KEY (name, raw)
);
"""

PARSED_COLUMNS = [
    'name', 'raw', 'slp_md5', 'slp_size', 'start_at', 'played_on',
    'last_frame', 'slippi_version', 'stage', 'timer', 'is_teams',
    'num_players', 'winner', 'valid', 'is_training', 'not_training_reason',
    'parse_error', 'pq_size', 'compression',
    'match_id', 'match_game', 'match_tiebreaker',
    'p0_port', 'p0_character', 'p0_type', 'p0_name_tag',
    'p0_netplay_name', 'p0_netplay_code', 'p0_netplay_suid', 'p0_damage_taken',
    'p1_port', 'p1_character', 'p1_type', 'p1_name_tag',
    'p1_netplay_name', 'p1_netplay_code', 'p1_netplay_suid', 'p1_damage_taken',
]

PARSED_INSERT_SQL = f"""
INSERT OR REPLACE INTO replays ({', '.join(PARSED_COLUMNS)})
VALUES ({', '.join('?' * len(PARSED_COLUMNS))})
"""


def _format_version(version: tuple | list | None) -> str | None:
  if version is None:
    return None
  return '.'.join(str(v) for v in version)


def _extract_player(player: dict, prefix: str) -> dict[str, Any]:
  netplay = player.get('netplay') or {}
  return {
    f'{prefix}_port': player.get('port'),
    f'{prefix}_character': player.get('character'),
    f'{prefix}_type': player.get('type'),
    f'{prefix}_name_tag': player.get('name_tag'),
    f'{prefix}_netplay_name': netplay.get('name'),
    f'{prefix}_netplay_code': netplay.get('code'),
    f'{prefix}_netplay_suid': netplay.get('suid'),
    f'{prefix}_damage_taken': player.get('damage_taken'),
  }


def result_to_tuple(row: dict) -> tuple:
  """Convert a parse result dict to a tuple for SQLite insertion."""
  match = row.get('match') or {}
  players = row.get('players') or []

  flat = {
    'name': row.get('name'),
    'raw': row.get('raw'),
    'slp_md5': row.get('slp_md5'),
    'slp_size': row.get('slp_size'),
    'start_at': row.get('startAt'),
    'played_on': row.get('playedOn'),
    'last_frame': row.get('lastFrame'),
    'slippi_version': _format_version(row.get('slippi_version')),
    'stage': row.get('stage'),
    'timer': row.get('timer'),
    'is_teams': row.get('is_teams'),
    'num_players': row.get('num_players'),
    'winner': row.get('winner'),
    'valid': row.get('valid'),
    'is_training': row.get('is_training'),
    'not_training_reason': row.get('not_training_reason'),
    'parse_error': row.get('reason'),
    'pq_size': row.get('pq_size'),
    'compression': row.get('compression'),
    'match_id': match.get('id'),
    'match_game': match.get('game'),
    'match_tiebreaker': match.get('tiebreaker'),
  }

  if len(players) >= 1:
    flat.update(_extract_player(players[0], 'p0'))
  else:
    flat.update(_extract_player({}, 'p0'))

  if len(players) >= 2:
    flat.update(_extract_player(players[1], 'p1'))
  else:
    flat.update(_extract_player({}, 'p1'))

  return tuple(flat[col] for col in PARSED_COLUMNS)


def parse_slp(
    file: utils.LocalFile,
    output_dir: str,
    tmpdir: Optional[str],
    compression: CompressionType = CompressionType.NONE,
    compression_level: Optional[int] = None,
) -> dict:
  slp_bytes = file.read()
  slp_size = len(slp_bytes)
  md5 = utils.md5(slp_bytes)

  result = dict(
      name=file.name,
      slp_md5=md5,
      slp_size=slp_size,
  )

  with tempfile.TemporaryDirectory(dir=tmpdir) as tmp_parent:
    path = os.path.join(tmp_parent, 'game.slp')
    with open(path, 'wb') as f:
      f.write(slp_bytes)
    del slp_bytes

    game = parse_peppi.read_slippi(path)
    metadata = preprocessing.get_metadata(game)
    is_training, reason = preprocessing.is_training_replay(metadata)

    result.update(metadata)  # nest?
    result.update(
        valid=True,
        is_training=is_training,
        not_training_reason=reason,
    )

    if is_training:
      game = parse_peppi.from_peppi(game)
      game_bytes = parsing_utils.convert_game(
        game, compression=compression, compression_level=compression_level)
      result.update(
          pq_size=len(game_bytes),
          compression=compression.value,
      )

      # TODO: consider writing to raw_name/slp_name
      with open(os.path.join(output_dir, md5), 'wb') as f:
        f.write(game_bytes)

  return result

def parse_slp_safe(file: utils.LocalFile, *args, debug: bool = False, **kwargs):
  if debug:
    return parse_slp(file, *args, **kwargs)

  try:
    return parse_slp(file, *args, **kwargs)
  except KeyboardInterrupt:
    raise
  except BaseException as e:
    return dict(name=file.name, valid=False, reason=repr(e))
  # except:  # should be a catch-all, but sadly prevents KeyboardInterrupt?
  #   result.update(valid=False, reason='uncaught exception')


def parse_slp_with_index(index: int, *args, **kwargs):
  return index, parse_slp_safe(*args, **kwargs)

def _monitor_results(
    results_iter: tp.Iterable[dict],
    total_files: int,
    log_interval: int = 30,
) -> list[dict]:
  """Monitor parsing results and log progress periodically."""
  pbar = tqdm.tqdm(total=total_files, desc="Parsing", unit="slp", smoothing=0)

  last_log_time = 0
  successful_parses = 0
  last_error: Optional[tuple[str, str]] = None

  results: list[dict] = []

  for result in results_iter:
    if result['valid']:
      successful_parses += 1
    else:
      last_error = (result['name'], result['reason'])

    results.append(result)
    pbar.update(1)

    if time.time() - last_log_time > log_interval:
      last_log_time = time.time()
      success_rate = successful_parses / pbar.n
      logging.info(f'Success rate: {success_rate:.2%}')
      if last_error is not None:
        logging.error(f'Last error: {last_error}')
        last_error = None

  pbar.close()

  return results

def parse_files(
    files: list[utils.LocalFile],
    output_dir: str,
    tmpdir: Optional[str],
    num_threads: int = 1,
    compression_options: dict = {},
    log_interval: int = 30,
    debug: bool = False,
) -> list[dict]:
  parse_slp_kwargs = dict(
      output_dir=output_dir,
      tmpdir=tmpdir,
      **compression_options,
  )

  if num_threads == 1:
    def results_iter():
      for f in files:
        yield parse_slp_safe(f, debug=debug, **parse_slp_kwargs)

    return _monitor_results(results_iter(), total_files=len(files), log_interval=log_interval)

  with concurrent.futures.ProcessPoolExecutor(num_threads) as pool:
    try:
      if sys.version_info < (3, 12):
        logging.warning(
            'Submitting large numbers of tasks to the process pool may cause '
            'a deadlock in python < 3.12, see '
            'https://github.com/python/cpython/issues/105829')

      futures = []
      for i, f in enumerate(tqdm.tqdm(files, desc='Submitting', unit='slp')):
        futures.append(pool.submit(parse_slp_with_index, i, f, **parse_slp_kwargs))

      results = [None] * len(files)

      def results_iter():
        for future in concurrent.futures.as_completed(futures):
          index, result = future.result()
          results[index] = result
          yield result

      _monitor_results(results_iter(), total_files=len(files), log_interval=log_interval)

      return results
    except KeyboardInterrupt:
      print('KeyboardInterrupt, shutting down')
      pool.shutdown(cancel_futures=True)
      raise

def parse_chunk(
    chunk: list[utils.LocalFile],
    output_dir: str,
    tmpdir: str,
    compression_options: dict = {},
    pool: Optional[concurrent.futures.ProcessPoolExecutor] = None,
) -> list[dict]:
  parse_slp_kwargs = dict(
      output_dir=output_dir,
      tmpdir=tmpdir,
      **compression_options,
  )

  if pool is None:
    results = []
    for file in chunk:
      results.append(parse_slp(file, **parse_slp_kwargs))
    return results
  else:
    futures = [
        pool.submit(parse_slp, f, **parse_slp_kwargs)
        for f in chunk]
    return [f.result() for f in futures]

def parse_7zs(
    raw_dir: str,
    to_process: list[str],
    output_dir: str,
    num_threads: int = 1,
    compression_options: dict = {},
    chunk_size_gb: float = 0.5,
    in_memory: bool = True,
) -> list[dict]:
  print("Processing 7z files.")
  to_process = [f for f in to_process if f.endswith('.7z')]
  if not to_process:
    print("No 7z files to process.")
    return []

  chunks: list[utils.SevenZipChunk] = []
  raw_names = []  # per chunk
  file_sizes = []
  for f in to_process:
    raw_path = os.path.join(raw_dir, f)
    new_chunks = utils.traverse_7z_fast(raw_path, chunk_size_gb=chunk_size_gb)
    chunks.extend(new_chunks)
    raw_names.extend([f] * len(new_chunks))
    file_sizes.append(os.path.getsize(raw_path))

  # print stats on 7z files?
  chunk_sizes = [len(c.files) for c in chunks]
  mean_chunk_size = sum(chunk_sizes) / len(chunks)
  total_size_gb = sum(file_sizes) / 1024**3
  print(f"Found {len(file_sizes)} 7z files totalling {total_size_gb:.2f} GB.")
  print(f"Split into {len(chunks)} chunks, mean size {mean_chunk_size:.1f}")

  # Would be nice to tqdm on files instead of chunks.
  iter_chunks = tqdm.tqdm(chunks, unit='chunk')
  chunks_and_raw_names = zip(iter_chunks, raw_names)

  results = []
  if num_threads == 1:
    pool = None
  else:
    pool = concurrent.futures.ProcessPoolExecutor(num_threads)

  for chunk, raw_name in chunks_and_raw_names:
    with chunk.extract(in_memory) as files:
      try:
        chunk_results = parse_chunk(
            files, output_dir,
            tmpdir=utils.get_tmp_dir(in_memory=in_memory),
            compression_options=compression_options,
            pool=pool)
      except BaseException as e:
        # print(e)
        if pool is not None:
          pool.shutdown()  # shutdown before cleaning up tmpdir
        raise e

    for result in chunk_results:
      result['raw'] = raw_name
    results.extend(chunk_results)

    # TODO: give updates on valid files
    # valid = [r['valid'] for r in chunk_results]
    # num_valid = sum(valid)
    # print(f"Chunk {raw_name} valid: {num_valid}/{len(valid)}")

  if pool is not None:
    pool.shutdown()

  return results

def _swap_upgraded_files(
    files: list[utils.ZipFile],
    raw: str,
    upgraded_dir: str,
    db_conn: sqlite3.Connection,
) -> list[utils.ZipFile]:
  """Replace raw files with upgraded versions where available.

  Modifies and returns the list in-place.
  """
  db_results = upgrade_slp.query_upgrade_results(db_conn, raw)
  successes = {
    name for name, result in db_results.items() if result == 'success'
  }
  if not successes:
    return files

  upgraded_path = os.path.join(upgraded_dir, raw)
  if not os.path.exists(upgraded_path):
    logging.warning(
      '%s: %d successful upgrades but upgraded archive not found at %s',
      raw, len(successes), upgraded_path)
    return files

  upgraded_files = utils.traverse_slp_files_zip(upgraded_path)
  upgraded_by_base = {f.base_name: f for f in upgraded_files}
  num_swapped = 0
  for i, f in enumerate(files):
    if f.base_name in successes and f.base_name in upgraded_by_base:
      files[i] = upgraded_by_base[f.base_name]
      num_swapped += 1
  logging.info(
    '%s: swapped %d/%d files from upgraded archive',
    raw, num_swapped, len(files))
  return files


def _query_missing_netplay(parsed_db_path: str) -> dict[str, list[str]]:
  """Query parsed.sqlite for valid replays missing netplay info.

  Returns {raw_archive: [base_name, ...]} for files to re-parse.
  """
  if not os.path.exists(parsed_db_path):
    return {}

  conn = sqlite3.connect(f'file:{parsed_db_path}?mode=ro', uri=True)
  cursor = conn.execute("""
    SELECT name, raw FROM replays
    WHERE valid = 1
      AND p0_netplay_name IS NULL
      AND p0_netplay_code IS NULL
      AND p1_netplay_name IS NULL
      AND p1_netplay_code IS NULL
  """)
  rows = cursor.fetchall()
  conn.close()

  if not rows:
    return {}

  by_archive: dict[str, list[str]] = {}
  for name, raw in rows:
    # name is "base_name.slp", strip the suffix to get base_name
    base_name = name.removesuffix('.slp')
    by_archive.setdefault(raw, []).append(base_name)

  return by_archive


def run_parsing(
    root: str,
    num_threads: int = 1,
    compression_options: dict = {},
    chunk_size_gb: float = 0.5,
    in_memory: bool = True,
    reprocess: bool = False,
    reparse_missing_netplay: bool = False,
    dry_run: bool = False,
    log_interval: int = 30,
    debug: bool = False,
):
  # Cache tmp dir once
  tmpdir = utils.get_tmp_dir(in_memory=in_memory)

  raw_dir = os.path.join(root, 'Raw')

  raw_db_path = os.path.join(root, 'raw.json')
  if os.path.exists(raw_db_path):
    with open(raw_db_path, 'r') as f:
      raw_db = json.load(f)
  else:
    raw_db = []

  raw_by_name = {row['name']: row for row in raw_db}

  to_process: list[str] = []
  for dirpath, _, filenames in os.walk(raw_dir):
    reldirpath = os.path.relpath(dirpath, raw_dir)
    for name in filenames:
      path = os.path.join(reldirpath, name).removeprefix('./')
      if path not in raw_by_name:
        raw_by_name[path] = dict(processed=False, name=path)
      if reprocess or not raw_by_name[path]['processed']:
        to_process.append(path)

  print("To process:", to_process)

  # Open upgrades DB if it exists.
  upgrades_db_path = os.path.join(root, 'upgrades.sqlite')
  if os.path.exists(upgrades_db_path):
    db_conn = sqlite3.connect(f'file:{upgrades_db_path}?mode=ro', uri=True)
    logging.info('Opened upgrades DB at %s', upgrades_db_path)
  else:
    db_conn = None

  upgraded_dir = os.path.join(root, 'Upgraded')

  # Now handle zip files.
  slp_files: list[utils.LocalFile] = []
  raw_names: list[str] = []
  for raw in to_process:
    raw_path = os.path.join(raw_dir, raw)
    if not raw.endswith('.zip'):
      continue
    files = utils.traverse_slp_files_zip(raw_path)
    if db_conn is not None:
      _swap_upgraded_files(files, raw, upgraded_dir, db_conn)
    print(f"Found {len(files)} slp files in {raw}")
    slp_files.extend(files)
    raw_names.extend([raw] * len(files))

  # Re-parse files missing netplay info from already-processed archives.
  if reparse_missing_netplay:
    parsed_db_path = os.path.join(root, 'parsed.sqlite')
    missing = _query_missing_netplay(parsed_db_path)
    # Exclude archives we're already fully reprocessing.
    to_process_set = set(to_process)
    missing = {k: v for k, v in missing.items() if k not in to_process_set}
    if missing:
      total_files = sum(len(v) for v in missing.values())
      print(f"Re-parsing {total_files} files missing netplay info "
            f"from {len(missing)} archives.")
    else:
      print("No files missing netplay info.")

    for raw, base_names in missing.items():
      raw_path = os.path.join(raw_dir, raw)
      all_files = utils.traverse_slp_files_zip(raw_path)
      if db_conn is not None:
        _swap_upgraded_files(all_files, raw, upgraded_dir, db_conn)
      need = set(base_names)
      selected = [f for f in all_files if f.base_name in need]
      slp_files.extend(selected)
      raw_names.extend([raw] * len(selected))

  if dry_run:
    return

  output_dir = os.path.join(root, 'Parsed')
  os.makedirs(output_dir, exist_ok=True)

  # Special-case 7z files which we process in chunks.
  results = parse_7zs(
      raw_dir, to_process, output_dir, num_threads,
      compression_options, chunk_size_gb, in_memory)

  # TODO: handle raw .slp and .slp.gz files

  print("Processing zip files.")
  zip_results = parse_files(
      slp_files, output_dir,
      tmpdir=tmpdir,
      num_threads=num_threads,
      compression_options=compression_options,
      log_interval=log_interval,
      debug=debug)
  assert len(zip_results) == len(slp_files)

  # Point back to raw file
  for result, raw_name in zip(zip_results, raw_names):
    result['raw'] = raw_name

  # Combine 7z and zip results
  results.extend(zip_results)

  if results:
    num_valid = sum(r['valid'] for r in results)
    print(f"Processed {num_valid}/{len(results)} valid files.")

  # Now record the results.
  for raw_name in to_process:
    raw_by_name[raw_name].update(
        processed=True,
    )

  # Record raw metadata
  with open(raw_db_path, 'w') as f:
    json.dump(list(raw_by_name.values()), f, indent=2)

  # Record slp metadata in SQLite.
  parsed_db_path = os.path.join(root, 'parsed.sqlite')
  parsed_conn = sqlite3.connect(parsed_db_path)
  parsed_conn.executescript(PARSED_SCHEMA)

  tuples = [result_to_tuple(r) for r in results]
  parsed_conn.executemany(PARSED_INSERT_SQL, tuples)
  parsed_conn.commit()

  count = parsed_conn.execute("SELECT COUNT(*) FROM replays").fetchone()[0]
  print(f"parsed.sqlite now has {count} records.")
  parsed_conn.close()

  if db_conn is not None:
    db_conn.close()

if __name__ == '__main__':
  ROOT = flags.DEFINE_string('root', None, 'root directory', required=True)
  # MAX_FILES = flags.DEFINE_integer('max_files', None, 'max files to process')
  THREADS = flags.DEFINE_integer('threads', 1, 'number of threads')
  CHUNK_SIZE = flags.DEFINE_float('chunk_size', 0.5, 'max chunk size in GB')
  IN_MEMORY = flags.DEFINE_bool('in_memory', True, 'extract in memory')
  LOG_INTERVAL = flags.DEFINE_integer('log_interval', 30, 'seconds between progress logs')
  COMPRESSION = flags.DEFINE_enum_class(
      name='compression',
      default=parsing_utils.CompressionType.ZLIB,  # best one
      enum_class=parsing_utils.CompressionType,
      help='Type of compression to use.')
  COMPRESSION_LEVEL = flags.DEFINE_integer('compression_level', None, 'Compression level.')
  REPROCESS = flags.DEFINE_bool('reprocess', False, 'Reprocess raw archives.')
  REPARSE_MISSING_NETPLAY = flags.DEFINE_bool(
      'reparse_missing_netplay', False,
      'Re-parse valid replays that are missing netplay info.')
  DRY_RUN = flags.DEFINE_bool('dry_run', False, 'dry run')
  DEBUG = flags.DEFINE_bool('debug', False, 'debug mode (no exception catching)')

  def main(_):
    run_parsing(
        ROOT.value,
        num_threads=THREADS.value,
        chunk_size_gb=CHUNK_SIZE.value,
        in_memory=IN_MEMORY.value,
        compression_options=dict(
            compression=COMPRESSION.value,
            compression_level=COMPRESSION_LEVEL.value,
        ),
        reprocess=REPROCESS.value,
        reparse_missing_netplay=REPARSE_MISSING_NETPLAY.value,
        dry_run=DRY_RUN.value,
        log_interval=LOG_INTERVAL.value,
        debug=DEBUG.value,
    )

  app.run(main)
