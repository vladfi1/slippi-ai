"""Run parsing in the local filesystem.

We skip the "slp" step, parsing directly from raw archives.

Root
  Raw
  raw.json
  Parsed
  parsed.pq
"""

import concurrent.futures
import json
import os
import pickle
from typing import Optional

from absl import app, flags
import tqdm

import peppi_py

from slippi_db import parse_peppi
from slippi_db import preprocessing
from slippi_db import utils
from slippi_db import parsing_utils
from slippi_db.parsing_utils import CompressionType

def parse_slp(
    file: utils.LocalFile,
    output_dir: str,
    compression: CompressionType = CompressionType.NONE,
    compression_level: Optional[int] = None,
) -> dict:
  with file.extract() as path:
    with open(path, 'rb') as f:
      slp_bytes = f.read()
      slp_size = len(slp_bytes)
      md5 = utils.md5(slp_bytes)
      del slp_bytes

    result = dict(
        name=file.name,
        slp_md5=md5,
        slp_size=slp_size,
    )

    try:
      game = peppi_py.read_slippi(path)
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

    except KeyboardInterrupt as e:
      raise
    except BaseException as e:
      result.update(valid=False, reason=repr(e))
    # except:  # should be a catch-all, but sadly prevents KeyboardInterrupt?
    #   result.update(valid=False, reason='uncaught exception')

    return result

def parse_files(
    files: list[utils.LocalFile],
    output_dir: str,
    num_threads: int = 1,
    compression_options: dict = {},
) -> list[dict]:
  if num_threads == 1:
    return [
        parse_slp(f, output_dir, **compression_options)
        for f in tqdm.tqdm(files, unit='slp')]

  with concurrent.futures.ProcessPoolExecutor(num_threads) as pool:
    futures = [
        pool.submit(parse_slp, f, output_dir, **compression_options)
        for f in files]
    as_completed = concurrent.futures.as_completed(futures)
    results = [
        f.result() for f in
        tqdm.tqdm(as_completed, total=len(files), smoothing=1e-2, unit='slp')]
    return results

def parse_chunk(
    chunk: list[utils.LocalFile],
    output_dir: str,
    compression_options: dict = {},
    pool: Optional[concurrent.futures.ProcessPoolExecutor] = None,
) -> list[dict]:

  if pool is None:
    results = []
    for file in chunk:
      results.append(parse_slp(file, output_dir, **compression_options))
    return results
  else:
    futures = [
        pool.submit(parse_slp, f, output_dir, **compression_options)
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
            files, output_dir, compression_options, pool=pool)
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

def run_parsing(
    root: str,
    num_threads: int = 1,
    compression_options: dict = {},
    chunk_size_gb: float = 0.5,
    in_memory: bool = True,
    wipe: bool = False,
):
  raw_dir = os.path.join(root, 'Raw')

  raw_db_path = os.path.join(root, 'raw.json')
  if os.path.exists(raw_db_path):
    with open(raw_db_path) as f:
      raw_db = json.load(f)
  else:
    raw_db = []

  raw_by_name = {row['name']: row for row in raw_db}

  to_process = []
  for dirpath, _, filenames in os.walk(raw_dir):
    reldirpath = os.path.relpath(dirpath, raw_dir)
    for name in filenames:
      path = os.path.join(reldirpath, name).removeprefix('./')
      if path not in raw_by_name:
        raw_by_name[path] = dict(processed=False, name=path)
      if wipe or not raw_by_name[path]['processed']:
        to_process.append(path)

  print("To process:", to_process)

  output_dir = os.path.join(root, 'Parsed')
  os.makedirs(output_dir, exist_ok=True)

  # Special-case 7z files which we process in chunks.
  results = parse_7zs(
      raw_dir, to_process, output_dir, num_threads,
      compression_options, chunk_size_gb, in_memory)

  # Now handle zip files.
  print("Processing zip files.")
  slp_files = []
  raw_names = []
  for f in to_process:
    raw_path = os.path.join(raw_dir, f)
    if f.endswith('.zip'):
      fs = utils.traverse_slp_files_zip(raw_path)
    else:
      # print(f"Can't handle {f} yet.")
      continue
    print(f"Found {len(fs)} slp files in {f}")
    slp_files.extend(fs)
    raw_names.extend([f] * len(fs))

  zip_results = parse_files(
      slp_files, output_dir, num_threads, compression_options)
  assert len(zip_results) == len(slp_files)

  # Point back to raw file
  for result, raw_name in zip(zip_results, raw_names):
    result['raw'] = raw_name

  # Combine 7z and zip results
  results.extend(zip_results)

  # Now record the results.
  for raw_name in to_process:
    raw_by_name[raw_name].update(
        processed=True,
    )

  # Record raw metadata
  with open(raw_db_path, 'w') as f:
    json.dump(list(raw_by_name.values()), f, indent=2)

  # Record slp metadata.
  # TODO: column-major would be more efficient
  slp_db_path = os.path.join(root, 'parsed.pkl')
  if os.path.exists(slp_db_path):
    with open(slp_db_path, 'rb') as f:
      slp_meta = pickle.load(f)
    print(f"Loaded slp metadata with {len(slp_meta)} records.")
  else:
    slp_meta = []

  by_md5 = {row['slp_md5']: row for row in slp_meta}
  for result in results:
    by_md5[result['slp_md5']] = result

  with open(os.path.join(root, 'parsed.pkl'), 'wb') as f:
    pickle.dump(list(by_md5.values()), f)

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
      wipe=WIPE.value,
  )

if __name__ == '__main__':
  ROOT = flags.DEFINE_string('root', None, 'root directory', required=True)
  # MAX_FILES = flags.DEFINE_integer('max_files', None, 'max files to process')
  THREADS = flags.DEFINE_integer('threads', 1, 'number of threads')
  CHUNK_SIZE = flags.DEFINE_float('chunk_size', 0.5, 'max chunk size in GB')
  IN_MEMORY = flags.DEFINE_bool('in_memory', True, 'extract in memory')
  # LOG_INTERVAL = flags.DEFINE_integer('log_interval', 20, 'log interval')
  COMPRESSION = flags.DEFINE_enum_class(
      name='compression',
      default=parsing_utils.CompressionType.ZLIB,  # best one
      enum_class=parsing_utils.CompressionType,
      help='Type of compression to use.')
  COMPRESSION_LEVEL = flags.DEFINE_integer('compression_level', None, 'Compression level.')
  WIPE = flags.DEFINE_bool('wipe', False, 'Wipe existing metadata')

  app.run(main)
