import concurrent.futures
import dataclasses
import json
from pathlib import Path
import time
import typing as tp

from absl import app, flags, logging
import tqdm

import peppi_py

from slippi_db import upgrade_slp
from slippi_db import utils

@dataclasses.dataclass(slots=True)
class CheckResult:
  """Result of checking a file upgrade."""
  name: str
  input_archive: str
  output_archive: str
  error: tp.Optional[str] = None
  skipped: bool = False

def _check_worker(
    name: str,
    input_file: utils.SlpZipFile,
    output_file: utils.SlpZipFile,
    debug: bool = False,
    check_if_needed: bool = False,
) -> CheckResult:
  """Worker function for checking a single file."""
  tmpdir = utils.get_tmp_dir(in_memory=True)

  with input_file.extract(tmpdir) as path:
    input_game = peppi_py.read_slippi(path, rollback_mode=peppi_py.RollbackMode.LAST)

  if check_if_needed and not upgrade_slp.needs_upgrade(input_game):
    return CheckResult(
        name=name,
        input_archive=input_file.root,
        output_archive=output_file.root,
        skipped=True
    )

  with output_file.extract(tmpdir) as path:
    output_game = peppi_py.read_slippi(path, rollback_mode=peppi_py.RollbackMode.LAST)

  errors = upgrade_slp.check_games(input_game, output_game, debug=debug)
  return CheckResult(
      name=name,
      input_archive=input_file.root,
      output_archive=output_file.root,
      error=upgrade_slp.errors_to_str(errors) if errors else None,
  )

def _check_worker_safe(
    name: str,
    input_file: utils.SlpZipFile,
    output_file: utils.SlpZipFile,
    debug: bool = False,
    **kwargs,
) -> CheckResult:
  kwargs.update(
      name=name,
      input_file=input_file,
      output_file=output_file,
      debug=debug,
  )

  if debug:
    return _check_worker(**kwargs)

  try:
    return _check_worker(**kwargs)
  except BaseException as e:
    return CheckResult(
        name=name,
        input_archive=input_file.root,
        output_archive=output_file.root,
        error=f"{type(e).__name__}: {str(e)}"
    )

def collect_files(input_archive, output_archive):
  input_files = {f.name: f for f in utils.traverse_slp_files_zip(input_archive)}
  output_files = {f.name: f for f in utils.traverse_slp_files_zip(output_archive)}

  results = []

  not_found = []

  for name in output_files:
    if name not in input_files:
      not_found.append(name)
      # raise FileNotFoundError(f'File {name} is in output but not input')
      continue

    results.append((name, input_files[name], output_files[name]))

  if not_found:
    logging.warning(f'{len(not_found)} files found in output {output_archive} but not input {input_archive}')

  return results

def _monitor_results(
    results_iter: tp.Iterator[CheckResult],
    total_files: int,
    log_interval: int = 30,
) -> list[CheckResult]:
  """Monitor and collect results from parallel workers."""
  pbar = tqdm.tqdm(total=total_files, desc="Checking", unit="file", smoothing=0)

  last_log_time = 0
  successful_checks = 0
  skipped_checks = 0
  last_error: tp.Optional[tuple[str, str]] = None

  results: list[CheckResult] = []

  def _log():
    nonlocal last_log_time
    nonlocal last_error

    last_log_time = time.time()
    non_skipped = pbar.n - skipped_checks
    success_rate = successful_checks / non_skipped if non_skipped > 0 else 0
    logging.info(f'Success rate: {success_rate:.2%} ({skipped_checks} skipped)')
    if last_error is not None:
      logging.error(f'Last error: {last_error}')
      last_error = None

  for result in results_iter:
    if result.skipped:
      skipped_checks += 1
    elif result.error is None:
      successful_checks += 1
    else:
      last_error = result.name, result.error

    results.append(result)
    pbar.update(1)

    if time.time() - last_log_time > log_interval:
      _log()

  pbar.close()

  _log()

  return results


INPUT = flags.DEFINE_string('input', None, 'Input archive or directory to convert.', required=True)
OUTPUT = flags.DEFINE_string('output', None, 'Output archive or directory to write.', required=True)
NUM_THREADS = flags.DEFINE_integer('threads', 1, 'Number of threads to use for conversion.')
LOG_INTERVAL = flags.DEFINE_integer('log_interval', 30, 'Interval in seconds to log progress during conversion.')
DEBUG = flags.DEFINE_bool('debug', False, 'Enable debug mode with more verbose output.')
CHECK_IF_NEEDED = flags.DEFINE_bool('check_if_needed', False, 'Check if the file would have been upgraded, skip if not.')
LIMIT = flags.DEFINE_integer('limit', None, 'Maximum number of files to check. When set, samples files intelligently across archives.')

def sample_files(
    archives_todo: dict[str, list[tuple[str, utils.SlpZipFile, utils.SlpZipFile]]],
    limit: int
) -> list[tuple[str | None, str, utils.SlpZipFile, utils.SlpZipFile]]:
  """Sample files from archives using a mixed strategy.

  Takes half of the limit evenly from each archive, and the other half
  proportionally based on archive sizes.
  """
  import random

  result = []
  total_files = sum(len(files) for files in archives_todo.values())

  if total_files <= limit:
    # If we have fewer files than the limit, return all
    for archive_path, files in archives_todo.items():
      for file_tuple in files:
        result.append((archive_path if len(archives_todo) > 1 else None, *file_tuple))
    return result

  # Calculate per-archive allocations
  num_archives = len(archives_todo)
  even_quota = limit // 2  # Half of limit distributed evenly
  proportional_quota = limit - even_quota  # Remaining distributed by size

  # Even distribution
  even_per_archive = even_quota // num_archives
  even_remainder = even_quota % num_archives

  # Proportional distribution based on archive sizes
  archive_sizes = {path: len(files) for path, files in archives_todo.items()}

  samples_per_archive = {}
  for i, (archive_path, size) in enumerate(archive_sizes.items()):
    # Even part: each archive gets even_per_archive, plus 1 for remainder
    even_samples = even_per_archive + (1 if i < even_remainder else 0)

    # Proportional part: based on archive size
    proportion = size / total_files
    proportional_samples = int(proportional_quota * proportion)

    samples_per_archive[archive_path] = min(
        even_samples + proportional_samples,
        size  # Don't sample more than available
    )

  # Adjust for rounding errors
  total_sampled = sum(samples_per_archive.values())
  if total_sampled < limit:
    # Add remaining samples to largest archives that have room
    remaining = limit - total_sampled
    sorted_archives = sorted(
        archives_todo.keys(),
        key=lambda p: archive_sizes[p] - samples_per_archive[p],
        reverse=True
    )
    for archive_path in sorted_archives:
      if remaining <= 0:
        break
      available = archive_sizes[archive_path] - samples_per_archive[archive_path]
      add = min(remaining, available)
      samples_per_archive[archive_path] += add
      remaining -= add

  # Sample from each archive
  for archive_path, files in archives_todo.items():
    num_samples = samples_per_archive[archive_path]
    if num_samples >= len(files):
      sampled = files
    else:
      # Random sampling without replacement
      sampled = random.sample(files, num_samples)

    logging.info(f'Sampling {num_samples}/{len(files)} files from {archive_path}')

    for file_tuple in sampled:
      result.append((archive_path if len(archives_todo) > 1 else None, *file_tuple))

  random.shuffle(result)  # Shuffle to mix archives
  return result

def main(_):
  input_path = Path(INPUT.value)
  output_path = Path(OUTPUT.value)
  num_threads = NUM_THREADS.value
  log_interval = LOG_INTERVAL.value
  debug = DEBUG.value
  check_if_needed = CHECK_IF_NEEDED.value
  limit = LIMIT.value

  todo: list[tuple[str | None, str, utils.SlpZipFile, utils.SlpZipFile]] = []
  archives_todo: dict[str, list[tuple[str, utils.SlpZipFile, utils.SlpZipFile]]] = {}

  if input_path.is_file():
    # Process single file
    if not str(input_path).endswith('.zip'):
      raise ValueError(f'Input file must be a .zip file: {input_path}')

    archive_files = collect_files(str(input_path), str(output_path))
    if archive_files:
      archives_todo[str(input_path)] = archive_files

  elif input_path.is_dir():
    # Process directory recursively
    if not output_path.is_dir():
      raise FileNotFoundError(f'Output must be a directory')

    # Find all zip files recursively
    zip_files = list(input_path.rglob('*.zip'))
    print(f'Found {len(zip_files)} zip files to process')

    for zip_file in zip_files:
      # Calculate relative path from input directory
      rel_path = zip_file.relative_to(input_path)
      output_file = output_path / rel_path

      if not output_file.is_file():
        raise FileNotFoundError(f'Output file does not exist: {output_file}')

    for zip_file in zip_files:
      # Calculate relative path from input directory
      rel_path = zip_file.relative_to(input_path)
      output_file = output_path / rel_path

      archive_files = collect_files(str(zip_file), str(output_file))
      if archive_files:
        archives_todo[str(rel_path)] = archive_files
  else:
    raise FileNotFoundError(f'Input path does not exist: {input_path}')

  # Apply sampling if limit is set
  if limit is not None and archives_todo:
    todo = sample_files(archives_todo, limit)
    print(f'Found {sum(len(files) for files in archives_todo.values())} total files, sampling {len(todo)} files')
  else:
    # No sampling, use all files
    for archive_path, files in archives_todo.items():
      for file_tuple in files:
        todo.append((archive_path if len(archives_todo) > 1 else None, *file_tuple))
    print(f'Found {len(todo)} files to process')

  results: list[CheckResult] = []

  if num_threads == 1:
    # Single-threaded execution
    def results_iter():
      for rel_path, name, input_file, output_file in todo:
        yield _check_worker_safe(
            name, input_file, output_file,
            debug=debug, check_if_needed=check_if_needed)

    results = _monitor_results(
        results_iter(),
        total_files=len(todo),
        log_interval=log_interval,
    )
  else:
    # Multi-threaded execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
      try:
        futures: list[concurrent.futures.Future] = []
        for rel_path, name, input_file, output_file in tqdm.tqdm(
            todo, desc='Submitting'):
          futures.append(executor.submit(
              _check_worker_safe,
              name=name,
              input_file=input_file,
              output_file=output_file,
              debug=False,
              check_if_needed=check_if_needed,
          ))

        def results_iter() -> tp.Iterator[CheckResult]:
          for future in concurrent.futures.as_completed(futures):
            yield future.result()

        results = _monitor_results(
            results_iter(),
            total_files=len(futures),
            log_interval=log_interval,
        )
      except KeyboardInterrupt:
        print('KeyboardInterrupt, shutting down')
        executor.shutdown(cancel_futures=True)
        raise

  # Print summary
  skipped_checks = sum(1 for r in results if r.skipped)
  non_skipped_results = [r for r in results if not r.skipped]
  successful_checks = sum(1 for r in non_skipped_results if r.error is None)
  failed_checks = len(non_skipped_results) - successful_checks

  print(f"\nCheck complete: {successful_checks} successful, {failed_checks} failed, {skipped_checks} skipped")

  if non_skipped_results:
    overall_success_rate = successful_checks / len(non_skipped_results) * 100
    print(f"Overall success rate (excluding skipped): {overall_success_rate:.1f}%")

  # Calculate per-archive statistics
  from collections import defaultdict
  archive_stats = defaultdict(lambda: {'success': 0, 'failed': 0, 'skipped': 0})

  for result in results:
    archive_key = result.input_archive
    if result.skipped:
      archive_stats[archive_key]['skipped'] += 1
    elif result.error is None:
      archive_stats[archive_key]['success'] += 1
    else:
      archive_stats[archive_key]['failed'] += 1

  # Print per-archive success rates
  print("\nPer-archive success rates (excluding skipped):")
  for archive_path in sorted(archive_stats.keys()):
    stats = archive_stats[archive_path]
    non_skipped = stats['success'] + stats['failed']
    if non_skipped > 0:
      success_rate = stats['success'] / non_skipped * 100
      print(f"  {archive_path}: {stats['success']}/{non_skipped} ({success_rate:.1f}%) [{stats['skipped']} skipped]")
    else:
      print(f"  {archive_path}: all {stats['skipped']} files skipped")

  results_json = [dataclasses.asdict(r) for r in results]
  with open('upgrade_check.json', 'w') as f:
    json.dump(results_json, f, indent=2)

if __name__ == '__main__':
  app.run(main)
