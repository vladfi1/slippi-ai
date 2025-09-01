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

def _check_worker(
    name: str,
    input_file: utils.ZipFile,
    output_file: utils.ZipFile,
    debug: bool = False,
    check_if_needed: bool = False,
) -> CheckResult:
  """Worker function for checking a single file."""
  try:
    tmpdir = utils.get_tmp_dir(in_memory=True)

    with input_file.extract(tmpdir) as path:
      input_game = peppi_py.read_slippi(path, rollback_mode=peppi_py.RollbackMode.LAST)

    if check_if_needed and not upgrade_slp.needs_upgrade(input_game):
      return CheckResult(name=name, input_archive=input_file.root, output_archive=output_file.root)

    with output_file.extract(tmpdir) as path:
      output_game = peppi_py.read_slippi(path, rollback_mode=peppi_py.RollbackMode.LAST)

    errors = upgrade_slp.check_games(input_game, output_game, debug=debug)
    return CheckResult(
        name=name,
        input_archive=input_file.root,
        output_archive=output_file.root,
        error=upgrade_slp.errors_to_str(errors),
    )
  except Exception as e:
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
  last_error: tp.Optional[tuple[str, str]] = None

  results: list[CheckResult] = []

  def _log():
    nonlocal last_log_time
    nonlocal last_error

    last_log_time = time.time()
    success_rate = successful_checks / pbar.n if pbar.n > 0 else 0
    logging.info(f'Success rate: {success_rate:.2%}')
    if last_error is not None:
      logging.error(f'Last error: {last_error}')
      last_error = None

  for result in results_iter:
    if result.error is None:
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

def main(_):
  input_path = Path(INPUT.value)
  output_path = Path(OUTPUT.value)
  num_threads = NUM_THREADS.value
  log_interval = LOG_INTERVAL.value
  debug = DEBUG.value
  check_if_needed = CHECK_IF_NEEDED.value

  todo: list[tuple[str | None, str, utils.ZipFile, utils.ZipFile]] = []

  if input_path.is_file():
    # Process single file
    if not str(input_path).endswith('.zip'):
      raise ValueError(f'Input file must be a .zip file: {input_path}')

    for xs in collect_files(str(input_path), str(output_path)):
      todo.append((None, *xs))

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

      for xs in collect_files(str(zip_file), str(output_file)):
        todo.append((str(rel_path), *xs))
  else:
    raise FileNotFoundError(f'Input path does not exist: {input_path}')

  print(f'Found {len(todo)} files to process')

  results: list[CheckResult] = []

  if num_threads == 1:
    # Single-threaded execution
    def results_iter():
      for rel_path, name, input_file, output_file in todo:
        yield _check_worker(
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
              _check_worker,
              name=name,
              input_file=input_file,
              output_file=output_file,
              debug=debug,
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
  successful_checks = sum(1 for r in results if r.error is None)
  failed_checks = len(results) - successful_checks
  print(f"\nCheck complete: {successful_checks} successful, {failed_checks} failed")

  results_json = [dataclasses.asdict(r) for r in results]
  with open('upgrade_check.json', 'w') as f:
    json.dump(results_json, f)

if __name__ == '__main__':
  app.run(main)
