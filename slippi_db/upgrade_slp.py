import concurrent.futures
import configparser
import dataclasses
import gzip
import importlib
import json
import os
import psutil
import subprocess
import tempfile
import time
from typing import Optional, Iterator, TypeVar
import zipfile

from absl import logging
import tqdm

import melee
import peppi_py

from slippi_ai.types import game_array_to_nt
from slippi_ai.utils import check_same_structure
from slippi_db import utils
from slippi_db import parse_peppi

@dataclasses.dataclass
class DolphinConfig:
  dolphin_path: str
  ssbm_iso_path: str

class DolphinTimeoutError(TimeoutError):
  """Custom exception for Dolphin command timeouts."""
  pass

class DolphinExecutionError(RuntimeError):
  """Custom exception for Dolphin command execution errors."""
  pass

def upgrade_slp(
    input_path: str,
    output_path: str,
    dolphin_config: DolphinConfig,
    in_memory: bool = True,
    time_limit: Optional[int] = None,
    copy_slp_metadata_binary: str = 'copy_slp_metadata',
    headless: bool = True,
    fast_forward: bool = True,
):
  """Upgrade a Slippi replay file to the latest version."""

  # TODO: check that we have the right dolphin executable

  tmp_parent_dir = None
  if in_memory:
    tmp_parent_dir = utils.get_tmp_dir(in_memory=True)

  with tempfile.TemporaryDirectory(dir=tmp_parent_dir) as tmp_dir:
    user_dir = os.path.join(tmp_dir, 'user')
    os.makedirs(user_dir)

    replay_dir = os.path.join(tmp_dir, 'replays')
    os.makedirs(replay_dir)

    replay_json_path = os.path.join(tmp_dir, 'replay.json')

    replay_json = {
        'replay': os.path.abspath(input_path),
        'shouldResync': True,
        # 'rollbackDisplayMethod': 'normal',
    }
    if fast_forward:
      replay_json['startFrame'] = 1000000
    with open(replay_json_path, 'w') as f:
      json.dump(replay_json, f)

    config_path = os.path.join(user_dir, 'Config')
    os.makedirs(config_path)
    dolphin_ini_path = os.path.join(config_path, 'Dolphin.ini')

    config = configparser.ConfigParser()
    for section in ['Slippi', 'Core', 'DSP']:
      config.add_section(section)

    # Assumes Mainline playback dolphin
    config.set('Slippi', 'SaveReplays', 'True')
    config.set('Slippi', 'ReplayMonthlyFolders', 'False')
    config.set('Slippi', 'ReplayDir', replay_dir)
    config.set('DSP', 'Backend', 'No Audio Output')
    if headless:
      config.set('Core', 'GFXBackend', 'Null')
      config.set('Core', 'EmulationSpeed', '0')

    with open(dolphin_ini_path, 'w') as dolphin_ini_file:
      config.write(dolphin_ini_file)

    command = [
        dolphin_config.dolphin_path,
        '--exec', dolphin_config.ssbm_iso_path,
        '--user', user_dir,
        '-i', replay_json_path,
    ]
    if headless:
      command.extend(['--platform', 'headless'])

    try:
      subprocess.run(command, capture_output=True, check=True, timeout=time_limit)
    except subprocess.TimeoutExpired:
      raise DolphinTimeoutError(f'Timed out after {time_limit} seconds')
    except subprocess.CalledProcessError as e:
      raise DolphinExecutionError(e.stderr.decode())

    replays = os.listdir(replay_dir)
    if len(replays) != 1:
      raise ValueError(f'Expected exactly one replay in {replay_dir}, found {len(replays)}')

    replay_path = os.path.join(replay_dir, replays[0])

    # Upgrading loses some of the original metadata
    subprocess.check_call([copy_slp_metadata_binary, input_path, replay_path])

    os.rename(replay_path, output_path)

def test_upgrade_slp(
    input_path: str,
    dolphin_config: DolphinConfig,
    in_memory: bool = True,
    time_limit: Optional[int] = 30,
):
  """Test the upgrade_slp function."""
  if not os.path.exists(input_path):
    raise FileNotFoundError(f'Input path does not exist: {input_path}')

  with tempfile.TemporaryDirectory(dir=utils.get_tmp_dir(in_memory=in_memory)) as tmp_dir:
    output_path = os.path.join(tmp_dir, 'upgraded.slp')
    upgrade_slp(input_path, output_path, dolphin_config,
                in_memory=in_memory, time_limit=time_limit)

    game = game_array_to_nt(parse_peppi.get_slp(input_path))
    upgraded_game = game_array_to_nt(parse_peppi.get_slp(output_path))

    errors = check_same_structure(game, upgraded_game)
    if errors:
      for path, error in errors:
        print(f'Error at {path}: {error}')
      # import ipdb; ipdb.set_trace()
    else:
      print('Upgrade successful, no errors found.')

def is_online(start: dict) -> bool:
  """Check if a game is an online game based on the start dictionary."""
  return start['scene']['major'] == 8

def is_unfrozen_ps(start: dict) -> bool:
  """Check if a game on Pokemon Stadium is frozen based on the start dictionary."""
  # TODO: really we should be looking at the gecko codes like playback does

  stage = melee.enums.to_internal_stage(start['stage'])
  if stage is not melee.Stage.POKEMON_STADIUM:
    return False

  # NOTE: the is_frozen_ps flag was deprecated at some point

  # TODO: This will no longer be true in Slippi 3.19.0
  if is_online(start):
    return False

  return True

def needs_upgrade(input_path: str):
  """Check if a Slippi replay needs to be upgraded."""
  if not os.path.exists(input_path):
    raise FileNotFoundError(f'Input path does not exist: {input_path}')

  peppi_py_version = importlib.metadata.version('peppi-py')
  if peppi_py_version != '0.6.0':
    raise ImportError(f'peppi-py version {peppi_py_version} is not supported, please use version 0.6.0')

  game = peppi_py.read_slippi(input_path)  # skip_frames=True crashes :(
  start: dict = game.start
  stage = melee.enums.to_internal_stage(start['stage'])

  # Stage events were added in Slippi 3.18.0
  if start['slippi']['version'] < [3, 18, 0]:
    # platform heights
    if stage is melee.Stage.FOUNTAIN_OF_DREAMS:
      return True

    # transformations
    if is_unfrozen_ps(start):
      return True

  # TODO: Dreamland whispy blow direction?
  # NOTE: Randall is not a stage event

  return False

@dataclasses.dataclass(slots=True)
class UpgradeResult:
  """Result of upgrading a Slippi replay."""
  local_file: utils.LocalFile
  error: Optional[str] = None
  skipped: bool = False

def _upgrade_slp_in_archive(
    local_file: utils.LocalFile,
    output_path: str,
    dolphin_config: DolphinConfig,
    in_memory: bool = True,
    check_same_parse: bool = True,
    gzip_output: bool = True,
    dolphin_timeout: Optional[int] = None,
    check_if_needed: bool = False,
) -> UpgradeResult:
  skipped = False

  try:
    tmp_parent_dir = utils.get_tmp_dir(in_memory=in_memory)

    with tempfile.TemporaryDirectory(dir=tmp_parent_dir) as tmp_dir:
      slp_path = os.path.join(tmp_dir, 'game.slp')
      slp_data = local_file.read()
      with open(slp_path, 'wb') as f:
        f.write(slp_data)

      if check_if_needed and not needs_upgrade(slp_path):
        upgraded_data = slp_data
        del slp_data  # Free memory
        skipped = True
      else:
        del slp_data  # Free memory
        upgraded_path = os.path.join(tmp_dir, 'upgraded.slp')
        upgrade_slp(slp_path, upgraded_path, dolphin_config, in_memory=in_memory,
                    time_limit=dolphin_timeout)

        if check_same_parse:
          game = game_array_to_nt(parse_peppi.get_slp(slp_path))
          upgraded_game = game_array_to_nt(parse_peppi.get_slp(upgraded_path))

          errors = check_same_structure(game, upgraded_game)
          if errors:
            return UpgradeResult(local_file, 'different parse')

        with open(upgraded_path, 'rb') as f:
          upgraded_data = f.read()

      # TODO: don't re-gzip if already gzipped
      if gzip_output:
        upgraded_data = gzip.compress(upgraded_data)

      with open(output_path, 'wb') as f:
        f.write(upgraded_data)

  except Exception as e:
    return UpgradeResult(local_file, str(e), skipped=skipped)

  return UpgradeResult(local_file, None, skipped=skipped)

def _monitor_results(
    results_iter: Iterator[UpgradeResult],
    total_files: int,
    log_interval: int = 30,
) -> list[UpgradeResult]:
  pbar = tqdm.tqdm(total=total_files, desc="Upgrading", unit="file", smoothing=0)

  last_log_time = 0
  successful_conversions = 0
  last_error: Optional[tuple[str, str]] = None

  results: list[UpgradeResult] = []

  for result in results_iter:
    if result.error is None:
      successful_conversions += 1
    else:
      last_error = result.local_file.name, result.error

    results.append(result)

    pbar.update(1)

    if time.time() - last_log_time > log_interval:
      last_log_time = time.time()
      success_rate = successful_conversions / pbar.n
      logging.info(f'Success rate: {success_rate:.2%}')
      if last_error is not None:
        logging.error(f'Last error: {last_error}')
        last_error = None

  pbar.close()

  return results

def upgrade_archive(
    input_path: str,
    output_path: str,
    dolphin_config: DolphinConfig,
    in_memory: bool = True,
    work_dir: Optional[str] = None,
    num_threads: int = 1,
    check_same_parse: bool = True,
    gzip_output: bool = True,
    log_interval: int = 30,  # seconds between logs
    dolphin_timeout: Optional[int] = 60,
    check_if_needed: bool = False,
) -> list[UpgradeResult]:
  """Upgrade a Slippi replay archive to the latest version."""
  if not os.path.exists(input_path):
    raise FileNotFoundError(f'Input path does not exist: {input_path}')

  if not input_path.endswith('.zip'):
    raise ValueError(f'Input path must be a .zip file: {input_path}')

  if not output_path.endswith('.zip'):
    raise ValueError(f'Output path must be a .zip file: {output_path}')

  existing_outputs = set()
  if os.path.exists(output_path):
    with zipfile.ZipFile(output_path, 'r') as output_zip:
      for info in output_zip.infolist():
        if info.is_dir():
          continue
        existing_outputs.add(info.filename.removesuffix(utils.GZ_SUFFIX))
    logging.info(f'Found {len(existing_outputs)} existing files in output archive')

  zf = zipfile.ZipFile(input_path, 'r')
  total_size = 0
  skipped_files = 0
  files: list[utils.ZipFile] = []

  for zip_info in zf.infolist():
    if zip_info.is_dir():
      continue

    if not utils.is_slp_file(zip_info.filename):
      raise ValueError(f'Invalid file in archive: {zip_info.filename}')

    if zip_info.filename in existing_outputs:
      skipped_files += 1
      continue

    total_size += zip_info.compress_size if gzip_output else zip_info.file_size
    files.append(utils.ZipFile(input_path, zip_info.filename))

  zf.close()

  logging.info(f'Found {len(files)} .slp files in archive, total size: {total_size / (2 ** 30):.2f} GB')

  if skipped_files > 0:
    logging.info(f'Skipped {skipped_files} files that already exist in output archive')

  if in_memory:
    free_space = psutil.virtual_memory().available
  elif work_dir:
    free_space = psutil.disk_usage(work_dir).free
  else:
    raise ValueError('Either in_memory must be True or work_dir must be specified')

  # We use the compressed size of the files in the zip archive to estimate
  # the required space on disk. This is conservative because we will gzip each
  # output file which has better compression than plain zip.
  if total_size > free_space:
    raise MemoryError(f'Not enough free space to process archive: {total_size / (2 ** 30):.2f} GB required, {free_space / (2 ** 30):.2f} GB available')

  if in_memory:
    tmp_parent_dir = utils.get_tmp_dir(in_memory=True)
  else:
    tmp_parent_dir = work_dir

  with tempfile.TemporaryDirectory(dir=tmp_parent_dir) as output_dir:
    print(f'Using temporary output directory: {output_dir}')

    # Create output directories
    output_dirs = set()
    for f in files:
      output_dirs.add(os.path.join(output_dir, os.path.dirname(f.name)))
    for d in output_dirs:
      os.makedirs(d, exist_ok=True)

    results: list[UpgradeResult] = []

    if num_threads == 1:
      def results_iter1():
        for f in files:
          output_file_path = os.path.join(output_dir, f.name)
          if gzip_output:
            output_file_path += utils.GZ_SUFFIX
          yield _upgrade_slp_in_archive(
              local_file=f,
              output_path=output_file_path,
              dolphin_config=dolphin_config,
              in_memory=True,
              check_same_parse=check_same_parse,
              gzip_output=gzip_output,
              dolphin_timeout=dolphin_timeout,
              check_if_needed=check_if_needed,
          )

      results = _monitor_results(
          results_iter1(),
          total_files=len(files),
          log_interval=log_interval,
      )
    else:
      with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        try:
          futures: list[concurrent.futures.Future] = []
          for f in files:
            output_file_path = os.path.join(output_dir, f.name)
            if gzip_output:
              output_file_path += utils.GZ_SUFFIX
            futures.append(executor.submit(
                _upgrade_slp_in_archive,
                local_file=f,
                output_path=output_file_path,
                dolphin_config=dolphin_config,
                in_memory=True,
                check_same_parse=check_same_parse,
                gzip_output=gzip_output,
                dolphin_timeout=dolphin_timeout,
                check_if_needed=check_if_needed,
            ))

          def results_iter() -> Iterator[UpgradeResult]:
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

    # Count successful conversions
    successful_conversions = sum(1 for result in results if result.error is None)
    failed_conversions = len(results) - successful_conversions
    skipped_conversions = sum(1 for result in results if result.skipped)

    print(f"Conversion complete: {successful_conversions} successful, {failed_conversions} failed, {skipped_conversions} skipped")

    if successful_conversions == 0:
      print("No files were successfully converted")
      return results

    # Write results to output zip
    command = ['zip', '-r', '-q']
    if gzip_output:
      command.append('-0')  # Files are individually compressed with gzip
    command.extend([output_path, '.'])

    subprocess.check_call(command, cwd=output_dir)

  print(f"Output saved to: {output_path}")
  return results
