import concurrent.futures
import configparser
import dataclasses
import enum
import json
import os
import psutil
import subprocess
import tempfile
import time
from typing import Optional, Iterator, Any
import zipfile

from absl import logging
import numpy as np
import pyarrow as pa
import tqdm
import tree

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

class MetadataUpdateError(RuntimeError):
  pass

# https://github.com/project-slippi/slippi-wiki/blob/master/COMM_SPEC.md#top-level
class RollbackDisplayMethod(enum.Enum):
  OFF = 'off'
  NORMAL = 'normal'
  VISIBLE = 'visible'

def upgrade_slp(
    input_path: str,
    output_path: str,
    dolphin_config: DolphinConfig,
    in_memory: bool = True,
    time_limit: Optional[int] = None,
    copy_slp_metadata_binary: str = 'copy_slp_metadata',
    headless: bool = True,
    fast_forward: bool = True,
    rollback_display_method: RollbackDisplayMethod = RollbackDisplayMethod.OFF,
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
        'rollbackDisplayMethod': rollback_display_method.value,
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
    try:
      subprocess.run(
          [copy_slp_metadata_binary, input_path, replay_path],
          check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
      raise MetadataUpdateError(e.stderr)

    os.rename(replay_path, output_path)

def test_upgrade_slp(
    input_path: str,
    dolphin_config: DolphinConfig,
    in_memory: bool = True,
    **upgrade_kwargs,
):
  """Test the upgrade_slp function."""
  if not os.path.exists(input_path):
    raise FileNotFoundError(f'Input path does not exist: {input_path}')

  with tempfile.TemporaryDirectory(dir=utils.get_tmp_dir(in_memory=in_memory)) as tmp_dir:
    output_path = os.path.join(tmp_dir, 'upgraded.slp')
    upgrade_slp(input_path, output_path, dolphin_config,
                in_memory=in_memory, **upgrade_kwargs)

    game = game_array_to_nt(parse_peppi.get_slp(input_path))
    upgraded_game = game_array_to_nt(parse_peppi.get_slp(output_path))

    errors = check_same_structure(game, upgraded_game)
    if errors:
      for path, error in errors:
        print(f'Error at {path}: {error}')
      # import ipdb; ipdb.set_trace()
    else:
      print('Upgrade successful, no errors found.')

def is_online(game: peppi_py.game.Game) -> bool:
  """Check if a game is an online game based on the start dictionary."""
  # NOTE: This probably won't work properly for upgraded replays,
  # but that's ok because upgraded PS games will already have stage events.

  if game.metadata:
    players = game.metadata['players'].values()
    player = next(iter(players))
    return 'netplay' in player['names']

  if game.start.scene is not None:
    return game.start.scene.major == 8

  # Guess that it was not online so that we do upgrade it.
  return False

def is_unfrozen_ps(game: peppi_py.game.Game) -> bool:
  """Check if a game on Pokemon Stadium is frozen based on the start dictionary."""
  # TODO: really we should be looking at the gecko codes like playback does

  stage = melee.enums.to_internal_stage(game.start.stage)
  if stage is not melee.Stage.POKEMON_STADIUM:
    return False

  # NOTE: is_frozen_ps was deprecated at some point, and re-enabled in 3.19.0
  if game.start.slippi.version >= (3, 19, 0):
    return not game.start.is_frozen_ps
  elif is_online(game):
    return False

  return True

DEFAULT_MIN_VERSION = (3, 2, 0)

def needs_upgrade(
    # input_path: str,
    game: peppi_py.Game,
    min_version: tuple[int, int, int] = DEFAULT_MIN_VERSION,
):
  """Check if a Slippi replay needs to be upgraded."""
  start = game.start
  stage = melee.enums.to_internal_stage(start.stage)

  if start.slippi.version < min_version:
    return True

  # Stage events were added in Slippi 3.18.0
  if start.slippi.version < (3, 18, 0):
    # platform heights
    if stage is melee.Stage.FOUNTAIN_OF_DREAMS:
      return True

    # transformations
    if is_unfrozen_ps(game):
      return True

  # TODO: Dreamland whispy blow direction?
  # NOTE: Randall is not a stage event

  return False

@dataclasses.dataclass(slots=True)
class UpgradeResult:
  """Result of upgrading a Slippi replay."""
  local_file: utils.ZipFile
  error: Optional[str] = None
  skipped: bool = False


def check_same_metadata(
    old_game: peppi_py.Game,
    new_game: peppi_py.Game,
) -> Optional[str]:
  """Check if two games have the same metadata."""
  if old_game.start.match is not None:  # old game might have no metadata
    if old_game.start.match != new_game.start.match:
      return 'different match'

  for p1, p2 in zip(old_game.start.players, new_game.start.players):
    if p1.name_tag != p2.name_tag:
      return 'different name_tag'

    if p1.netplay is not None:
      for key in ['name', 'code']:
        if getattr(p1.netplay, key) != getattr(p2.netplay, key):
          return f'different netplay.{key}'

  if old_game.end is not None:
    if new_game.end is None:
      return 'missing end'

    # Note: lras_initiator is sometimes different
    for key in ['method', 'players']:
      old = getattr(old_game.end, key)
      new = getattr(new_game.end, key)
      if old is not None and new != old:
        return f'different end.{key}'

  return None

def _known_game_read_exception(e: BaseException) -> str | None:
  message = str(e)

  game_end_prefix = 'invalid data: invalid game end method: '
  if isinstance(e, OSError) and message.startswith(game_end_prefix):
    return 'invalid game end method'

  return None


def check_leaf(path: tuple, original_array, upgraded_array) -> Optional[str]:
  if original_array is None:
    return None

  if upgraded_array is None:
    return 'field missing in upgraded'

  if path[-1] == 'buttons_physical':
    return None

  if path[-2:-1] == ('triggers_physical',):
    return None

  if path[:2] == ('items', 'misc'):
    return None

  if path[-3:] == ('post', 'state_flags', 4):
    return None

  if isinstance(original_array, pa.ListArray):
    assert isinstance(upgraded_array, pa.ListArray)

    if not np.array_equal(
        original_array.offsets.to_numpy(),
        upgraded_array.offsets.to_numpy()):
      return 'offsets different'

    original_array = original_array.values
    upgraded_array = upgraded_array.values

  original_np = original_array.to_numpy(zero_copy_only=False)
  upgraded_np = upgraded_array.to_numpy(zero_copy_only=False)

  if original_np.shape != upgraded_np.shape:
      return f'shape mismatch {original_np.shape} != {upgraded_np.shape}'

  if np.issubdtype(original_np.dtype, np.floating):
    is_different = ~np.isclose(
        upgraded_np, original_np,
        rtol=1e-5, atol=1e-5, equal_nan=True)
  else:
    is_different = original_np != upgraded_np

  if np.any(is_different):
    game_len = len(is_different)
    diff_indices = np.arange(game_len)[is_different]
    return f'difference at {len(diff_indices)}/{game_len} frames, example index {diff_indices[0]}'

  return None

def check_games(
    original: peppi_py.Game,
    upgraded: peppi_py.Game,
    debug: bool = False,
) -> list[tuple[Any, str]]:

  # fod_platforms_added = (
  #     original.start.slippi.version < (3, 18, 0) and
  #     upgraded.start.slippi.version >= (3, 18, 0) and
  #     melee.enums.to_internal_stage(original.start.stage) is melee.Stage.FOUNTAIN_OF_DREAMS)

  error = check_same_metadata(original, upgraded)
  if error is not None:
    if debug:
      print(error)
      import ipdb; ipdb.set_trace()

    return [('metadata', error)]

  original_frames = dataclasses.asdict(original.frames)
  upgraded_frames = dataclasses.asdict(upgraded.frames)
  errors = tree.flatten_with_path(
      tree.map_structure_with_path(
          check_leaf, original_frames, upgraded_frames))

  errors = [(path, msg) for path, msg in errors if msg is not None]

  if errors:
    if debug:
      for path, message in errors:
        print(path, message)
      import ipdb; ipdb.set_trace()

  return errors

def errors_to_str(errors: list[tuple[Any, str]]) -> str:
  path, msg = errors[0]
  return f'{len(errors)} errors, first at {path}: {msg}'

def _upgrade_slp_in_archive(
    local_file: utils.ZipFile,
    output_path: str,
    dolphin_config: DolphinConfig,
    in_memory: bool = True,
    check_same_parse: bool = True,
    dolphin_timeout: Optional[int] = None,
    check_if_needed: bool = False,
    min_version: tuple[int, int, int] = DEFAULT_MIN_VERSION,
    expected_version: tuple[int, int, int] = (3, 18, 0),
) -> UpgradeResult:
  skipped = False

  try:
    tmp_parent_dir = utils.get_tmp_dir(in_memory=in_memory)

    with tempfile.TemporaryDirectory(dir=tmp_parent_dir) as tmp_dir:
      raw_data = local_file.read_raw()

      slp_path = os.path.join(tmp_dir, 'game.slp')
      with open(slp_path, 'wb') as f:
        f.write(local_file.from_raw(raw_data))

      game = peppi_py.read_slippi(
          slp_path, rollback_mode=peppi_py.RollbackMode.LAST)

      if check_if_needed and not needs_upgrade(game, min_version):
        upgraded_path = slp_path
        skipped = True
      else:
        upgraded_path = os.path.join(tmp_dir, 'upgraded.slp')
        upgrade_slp(slp_path, upgraded_path, dolphin_config, in_memory=in_memory,
                    time_limit=dolphin_timeout)

        if check_same_parse:
          upgraded_game = peppi_py.read_slippi(
              upgraded_path, rollback_mode=peppi_py.RollbackMode.LAST)

          if upgraded_game.start.slippi.version != expected_version:
            return UpgradeResult(local_file, f'unexpected version {upgraded_game.start.slippi.version}')

          errors = check_games(game, upgraded_game)
          if errors:
            return UpgradeResult(local_file, errors_to_str(errors))

      if skipped and local_file.is_slpz:
        with open(output_path, 'wb') as f:
          f.write(raw_data)
      else:
        try:
          subprocess.run(
              ['slpz', '-x', upgraded_path, '-o', output_path],
              check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
          return UpgradeResult(local_file, 'slpz: ' + e.stderr)

  except KeyboardInterrupt:
    raise

  except BaseException as e:
    known_error = _known_game_read_exception(e)
    if known_error:
      return UpgradeResult(local_file, known_error, skipped=True)

    return UpgradeResult(local_file, type(e).__name__ + ": " + str(e), skipped=skipped)

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

_ARCHIVE_SUFFIXES = ['.zip', '.7z', '.rar']

def _is_archive(path: str) -> bool:
  return any(path.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)

def _safe_path(path: str) -> str:
  components = os.path.normpath(path).split('/')
  components = [c for c in components if c and c != '..']
  return os.path.join(*components)

def upgrade_archive(
    input_path: str,
    output_path: str,
    dolphin_config: DolphinConfig,
    in_memory: bool = True,
    work_dir: Optional[str] = None,
    num_threads: int = 1,
    check_same_parse: bool = True,
    log_interval: int = 30,  # seconds between logs
    dolphin_timeout: Optional[int] = 60,
    check_if_needed: bool = False,
    expected_version: tuple[int, int, int] = (3, 18, 0),
    remove_input: bool = False,
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
        if not info.filename.endswith(utils.SLPZ_SUFFIX):
          raise ValueError(f'Output archive {output_path} contains non-slpz file: {info.filename}')
        existing_outputs.add(info.filename)
    logging.info(f'Found {len(existing_outputs)} existing files in output archive')

  zf = zipfile.ZipFile(input_path, 'r')
  total_size = 0
  skipped_files = 0
  todo: list[tuple[utils.ZipFile, str]] = []
  to_remove: list[str] = []

  for zip_info in zf.infolist():
    if zip_info.is_dir():
      continue

    if _is_archive(zip_info.filename):
      raise ValueError(f'Input archive contains nested archive: {zip_info.filename}')

    if not utils.is_slp_file(zip_info.filename):
      continue

    local_file = utils.ZipFile(input_path, zip_info.filename)
    output_filename = _safe_path(local_file.base_name) + utils.SLPZ_SUFFIX

    if output_filename in existing_outputs:
      skipped_files += 1
      if remove_input:
        to_remove.append(zip_info.filename)
      continue

    # True compression rate may be up to ~2x better thanks to slpz, if the
    # input is just a [g]zipped .slp file.
    if local_file.is_slpz:
      total_size += zip_info.file_size
    else:
      total_size += zip_info.compress_size * 0.6

    todo.append((local_file, output_filename))

  zf.close()

  if skipped_files > 0:
    logging.info(f'Skipped {skipped_files} files that already exist in output archive')

  # if not todo:
  #   logging.info('No files to process')
  #   return []

  logging.info(f'Found {len(todo)} .slp files in archive, estimated slpz size: {total_size / (2 ** 30):.2f} GB')

  if in_memory:
    work_dir_space = psutil.virtual_memory().available
  elif work_dir:
    work_dir_space = psutil.disk_usage(work_dir).free
  else:
    raise ValueError('Either in_memory must be True or work_dir must be specified')

  if total_size > work_dir_space:
    raise MemoryError(f'Not enough free space to process archive: {total_size / (2 ** 30):.2f} GB required, {work_dir_space / (2 ** 30):.2f} GB available')

  output_parent = os.path.dirname(output_path)
  os.makedirs(output_parent, exist_ok=True)
  output_space = psutil.disk_usage(output_parent).free
  if total_size > output_space:
    raise MemoryError(f'Not enough free space to write output archive: {total_size / (2 ** 30):.2f} GB required, {output_space / (2 ** 30):.2f} GB available')

  if in_memory:
    tmp_parent_dir = utils.get_tmp_dir(in_memory=True)
  else:
    tmp_parent_dir = work_dir

  # TODO: optionally keep around work dir?
  with tempfile.TemporaryDirectory(dir=tmp_parent_dir) as output_dir:
    print(f'Using temporary output directory: {output_dir}')

    # Create output directories
    output_dirs = set()
    for _, output_file_path in todo:
      output_dirs.add(os.path.join(output_dir, os.path.dirname(output_file_path)))
    for d in output_dirs:
      os.makedirs(d, exist_ok=True)

    results: list[UpgradeResult] = []

    if num_threads == 1:
      def results_iter1():
        for f, output_file_path in todo:
          yield _upgrade_slp_in_archive(
              local_file=f,
              output_path=os.path.join(output_dir, output_file_path),
              dolphin_config=dolphin_config,
              in_memory=True,
              check_same_parse=check_same_parse,
              dolphin_timeout=dolphin_timeout,
              check_if_needed=check_if_needed,
              expected_version=expected_version,
          )

      results = _monitor_results(
          results_iter1(),
          total_files=len(todo),
          log_interval=log_interval,
      )
    else:
      with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        try:
          futures: list[concurrent.futures.Future] = []
          for f, output_file_path in todo:
            futures.append(executor.submit(
                _upgrade_slp_in_archive,
                local_file=f,
                output_path=os.path.join(output_dir, output_file_path),
                dolphin_config=dolphin_config,
                in_memory=True,
                check_same_parse=check_same_parse,
                dolphin_timeout=dolphin_timeout,
                check_if_needed=check_if_needed,
                expected_version=expected_version,
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
    non_skipped_results = [result for result in results if not result.skipped]
    skipped_conversions = len(results) - len(non_skipped_results)

    successful_conversions = sum(1 for result in results if result.error is None)
    failed_conversions = len(results) - successful_conversions

    print(f"Conversion complete: {successful_conversions} successful, {failed_conversions} failed, {skipped_conversions} skipped")

    if successful_conversions > 0:
      # Write results to output zip. Files are individually compressed with slpz,
      # so we don't use compression at the zip level (-0).
      os.makedirs(os.path.dirname(output_path), exist_ok=True)
      command = ['zip', '-r', '-q', '-0', os.path.abspath(output_path), '.']
      subprocess.check_call(command, cwd=output_dir)

    if remove_input:
      if failed_conversions == 0:
        print(f"Removing input archive: {input_path}")
        os.remove(input_path)
      else:
        # Note: some errors are "known" and marked as skipped; we remove those too
        to_remove.extend([
            result.local_file.path for result in results
            if result.error is None or result.skipped
        ])
        print(f"Removing {len(to_remove)} files from input archive at {input_path}")
        if to_remove:
          utils.delete_from_zip(input_path, to_remove)

  print(f"Output saved to: {output_path}")
  return results
