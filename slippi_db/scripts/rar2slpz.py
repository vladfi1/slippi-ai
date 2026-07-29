"""Convert rar archives of .slp files into zips of individually-compressed .slpz.

Replays are streamed out of `unrar p` and piped through `slpz`, so nothing is
extracted to the filesystem; the only output is the zip itself. Because each
member is already compressed by slpz, the zip is stored rather than deflated.

Requires `unrar` and `slpz` to be installed.
"""

import collections
import concurrent.futures
import os
import shutil
import subprocess
import tempfile
import typing as tp
import zipfile
import zlib
from pathlib import Path

import tqdm
from absl import app, flags, logging

INPUT = flags.DEFINE_string('input', None, 'Input .rar file or directory.', required=True)
OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'Output directory. If not specified, writes alongside the input.')
THREADS = flags.DEFINE_integer('threads', min(os.cpu_count() or 1, 8), 'Number of concurrent slpz processes.')
SMALL = flags.DEFINE_bool('small', False, 'Use `slpz --small` instead of `--fast`; ~5x slower for ~1% better ratio.')
MAX_PENDING = flags.DEFINE_integer('max_pending', 64, 'Max replays in flight, which bounds memory use.')
REMOVE_ORIGINAL = flags.DEFINE_bool('remove_original', False, 'Remove original rar files after successful conversion.')
CHECK_SPACE = flags.DEFINE_bool('check_space', True, 'Skip archives whose estimated output does not fit.')
LIMIT = flags.DEFINE_integer('limit', None, 'Limit number of files to process (for testing).')

# slpz compresses .slp to roughly a tenth of its size; pad for the estimate.
SLPZ_SIZE_FACTOR = 0.15

_SLP_SUFFIX = '.slp'
_SLPZ_SUFFIX = '.slpz'

def check_dependencies():
  for tool in ['unrar', 'slpz']:
    if shutil.which(tool) is None:
      raise RuntimeError(f'{tool} is not installed. Please install it.')

class Entry(tp.NamedTuple):
  path: str
  size: int
  crc32: str

def list_entries(archive: str) -> list[Entry]:
  """Lists the archive's files, in the order `unrar p` will emit them."""
  output = subprocess.run(
      ['unrar', 'lt', archive],
      check=True, capture_output=True, text=True).stdout

  entries: list[Entry] = []
  current: dict[str, str] = {}

  def flush():
    # Directories have no Size or CRC32 and produce no data in the stream.
    if current.get('Type') == 'File':
      entries.append(Entry(
          path=current['Name'],
          size=int(current['Size']),
          crc32=current.get('CRC32', '')))

  for line in output.splitlines():
    key, _, value = line.strip().partition(': ')
    if key == 'Name':
      flush()
      current = {'Name': value}
    elif key in ('Type', 'Size', 'CRC32'):
      current[key] = value
  flush()

  return entries

def format_size(num_bytes: int) -> str:
  return f'{num_bytes / 1024 ** 3:.2f} GiB'

def compress(data: bytes, small: bool) -> bytes:
  """Compresses one replay via slpz, reading and writing over pipes."""
  mode = '--small' if small else '--fast'
  proc = subprocess.run(
      ['slpz', mode, '-x', '-o', '-', '-'], input=data, capture_output=True)
  # slpz exits 0 even when it fails, so an empty result is the real signal.
  if proc.returncode != 0 or not proc.stdout:
    message = proc.stderr.decode(errors='replace').strip()
    raise RuntimeError(message or 'slpz produced no output')
  return proc.stdout

def output_name(path: str) -> str:
  if path.endswith(_SLP_SUFFIX):
    path = path[:-len(_SLP_SUFFIX)] + _SLPZ_SUFFIX
  # Zip entries always use forward slashes.
  return path.replace(os.sep, '/')

def convert(
    input_path: str,
    output_path: str,
    threads: int,
    small: bool,
    max_pending: int,
    limit: tp.Optional[int] = None,
) -> dict[str, str]:
  """Streams input_path into a stored zip of .slpz. Returns per-file errors."""
  input_path = os.path.abspath(input_path)
  output_path = os.path.abspath(output_path)

  entries = list_entries(input_path)
  if limit:
    entries = entries[:limit]
  total_bytes = sum(e.size for e in entries)
  estimated = int(total_bytes * SLPZ_SIZE_FACTOR)
  print(f'{len(entries)} files, {format_size(total_bytes)} uncompressed;'
        f' output ~{format_size(estimated)}')

  if CHECK_SPACE.value:
    free = shutil.disk_usage(os.path.dirname(output_path)).free
    if free < estimated:
      raise RuntimeError(
          f'Not enough space in {os.path.dirname(output_path)}: need '
          f'~{format_size(estimated)}, have {format_size(free)}.')

  errors: dict[str, str] = {}
  # Write beside the target so a failed run doesn't leave a usable-looking zip.
  partial_path = output_path + '.part'

  # unrar is quiet under -inul, but a filled stderr pipe would deadlock it,
  # so give it somewhere unbounded to write.
  with tempfile.TemporaryFile() as stderr_file:
    proc = subprocess.Popen(
        ['unrar', 'p', '-inul', '-p-', input_path],
        stdout=subprocess.PIPE, stderr=stderr_file, bufsize=1024 * 1024)

    try:
      with zipfile.ZipFile(
          partial_path, 'w', zipfile.ZIP_STORED, allowZip64=True) as zf, \
          concurrent.futures.ThreadPoolExecutor(threads) as pool, \
          tqdm.tqdm(total=total_bytes, desc='convert', unit='B',
                    unit_scale=True, unit_divisor=1024) as bar:

        pending = collections.deque()

        def drain(limit: int):
          """Writes finished replays until at most `limit` remain in flight."""
          while len(pending) > limit:
            entry, future = pending.popleft()
            try:
              zf.writestr(output_name(entry.path), future.result())
            except Exception as e:
              errors[entry.path] = str(e)
            bar.update(entry.size)

        for entry in entries:
          data = proc.stdout.read(entry.size)
          if len(data) != entry.size:
            raise RuntimeError(
                f'{entry.path}: stream ended early, got {len(data)} of '
                f'{entry.size} bytes')
          # The listing's CRC32 confirms we're still aligned with the stream.
          if entry.crc32 and format(zlib.crc32(data), '08X') != entry.crc32:
            raise RuntimeError(f'{entry.path}: CRC32 mismatch, stream desynced')

          if entry.path.endswith(_SLP_SUFFIX):
            pending.append((entry, pool.submit(compress, data, small)))
          else:
            # Not a replay; keep it verbatim rather than dropping it.
            zf.writestr(output_name(entry.path), data)
            bar.update(entry.size)

          drain(max_pending)

        drain(0)

        if limit:
          # We're deliberately not draining the rest of the stream.
          proc.kill()
        elif proc.stdout.read(1):
          raise RuntimeError('unrar emitted more data than the listing claimed')
    except BaseException:
      proc.kill()
      if os.path.exists(partial_path):
        os.remove(partial_path)
      raise
    finally:
      proc.stdout.close()

    if proc.wait() != 0 and not limit:
      stderr_file.seek(0)
      message = stderr_file.read().decode(errors='replace').strip()
      os.remove(partial_path)
      raise subprocess.CalledProcessError(
          proc.returncode, 'unrar p', stderr=message)

  os.replace(partial_path, output_path)
  return errors

def process_single_rar(input_path: Path, output_dir: Path) -> bool:
  output_path = output_dir / (input_path.stem + '.zip')
  print(f'Converting {input_path} to {output_path}')
  try:
    errors = convert(
        str(input_path), str(output_path),
        threads=THREADS.value, small=SMALL.value,
        max_pending=MAX_PENDING.value, limit=LIMIT.value)
  except Exception as e:
    logging.error(f'Failed to convert {input_path}: {e}')
    return False

  if errors:
    logging.error(f'{len(errors)} files failed to compress in {input_path}')
    path, message = next(iter(errors.items()))
    logging.error(f'Example failure in {path}: {message}')

  if REMOVE_ORIGINAL.value:
    if errors:
      logging.warning(f'Keeping {input_path} because some files failed')
    else:
      input_path.unlink()
      logging.info(f'Removed original file: {input_path}')

  return not errors

def main(_):
  check_dependencies()
  input_path = Path(INPUT.value)

  if input_path.is_file():
    if input_path.suffix != '.rar':
      raise ValueError(f'Input file must be a .rar file: {input_path}')

    if OUTPUT_DIR.value:
      output_dir = Path(OUTPUT_DIR.value)
      output_dir.mkdir(parents=True, exist_ok=True)
    else:
      output_dir = input_path.parent

    process_single_rar(input_path, output_dir)
  elif input_path.is_dir():
    output_dir = None
    if OUTPUT_DIR.value:
      output_dir = Path(OUTPUT_DIR.value)
      if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f'Output path must be a directory: {output_dir}')

    rar_files = list(input_path.rglob('*.rar'))
    logging.info(f'Found {len(rar_files)} rar files to process')

    successful_conversions = 0
    for rar_file in tqdm.tqdm(rar_files, desc='Converting files'):
      if output_dir:
        rel_path = rar_file.relative_to(input_path)
        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
      else:
        output_subdir = rar_file.parent

      if process_single_rar(rar_file, output_subdir):
        successful_conversions += 1

    logging.info(f'Successfully converted {successful_conversions}/{len(rar_files)} files')
  else:
    raise ValueError(f'Input path does not exist: {input_path}')

if __name__ == '__main__':
  app.run(main)
