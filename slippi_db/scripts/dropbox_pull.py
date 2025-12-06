#!/usr/bin/env python3
"""
Dropbox synchronization script using dbxcli.
Downloads files from a Dropbox folder to a local directory without overwriting existing files by default.
"""

import subprocess
import os
import sys
import argparse
import shutil
import time
from datetime import timedelta
from typing import List, Dict, Any, Optional, Union


def parse_size(size_str: str) -> int:
  """Parse size string (e.g., '1.5 GiB', '500 MiB') to bytes."""
  if not size_str or size_str == "-":
    return 0

  # Remove any extra whitespace
  size_str = size_str.strip()

  # Split number and unit
  parts = size_str.split()
  if len(parts) != 2:
    return 0

  try:
    number = float(parts[0])
    unit = parts[1].upper()

    # Convert to bytes
    units = {
      'B': 1,
      'KB': 1024,
      'KIB': 1024,
      'MB': 1024 * 1024,
      'MIB': 1024 * 1024,
      'GB': 1024 * 1024 * 1024,
      'GIB': 1024 * 1024 * 1024,
      'TB': 1024 * 1024 * 1024 * 1024,
      'TIB': 1024 * 1024 * 1024 * 1024
    }

    if unit in units:
      return int(number * units[unit])
    else:
      return 0
  except (ValueError, IndexError):
    return 0


def format_size(size_bytes: Union[int, float]) -> str:
  """Format bytes into human-readable size."""
  for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
    if size_bytes < 1024.0:
      return f"{size_bytes:.1f} {unit}"
    size_bytes /= 1024.0
  return f"{size_bytes:.1f} PiB"


def get_free_space(path: str) -> int:
  """Get free space available at the given path in bytes."""
  # Get the mount point for the path
  while not os.path.exists(path):
    path = os.path.dirname(path)

  stat = shutil.disk_usage(path)
  return stat.free


class ProgressTracker:
  """Tracks download progress and estimates time remaining."""

  def __init__(self, total_bytes: int, total_files: int) -> None:
    self.total_bytes = total_bytes
    self.total_files = total_files
    self.transferred_bytes = 0
    self.completed_files = 0
    self.start_time = time.time()
    self.current_file = ""
    self.current_file_size = 0
    self.current_file_downloaded = 0
    self.last_progress_time = time.time()
    self.last_progress_size = 0

  def start_file(self, filename: str, file_size: int) -> None:
    """Start tracking a new file download."""
    self.current_file = filename
    self.current_file_size = file_size
    self.current_file_downloaded = 0

  def update_file_progress(self, downloaded_bytes: int) -> None:
    """Update progress for the current file."""
    if downloaded_bytes > self.current_file_downloaded:
      self.last_progress_time = time.time()
      self.last_progress_size = downloaded_bytes
    self.current_file_downloaded = downloaded_bytes

  def is_stalled(self, timeout: int = 60) -> bool:
    """Check if download has stalled (no progress for timeout seconds)."""
    if self.current_file_size == 0:
      return False
    return (time.time() - self.last_progress_time) > timeout

  def complete_file(self) -> None:
    """Mark current file as completed."""
    self.transferred_bytes += self.current_file_size
    self.completed_files += 1
    self.current_file = ""
    self.current_file_size = 0
    self.current_file_downloaded = 0

  def get_progress(self) -> Dict[str, Any]:
    """Get current progress statistics."""
    elapsed = time.time() - self.start_time
    current_total_transferred = self.transferred_bytes + self.current_file_downloaded

    if current_total_transferred > 0 and elapsed > 0:
      transfer_rate = current_total_transferred / elapsed
      remaining_bytes = self.total_bytes - current_total_transferred
      eta_seconds = remaining_bytes / transfer_rate if transfer_rate > 0 else 0
      eta = timedelta(seconds=int(eta_seconds))
    else:
      transfer_rate = 0
      eta = timedelta(0)

    progress_percent = (current_total_transferred / self.total_bytes * 100) if self.total_bytes > 0 else 0

    return {
      'percent': progress_percent,
      'transferred': current_total_transferred,
      'total': self.total_bytes,
      'rate': transfer_rate,
      'eta': eta,
      'elapsed': timedelta(seconds=int(elapsed)),
      'files_completed': self.completed_files,
      'total_files': self.total_files,
      'current_file': os.path.basename(self.current_file) if self.current_file else ""
    }

  def print_progress(self) -> None:
    """Print current progress to console."""
    stats = self.get_progress()

    # Create progress bar
    # bar_width = 40
    # filled = int(bar_width * stats['percent'] / 100)
    # bar = '█' * filled + '░' * (bar_width - filled)

    # Format transfer rate
    rate_str = format_size(stats['rate']) + "/s" if stats['rate'] > 0 else "0 B/s"

    # Truncate current file name to fit in terminal
    current_file = stats['current_file'][:20] + "..." if len(stats['current_file']) > 20 else stats['current_file']

    # Clear the line first, then print progress
    progress_line = (f"{stats['percent']:.1f}% "
                    f"({format_size(stats['transferred'])}/{format_size(stats['total'])}) "
                    f"Files: {stats['files_completed']}/{stats['total_files']} "
                    f"Rate: {rate_str} "
                    f"ETA: {stats['eta']} "
                    f"{current_file}")

    # Clear the line and print progress (works better in screen sessions)
    print(f"\r\033[K{progress_line}", end='', flush=True)


def run_command(cmd: str) -> str:
  """Run a command and return the output or raise an exception on error."""
  try:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
    return result.stdout.strip()
  except subprocess.CalledProcessError as e:
    print(f"Error running command: {cmd}")
    print(f"Error output: {e.stderr}")
    raise


def get_dropbox_files(dropbox_path: str) -> List[Dict[str, Any]]:
  """Get list of files in Dropbox path recursively using dbxcli ls -l -R."""
  files = []

  # Use dbxcli ls -l -R to list all files recursively with details
  try:
    output = run_command(f"dbxcli ls -l -R '{dropbox_path}'")
  except subprocess.CalledProcessError:
    print(f"Error listing files in {dropbox_path}")
    return files

  if not output:
    print(f"No files found in {dropbox_path}")
    return files

  # Parse the output of dbxcli ls -l -R
  # Format: Revision                        Size     Last modified  Path
  lines = output.strip().split('\n')
  if len(lines) <= 1:  # Need at least header + 1 data line
    return files

  # Use header to determine column positions
  header = lines[0]
  if "Path" not in header:
    print("Unexpected header format")
    return files

  # Find column positions based on header
  path_col = header.index("Path")
  size_col = header.index("Size")
  modified_col = header.index("Last modified")

  # Process data lines
  for line in lines[1:]:
    if not line.strip() or len(line) < path_col + 1:
      continue

    # Extract path starting from the path column
    path = line[path_col:].strip()

    # Extract size between Size column and Last modified column
    size_part = line[size_col:modified_col].strip()

    # Skip directories (they have "-" for size)
    if size_part == "-":
      continue

    # Only include archive files (.zip, .7z, .rar)
    if path and path.endswith(('.zip', '.7z', '.rar')):
      files.append({
        'path': path,
        'name': os.path.basename(path),
        'size': size_part,
        'size_bytes': parse_size(size_part)
      })

  return files


def calculate_local_path(dropbox_path: str, dropbox_folder: str, local_folder: str) -> str:
  """Calculate the local path for a given dropbox file path."""
  assert dropbox_path.startswith(dropbox_folder), f"Path {dropbox_path} doesn't start with {dropbox_folder}"
  relative_path = dropbox_path[len(dropbox_folder):].lstrip('/')
  return os.path.join(local_folder, relative_path)


def download_file(
    dropbox_path: str,
    local_path: str,
    file_size: int,
    progress_tracker: Optional[ProgressTracker] = None,
    overwrite: bool = False,
    stall_timeout: int = 60,
    max_retries: int = 3
) -> bool:
  """Download a single file from Dropbox with progress tracking and stall detection."""
  # Create local directory if it doesn't exist
  local_dir = os.path.dirname(local_path)
  os.makedirs(local_dir, exist_ok=True)

  # Check if file already exists
  if os.path.exists(local_path) and not overwrite:
    if progress_tracker:
      progress_tracker.complete_file()  # Skip but count as completed
    return False

  success = False
  attempts = 0
  while attempts < max_retries:
    try:
      if attempts > 0:
        print(f"\nRetrying download (attempt {attempts + 1}/{max_retries})...")
        # If retrying, remove partial file
        if os.path.exists(local_path):
          os.remove(local_path)

      if progress_tracker:
        progress_tracker.start_file(local_path, file_size)
        progress_tracker.print_progress()

      # Start the download process
      process = subprocess.Popen(
          ["dbxcli", "get", dropbox_path, local_path],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.PIPE,
          text=True,
      )

      # Monitor file size while downloading
      stalled = False
      while process.poll() is None:
        current_size = 0
        if os.path.exists(local_path):
          current_size = os.path.getsize(local_path)

        if progress_tracker:
          progress_tracker.update_file_progress(min(current_size, file_size))
          progress_tracker.print_progress()

          # Check for stall
          if progress_tracker.is_stalled(stall_timeout):
            print(f"\nDownload stalled for {stall_timeout} seconds, terminating...")
            process.terminate()
            process.wait(timeout=5)
            if process.poll() is None:
              process.kill()
              process.wait()
            stalled = True
            break

        time.sleep(0.5)  # Check every 500ms

      # If process was terminated due to stall, retry
      if stalled:
        attempts += 1
        continue

      # Wait for process to complete (if not already done)
      _, stderr = process.communicate()

      if process.returncode != 0:
        if progress_tracker:
          print()  # New line after progress bar
        raise subprocess.CalledProcessError(process.returncode, f"dbxcli get '{dropbox_path}' '{local_path}'", stderr)

      success = True

      if progress_tracker:
        progress_tracker.complete_file()
        progress_tracker.print_progress()

      return True

    except subprocess.CalledProcessError as e:
      if progress_tracker:
        print()  # New line after progress bar
      print(f"Failed to download {dropbox_path}: {e.stderr}")
      attempts += 1
      if attempts < max_retries:
        print(f"Will retry...")
        continue
      return False

    finally:
      if not success and os.path.exists(local_path):
        # Remove partial file on failure
        os.remove(local_path)

  # If we exhausted all retries
  if progress_tracker:
    print()  # New line after progress bar
  print(f"Failed to download {dropbox_path} after {max_retries} attempts")
  return False


def check_disk_space(files: List[Dict[str, Any]], dropbox_folder: str, local_folder: str, overwrite: bool = False) -> Dict[str, Any]:
  """Check if there's enough disk space for the files to be downloaded."""
  # Calculate total size needed
  total_size_needed = 0
  files_to_download = []

  for file_info in files:
    dropbox_path = file_info['path']
    local_path = calculate_local_path(dropbox_path, dropbox_folder, local_folder)

    # Only count files that need to be downloaded
    if not os.path.exists(local_path) or overwrite:
      total_size_needed += file_info['size_bytes']
      files_to_download.append(file_info)

  # Get available disk space
  free_space = get_free_space(local_folder)

  # Add 10% safety margin
  required_space = int(total_size_needed * 1.1)

  return {
    'total_size': total_size_needed,
    'required_space': required_space,
    'free_space': free_space,
    'has_enough_space': free_space >= required_space,
    'files_to_download': len(files_to_download),
    'total_files': len(files)
  }


def sync_folder(dropbox_folder: str, local_folder: str, overwrite: bool = False, stall_timeout: int = 60) -> None:
  """Synchronize a Dropbox folder to a local folder."""
  print(f"Synchronizing {dropbox_folder} to {local_folder}")

  # Ensure local folder exists
  os.makedirs(local_folder, exist_ok=True)

  # Get list of files in Dropbox
  print("Discovering files on Dropbox...")
  files = get_dropbox_files(dropbox_folder)

  if not files:
    print("No files found to synchronize.")
    return

  print(f"Found {len(files)} files to process")

  # Check disk space
  print("\nChecking disk space...")
  space_info = check_disk_space(files, dropbox_folder, local_folder, overwrite)

  print(f"Files to download: {space_info['files_to_download']} of {space_info['total_files']}")
  print(f"Total size needed: {format_size(space_info['total_size'])}")
  print(f"Required space (with 10% margin): {format_size(space_info['required_space'])}")
  print(f"Available disk space: {format_size(space_info['free_space'])}")

  if not space_info['has_enough_space']:
    print(f"\nERROR: Not enough disk space!")
    print(f"Need {format_size(space_info['required_space'] - space_info['free_space'])} more space")
    return

  print("\nSufficient disk space available. Proceeding with sync...")

  # Initialize progress tracker for files that will be downloaded
  progress_tracker = ProgressTracker(space_info['total_size'], space_info['files_to_download'])

  downloaded = 0
  skipped = 0
  failed = 0

  print()  # Empty line before progress bar
  for file_info in files:
    dropbox_path = file_info['path']
    local_path = calculate_local_path(dropbox_path, dropbox_folder, local_folder)

    # Check if file needs to be downloaded
    needs_download = not os.path.exists(local_path) or overwrite

    if needs_download:
      if download_file(dropbox_path, local_path, file_info['size_bytes'], progress_tracker, overwrite, stall_timeout):
        downloaded += 1
      else:
        failed += 1
    else:
      # File already exists and not overwriting - just count as skipped
      skipped += 1

  print()  # New line after final progress update
  print(f"\nSync completed:")
  print(f"  Downloaded: {downloaded}")
  print(f"  Skipped: {skipped}")
  print(f"  Failed: {failed}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Synchronize Dropbox folder to local directory using dbxcli")
  parser.add_argument("dropbox_path", help="Dropbox folder path to sync from")
  parser.add_argument("local_path", help="Local directory path to sync to")
  parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
  parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without actually downloading")
  parser.add_argument("--stall-timeout", type=int, default=60, help="Timeout in seconds to detect stalled downloads (default: 60)")

  args = parser.parse_args()

  # Check if dbxcli is available
  try:
    run_command("dbxcli version")
  except (subprocess.CalledProcessError, FileNotFoundError):
    print("Error: dbxcli not found. Please install and configure dbxcli first.")
    sys.exit(1)

  if args.dry_run:
    print("DRY RUN MODE - No files will be downloaded")
    files = get_dropbox_files(args.dropbox_path)

    if not files:
      print("No files found.")
      return

    # Check disk space
    space_info = check_disk_space(files, args.dropbox_path, args.local_path, args.overwrite)

    print(f"\nDisk space analysis:")
    print(f"  Files to download: {space_info['files_to_download']} of {space_info['total_files']}")
    print(f"  Total size needed: {format_size(space_info['total_size'])}")
    print(f"  Required space (with 10% margin): {format_size(space_info['required_space'])}")
    print(f"  Available disk space: {format_size(space_info['free_space'])}")

    if not space_info['has_enough_space']:
      print(f"  WARNING: Not enough disk space! Need {format_size(space_info['required_space'] - space_info['free_space'])} more")
    else:
      print(f"  Status: OK - Sufficient space available")

    print(f"\nWould process {len(files)} files:")
    for file_info in files:
      local_path = calculate_local_path(file_info['path'], args.dropbox_path, args.local_path)
      status = "DOWNLOAD" if not os.path.exists(local_path) or args.overwrite else "SKIP"
      size_str = f" ({file_info['size']})" if status == "DOWNLOAD" else ""
      print(f"  {status}: {file_info['path']} -> {local_path}{size_str}")
  else:
    sync_folder(args.dropbox_path, args.local_path, args.overwrite, args.stall_timeout)


if __name__ == "__main__":
  main()