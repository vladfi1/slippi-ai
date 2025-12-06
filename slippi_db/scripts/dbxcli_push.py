#!/usr/bin/env python3
"""
Dropbox synchronization script using dbxcli.
Uploads files from a local directory to a Dropbox folder without overwriting existing files by default.
"""

import subprocess
import os
import sys
import argparse
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


class ProgressTracker:
  """Tracks upload progress and estimates time remaining."""

  def __init__(self, total_bytes: int, total_files: int) -> None:
    self.total_bytes = total_bytes
    self.total_files = total_files
    self.transferred_bytes = 0
    self.completed_files = 0
    self.start_time = time.time()
    self.current_file = ""
    self.current_file_size = 0
    self.current_file_uploaded = 0
    self.last_progress_time = time.time()
    self.last_progress_size = 0

  def start_file(self, filename: str, file_size: int) -> None:
    """Start tracking a new file upload."""
    self.current_file = filename
    self.current_file_size = file_size
    self.current_file_uploaded = 0

  def update_file_progress(self, uploaded_bytes: int) -> None:
    """Update progress for the current file."""
    if uploaded_bytes > self.current_file_uploaded:
      self.last_progress_time = time.time()
      self.last_progress_size = uploaded_bytes
    self.current_file_uploaded = uploaded_bytes

  def is_stalled(self, timeout: int = 60) -> bool:
    """Check if upload has stalled (no progress for timeout seconds)."""
    if self.current_file_size == 0:
      return False
    return (time.time() - self.last_progress_time) > timeout

  def complete_file(self) -> None:
    """Mark current file as completed."""
    self.transferred_bytes += self.current_file_size
    self.completed_files += 1
    self.current_file = ""
    self.current_file_size = 0
    self.current_file_uploaded = 0

  def get_progress(self) -> Dict[str, Any]:
    """Get current progress statistics."""
    elapsed = time.time() - self.start_time
    current_total_transferred = self.transferred_bytes + self.current_file_uploaded

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
    # Folder might not exist, which is fine for push
    return files

  if not output:
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


def get_local_files(local_folder: str) -> List[Dict[str, Any]]:
  """Get list of archive files in local folder recursively."""
  files = []
  
  for root, dirs, filenames in os.walk(local_folder):
    for filename in filenames:
      # Only include archive files (.zip, .7z, .rar)
      if filename.endswith(('.zip', '.7z', '.rar')):
        file_path = os.path.join(root, filename)
        file_size = os.path.getsize(file_path)
        relative_path = os.path.relpath(file_path, local_folder)
        
        files.append({
          'local_path': file_path,
          'relative_path': relative_path,
          'name': filename,
          'size_bytes': file_size
        })
  
  return files


def calculate_dropbox_path(local_path: str, local_folder: str, dropbox_folder: str) -> str:
  """Calculate the Dropbox path for a given local file path."""
  relative_path = os.path.relpath(local_path, local_folder)
  # Ensure dropbox path uses forward slashes
  relative_path = relative_path.replace(os.sep, '/')
  # Ensure dropbox_folder ends with /
  if not dropbox_folder.endswith('/'):
    dropbox_folder += '/'
  return dropbox_folder + relative_path


def upload_file(
    local_path: str,
    dropbox_path: str,
    file_size: int,
    progress_tracker: Optional[ProgressTracker] = None,
    overwrite: bool = False,
    stall_timeout: int = 60,
    max_retries: int = 3
) -> bool:
  """Upload a single file to Dropbox with progress tracking and stall detection."""
  success = False
  attempts = 0
  
  while attempts < max_retries:
    try:
      if attempts > 0:
        print(f"\nRetrying upload (attempt {attempts + 1}/{max_retries})...")

      if progress_tracker:
        progress_tracker.start_file(local_path, file_size)
        progress_tracker.print_progress()

      # Create the directory structure in Dropbox if needed
      dropbox_dir = os.path.dirname(dropbox_path)
      if dropbox_dir:
        # Try to create the directory (will fail silently if it exists)
        subprocess.run(
            ["dbxcli", "mkdir", dropbox_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

      # Start the upload process
      process = subprocess.Popen(
          ["dbxcli", "put", local_path, dropbox_path],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.PIPE,
          text=True,
      )

      # Monitor upload progress (simplified since we can't easily track upload progress)
      start_time = time.time()
      while process.poll() is None:
        elapsed = time.time() - start_time
        
        if progress_tracker:
          # Simulate progress based on time (since we can't track actual upload progress)
          # This is a rough estimate
          estimated_progress = min(file_size, int(file_size * (elapsed / max(1, file_size / 1024 / 1024 * 10))))
          progress_tracker.update_file_progress(estimated_progress)
          progress_tracker.print_progress()

          # Check for stall (simplified check based on process still running)
          if elapsed > stall_timeout and elapsed > file_size / 1024 / 100:  # More than timeout and unreasonably slow
            print(f"\nUpload appears stalled after {int(elapsed)} seconds, terminating...")
            process.terminate()
            process.wait(timeout=5)
            if process.poll() is None:
              process.kill()
              process.wait()
            attempts += 1
            continue

        time.sleep(0.5)  # Check every 500ms

      # Wait for process to complete (if not already done)
      _, stderr = process.communicate()

      if process.returncode != 0:
        if progress_tracker:
          print()  # New line after progress bar
        # Check if error is because file already exists
        if "already exists" in stderr.lower() and not overwrite:
          # This is expected - file already exists and we're not overwriting
          if progress_tracker:
            progress_tracker.complete_file()
          return False
        raise subprocess.CalledProcessError(process.returncode, f"dbxcli put '{local_path}' '{dropbox_path}'", stderr)

      success = True

      if progress_tracker:
        progress_tracker.complete_file()
        progress_tracker.print_progress()

      return True

    except subprocess.CalledProcessError as e:
      if progress_tracker:
        print()  # New line after progress bar
      print(f"Failed to upload {local_path}: {e.stderr}")
      attempts += 1
      if attempts < max_retries:
        print(f"Will retry...")
        continue
      return False

  # If we exhausted all retries
  if progress_tracker:
    print()  # New line after progress bar
  print(f"Failed to upload {local_path} after {max_retries} attempts")
  return False


def check_files_to_upload(local_files: List[Dict[str, Any]], dropbox_files: List[Dict[str, Any]], 
                          local_folder: str, dropbox_folder: str, overwrite: bool = False) -> Dict[str, Any]:
  """Determine which files need to be uploaded."""
  # Create a set of existing Dropbox file paths for quick lookup
  existing_paths = {file_info['path'] for file_info in dropbox_files}
  
  files_to_upload = []
  total_size_needed = 0
  
  for file_info in local_files:
    dropbox_path = calculate_dropbox_path(file_info['local_path'], local_folder, dropbox_folder)
    
    # Check if file exists in Dropbox
    if dropbox_path not in existing_paths or overwrite:
      files_to_upload.append(file_info)
      total_size_needed += file_info['size_bytes']
  
  return {
    'files_to_upload': files_to_upload,
    'total_size': total_size_needed,
    'num_files': len(files_to_upload),
    'total_files': len(local_files)
  }


def sync_folder(local_folder: str, dropbox_folder: str, overwrite: bool = False, stall_timeout: int = 60) -> None:
  """Synchronize a local folder to a Dropbox folder."""
  print(f"Synchronizing {local_folder} to {dropbox_folder}")

  # Check if local folder exists
  if not os.path.exists(local_folder):
    print(f"Error: Local folder {local_folder} does not exist")
    return

  # Get list of local files
  print("Discovering local files...")
  local_files = get_local_files(local_folder)

  if not local_files:
    print("No archive files found to upload.")
    return

  print(f"Found {len(local_files)} local archive files")

  # Get list of existing Dropbox files
  print("Checking existing files on Dropbox...")
  dropbox_files = get_dropbox_files(dropbox_folder)
  print(f"Found {len(dropbox_files)} existing files on Dropbox")

  # Determine which files to upload
  upload_info = check_files_to_upload(local_files, dropbox_files, local_folder, dropbox_folder, overwrite)

  print(f"\nFiles to upload: {upload_info['num_files']} of {upload_info['total_files']}")
  print(f"Total size to upload: {format_size(upload_info['total_size'])}")

  if upload_info['num_files'] == 0:
    print("All files already exist on Dropbox. Nothing to upload.")
    return

  print("\nProceeding with upload...")

  # Initialize progress tracker for files that will be uploaded
  progress_tracker = ProgressTracker(upload_info['total_size'], upload_info['num_files'])

  uploaded = 0
  skipped = 0
  failed = 0

  print()  # Empty line before progress bar
  for file_info in local_files:
    local_path = file_info['local_path']
    dropbox_path = calculate_dropbox_path(local_path, local_folder, dropbox_folder)

    # Check if file needs to be uploaded
    needs_upload = any(f['local_path'] == local_path for f in upload_info['files_to_upload'])

    if needs_upload:
      if upload_file(local_path, dropbox_path, file_info['size_bytes'], progress_tracker, overwrite, stall_timeout):
        uploaded += 1
      else:
        failed += 1
    else:
      # File already exists and not overwriting - just count as skipped
      skipped += 1

  print()  # New line after final progress update
  print(f"\nSync completed:")
  print(f"  Uploaded: {uploaded}")
  print(f"  Skipped: {skipped}")
  print(f"  Failed: {failed}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Upload local files to Dropbox folder using dbxcli")
  parser.add_argument("local_path", help="Local directory path to sync from")
  parser.add_argument("dropbox_path", help="Dropbox folder path to sync to")
  parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
  parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without actually uploading")
  parser.add_argument("--stall-timeout", type=int, default=60, help="Timeout in seconds to detect stalled uploads (default: 60)")

  args = parser.parse_args()

  # Check if dbxcli is available
  try:
    run_command("dbxcli version")
  except (subprocess.CalledProcessError, FileNotFoundError):
    print("Error: dbxcli not found. Please install and configure dbxcli first.")
    sys.exit(1)

  if args.dry_run:
    print("DRY RUN MODE - No files will be uploaded")
    
    # Check if local folder exists
    if not os.path.exists(args.local_path):
      print(f"Error: Local folder {args.local_path} does not exist")
      return
    
    local_files = get_local_files(args.local_path)
    
    if not local_files:
      print("No archive files found.")
      return

    # Get existing Dropbox files
    dropbox_files = get_dropbox_files(args.dropbox_path)
    
    # Determine which files to upload
    upload_info = check_files_to_upload(local_files, dropbox_files, args.local_path, args.dropbox_path, args.overwrite)

    print(f"\nUpload analysis:")
    print(f"  Files to upload: {upload_info['num_files']} of {upload_info['total_files']}")
    print(f"  Total size to upload: {format_size(upload_info['total_size'])}")

    print(f"\nWould process {len(local_files)} files:")
    for file_info in local_files:
      dropbox_path = calculate_dropbox_path(file_info['local_path'], args.local_path, args.dropbox_path)
      needs_upload = any(f['local_path'] == file_info['local_path'] for f in upload_info['files_to_upload'])
      status = "UPLOAD" if needs_upload else "SKIP"
      size_str = f" ({format_size(file_info['size_bytes'])})" if status == "UPLOAD" else ""
      print(f"  {status}: {file_info['local_path']} -> {dropbox_path}{size_str}")
  else:
    sync_folder(args.local_path, args.dropbox_path, args.overwrite, args.stall_timeout)


if __name__ == "__main__":
  main()