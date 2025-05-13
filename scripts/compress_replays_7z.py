#!/usr/bin/env python3
"""
archive_replays_7z.py

Scans a directory (default: Replays/), identifies the files that are **not**
currently open, and then calls the external **7-Zip** tool to add everything
else to a ZIP archive (default: Replays.zip).
After a successful 7-Zip run the script deletes the files that were archived.

Open-file detection order
-------------------------
1. If **lsof** is available, any file that appears in `lsof <file>` is treated
   as “open”.
2. If lsof is missing, a file counts as open **unless** it is older than one
   day (mtime > 24 h).

The list of *open* files is written to a temporary list-file, which is passed
to 7-Zip with the `-x@<list>` flag so that only **closed** files are added.
"""

import os
import subprocess, shutil
import sys
import tempfile
import time
from pathlib import Path

from absl import app, flags
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("source_dir", "Replays", "Directory to scan for replay files.")
flags.DEFINE_string("zip_filename", "Replays.zip", "Target ZIP archive path.")
flags.DEFINE_integer("cores", None, "Number of cores to use.")

ONE_DAY_SECS = 24 * 60 * 60


# ---------- utility helpers -------------------------------------------------
def cmd_exists(cmd: str) -> bool:
  """Return True iff *cmd* is on the PATH and can be executed."""
  return shutil.which(cmd) is not None
  # return subprocess.call(["command", "-v", cmd], stdout=subprocess.DEVNULL,
  #                        stderr=subprocess.DEVNULL, shell=False) == 0


def file_is_open_lsof(path: Path) -> bool:
  """True if *path* shows up in `lsof` output (i.e. some process has it open)."""
  try:
    result = subprocess.run(["lsof", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True)
    return result.returncode == 0 and bool(result.stdout.strip())
  except FileNotFoundError:    # should not happen if we checked cmd_exists
    return False


def file_is_open_mtime(path: Path) -> bool:
  """Fallback check: treat files newer than 1 day as still open."""
  return (time.time() - path.stat().st_mtime) < ONE_DAY_SECS


# ---------- main archiving routine -----------------------------------------
def archive_replays(source_dir: Path, zip_path: Path, cores: Optional[int] = None) -> None:
  if not source_dir.is_dir():
    sys.exit(f"[ERROR] Source directory {source_dir} does not exist.")

  lsof_ok = cmd_exists("lsof")

  seven_zip_ok = False
  for seven_zip_command in ("7z", "7zz", "7z.exe"):
    if cmd_exists(seven_zip_command):
      seven_zip_ok = True
      break
  if not seven_zip_ok:
    sys.exit("[ERROR] 7-Zip (‘7z’ or ‘7zz’) is not installed or not on PATH.")

  # Collect all files under source_dir
  all_files = [p for p in source_dir.rglob("*") if p.is_file()]
  if not all_files:
    print("No files found - nothing to do.")
    return

  open_files, closed_files = [], []
  # check_fn = file_is_open_lsof if lsof_ok else file_is_open_mtime
  check_fn = file_is_open_mtime

  for p in tqdm(all_files, desc="Checking files", unit="file"):
    if check_fn(p):
      open_files.append(p)
    else:
      closed_files.append(p)

  if not closed_files:
    print("No eligible files to archive.")
    return

  print(f"Found {len(closed_files)} files to archive, "
        f"skipping {len(open_files)} open files.")
  print(f"Archiving to {zip_path}...")

  # ----------------------------------------------------------------------
  # 1. Write *open* files to a list so 7z can exclude them (-x@list).
  #  Paths in the list must be *relative* to source_dir because we’ll
  #  change cwd to source_dir when calling 7-Zip.
  # ----------------------------------------------------------------------
  with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".lst") as tmp:
    for p in open_files:
      tmp.write(f"{p.relative_to(source_dir)}\n")
    exclude_list = tmp.name

  # ----------------------------------------------------------------------
  # 2. Run 7-Zip: add everything under . (source_dir) except the patterns
  #  in exclude_list.  We cd into source_dir so that the archive contains
  #  only relative paths (no leading “Replays/” component).
  # ----------------------------------------------------------------------
  mmt = "on" if cores is None else cores
  seven_zip_cmd = [
      seven_zip_command,
      "a",     # add / update
      "-tzip", str(zip_path),
      ".",     # add everything from the current directory
      f"-x@{exclude_list}",
      f"-mmt={mmt}",
  ]

  print("\nRunning:", " ".join(seven_zip_cmd))
  proc = subprocess.run(seven_zip_cmd, cwd=source_dir)
  os.remove(exclude_list)  # clean up the temporary list file

  if proc.returncode != 0:
    sys.exit(f"[ERROR] 7-Zip exited with code {proc.returncode}")

  # ----------------------------------------------------------------------
  # 3. Delete the files we successfully archived.
  # ----------------------------------------------------------------------
  for p in tqdm(closed_files, desc="Deleting archived files", unit="file"):
    try:
      p.unlink()
    except Exception as e:
      print(f"[WARN] Could not delete {p}: {e}")

  print("Archiving complete.")


# ---------- entry-point -----------------------------------------------------
def main(argv):
  del argv
  archive_replays(
      Path(FLAGS.source_dir).expanduser(),
      Path(FLAGS.zip_filename).expanduser(),
      cores=FLAGS.cores,
  )


if __name__ == "__main__":
  app.run(main)
