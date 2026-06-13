"""Periodically remove files older than a specified age from a directory."""

import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path


def parse_duration(s: str) -> timedelta:
  """Parse a duration string like '1d', '2h', '30m' into a timedelta."""
  units = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400, 'w': 604800}
  if s[-1] in units:
    return timedelta(seconds=int(s[:-1]) * units[s[-1]])
  return timedelta(seconds=int(s))


def format_size(num_bytes: int) -> str:
  for unit in ("B", "KB", "MB", "GB", "TB"):
    if num_bytes < 1024 or unit == "TB":
      return f"{num_bytes:.1f} {unit}"
    num_bytes /= 1024


def cleanup(directory: Path, max_age: timedelta, dry_run: bool) -> tuple[int, int]:
  """Remove files older than max_age under directory. Returns (count, total_bytes)."""
  cutoff = datetime.now() - max_age
  deleted = 0
  total_bytes = 0
  for root, dirs, files in os.walk(directory):
    for name in files:
      path = Path(root) / name
      try:
        st = path.stat()
        mtime = datetime.fromtimestamp(st.st_mtime)
        if mtime < cutoff:
          size = st.st_size
          if not dry_run:
            path.unlink()
          deleted += 1
          total_bytes += size
      except OSError as e:
        logging.warning("Skipping %s: %s", path, e)
  return deleted, total_bytes


def main() -> None:
  parser = argparse.ArgumentParser(description="Periodically delete old files from a directory.")
  parser.add_argument("directory", type=Path, help="Directory to clean up.")
  parser.add_argument(
    "--max-age", default="30d",
    help="Delete files older than this (e.g. 30d, 2w, 24h). Default: 30d.",
  )
  parser.add_argument(
    "--interval", default="1d",
    help="How often to run cleanup (e.g. 1d, 6h, 30m). Default: 1d.",
  )
  parser.add_argument(
    "--once", action="store_true",
    help="Run cleanup once and exit.",
  )
  parser.add_argument(
    "--dry-run", action="store_true",
    help="Log what would be deleted without actually deleting.",
  )
  parser.add_argument(
    "--log-level", default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
  )
  args = parser.parse_args()

  logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s %(levelname)s %(message)s",
  )

  directory: Path = args.directory.expanduser().resolve()
  if not directory.is_dir():
    parser.error(f"Not a directory: {directory}")

  max_age = parse_duration(args.max_age)
  interval = parse_duration(args.interval)

  logging.info(
    "Cleanup config: dir=%s max_age=%s interval=%s dry_run=%s",
    directory, args.max_age, args.interval, args.dry_run,
  )

  while True:
    logging.info("Running cleanup pass...")
    n, total_bytes = cleanup(directory, max_age, args.dry_run)
    verb = "would be deleted" if args.dry_run else "deleted"
    logging.info("Cleanup done: %d file(s) %s (%s).", n, verb, format_size(total_bytes))
    if args.once:
      break
    logging.info("Sleeping for %s...", args.interval)
    time.sleep(interval.total_seconds())


if __name__ == "__main__":
  main()
