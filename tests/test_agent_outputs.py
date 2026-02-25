#!/usr/bin/env python3
"""
Test script that downloads model and output files from URLs or uses local files and runs unroll_agent.py tests.
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple
import urllib.request
import urllib.error

from slippi_ai import paths, unroll_agent

TEST_CASES: List[Tuple[str, str]] = [
  (
    str(paths.DEMO_IMITATION_CHECKPOINT),
    str(paths.AGENT_OUTPUTS_DIR / 'demo.pkl')
  ),
  (
    'https://dl.dropbox.com/scl/fi/bppnln3rfktxfdocottuw/all_d21_imitation_v3?rlkey=46yqbsp7vi5222x04qt4npbkq&st=6knz106y&dl=1',
    str(paths.AGENT_OUTPUTS_DIR / 'all_d21_imitation_v3.pkl')
  ),
]


def is_url(path: str) -> bool:
  """Check if a path is a URL."""
  return path.startswith(('http://', 'https://'))


def get_file_path(source: str, dest_dir: Path, filename: str) -> Path:
  """Get file path, downloading if source is URL or copying if local file."""
  dest_path = dest_dir / filename

  if is_url(source):
    download_file(source, dest_path)
  else:
    # Source is a local file
    source_path = Path(source)
    if not source_path.exists():
      raise FileNotFoundError(f"Local file not found: {source}")

    # Copy the file to destination
    import shutil
    shutil.copy2(source_path, dest_path)
    print(f"Copied local file {source} to {dest_path}")

  return dest_path


def download_file(url: str, dest_path: Path) -> None:
  """Download a file from URL to destination path."""
  print(f"Downloading {url} to {dest_path}")
  try:
    urllib.request.urlretrieve(url, dest_path)
    print(f"Successfully downloaded {dest_path.name}")
  except urllib.error.URLError as e:
    print(f"Failed to download {url}: {e}")
    raise


def parse_args():
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser(
    description="Test agent outputs with model and output files (URLs or local files)"
  )
  parser.add_argument(
    "--input-dir",
    default=None,
    help="Path to input directory (defaults to TOY_DATA_DIR)"
  )
  return parser.parse_args()


def main():
  """Main test function."""
  args = parse_args()

  input_path = Path(args.input_dir) if args.input_dir else paths.TOY_DATA_DIR

  with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)

    for i, (model_source, output_source) in enumerate(TEST_CASES):
      print(f"\n--- Test case {i+1}/{len(TEST_CASES)} ---")

      model_path = get_file_path(model_source, tmpdir_path, f"model_{i}.pkl")
      output_path = get_file_path(output_source, tmpdir_path, f"output_{i}.pkl")

      # Run test
      unroll_agent.test_or_save_outputs(
          model_path=str(model_path),
          input_dir=str(input_path),
          output_path=str(output_path),
          overwrite=False,
      )

  print("\nAll tests passed!")
  return 0


if __name__ == "__main__":
  sys.exit(main())
