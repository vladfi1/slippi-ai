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
    str(paths.DEMO_CHECKPOINT),
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


def run_unroll_test(model_path: Path, output_path: Path, input_path: Path) -> bool:
  """Run the unroll_agent.py test with the given model and output files."""

  try:
    unroll_agent.test_or_save_outputs(
        model_path=str(model_path),
        input_dir=str(input_path),
        output_path=str(output_path),
        overwrite=False,
    )
  except ValueError:
    return False

  return True

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

  failures = []

  with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)

    for i, (model_source, output_source) in enumerate(TEST_CASES):
      print(f"\n--- Test case {i+1}/{len(TEST_CASES)} ---")

      try:
        model_path = get_file_path(model_source, tmpdir_path, f"model_{i}.pkl")
        output_path = get_file_path(output_source, tmpdir_path, f"output_{i}.pkl")
      except Exception as e:
        print(f"Failed to get files for test case {i+1}: {e}")
        failures.append((i+1, model_source, output_source, str(e)))
        continue

      # Run test
      if not run_unroll_test(model_path, output_path, input_path):
        failures.append((i+1, model_source, output_source, "Test execution failed"))

  # Report results
  print(f"\n=== Test Results ===")
  print(f"Total test cases: {len(TEST_CASES)}")
  print(f"Passed: {len(TEST_CASES) - len(failures)}")
  print(f"Failed: {len(failures)}")

  if failures:
    print("\nFailed test cases:")
    for case_num, model_source, output_source, error in failures:
      print(f"  Case {case_num}: {error}")
      print(f"  Model: {model_source}")
      print(f"  Output: {output_source}")
    return 1
  else:
    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
  sys.exit(main())
