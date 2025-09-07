#!/usr/bin/env python3
"""
Test script that downloads model and output files from URLs and runs unroll_agent.py tests.
"""

import sys
import tempfile
from pathlib import Path
from typing import List, Tuple
import urllib.request
import urllib.error

from slippi_ai import unroll_agent

TEST_CASES: List[Tuple[str, str]] = [
  (
    'https://dl.dropbox.com/scl/fi/bppnln3rfktxfdocottuw/all_d21_imitation_v3?rlkey=46yqbsp7vi5222x04qt4npbkq&st=6knz106y&dl=1',
    'https://dl.dropbox.com/scl/fi/ej321uyygfyiwobv1897a/all_d21_v3_demo_unroll_output.pkl?rlkey=zv9p1k559tza1sisbjxvx86tu&st=pjh8w101&dl=1'
  ),
]


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
        input_path=str(input_path),
        output_path=str(output_path),
        overwrite=False,
    )
  except ValueError:
    return False

  return True

def main():
  """Main test function."""
  if not TEST_CASES:
    print("Warning: No test cases defined. Add (model_url, output_url) pairs to TEST_CASES.")
    print("Test skipped.")
    return 0

  # Find a suitable input file from toy dataset
  from slippi_ai import paths
  try:
    input_path = next(paths.TOY_DATA_DIR.iterdir())
  except StopIteration:
    print("Error: No files found in toy dataset directory")
    return 1

  failures = []

  with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)

    for i, (model_url, output_url) in enumerate(TEST_CASES):
      print(f"\n--- Test case {i+1}/{len(TEST_CASES)} ---")

      # Download files
      model_path = tmpdir_path / f"model_{i}.pkl"
      output_path = tmpdir_path / f"output_{i}.pkl"

      try:
        download_file(model_url, model_path)
        download_file(output_url, output_path)
      except Exception as e:
        print(f"Failed to download files for test case {i+1}: {e}")
        failures.append((i+1, model_url, output_url, str(e)))
        continue

      # Run test
      if not run_unroll_test(model_path, output_path, input_path):
        failures.append((i+1, model_url, output_url, "Test execution failed"))

  # Report results
  print(f"\n=== Test Results ===")
  print(f"Total test cases: {len(TEST_CASES)}")
  print(f"Passed: {len(TEST_CASES) - len(failures)}")
  print(f"Failed: {len(failures)}")

  if failures:
    print("\nFailed test cases:")
    for case_num, model_url, output_url, error in failures:
      print(f"  Case {case_num}: {error}")
      print(f"  Model: {model_url}")
      print(f"  Output: {output_url}")
    return 1
  else:
    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
  sys.exit(main())
