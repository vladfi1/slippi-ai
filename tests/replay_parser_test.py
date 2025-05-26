"""Test to verify that replay parsing is the same using libmelee and peppi."""

import os
import tempfile
import zipfile
from pathlib import Path

import requests
from absl import app
from absl import flags

from slippi_db import preprocessing
from slippi_db import utils

TEST_DATASET_URL = "https://www.dropbox.com/scl/fi/xbja5vqqlg3m8jutyjcn7/TestDataset-32.zip?rlkey=nha6ycc6npr3wmxzickeyqpfh&st=i87xxfxk&dl=1"

FLAGS = flags.FLAGS
flags.DEFINE_string("url", TEST_DATASET_URL, "URL to download the test dataset from")
flags.DEFINE_string("temp_dir", None, "Temporary directory to extract files to (default: system temp dir)")
flags.DEFINE_boolean("keep_files", False, "Whether to keep the downloaded and extracted files")


def download_file(url, destination):
  """Download a file from a URL to a local destination."""
  print(f"Downloading from {url} to {destination}...")
  response = requests.get(url, stream=True)
  response.raise_for_status()

  with open(destination, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
      f.write(chunk)

  print(f"Download complete: {destination}")
  return destination


def extract_zip(zip_path, extract_dir):
  """Extract a zip file to a directory."""
  print(f"Extracting {zip_path} to {extract_dir}...")
  with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)
  print(f"Extraction complete: {extract_dir}")
  return extract_dir


def test_parsing_equality(directory):
  """Test that libmelee and peppi parse SLP files the same way."""
  files = utils.traverse_slp_files(directory)
  print(f"Found {len(files)} .slp files to test")

  results = {
    "passed": 0,
    "failed": 0,
    "errors": []
  }

  for i, file in enumerate(files):
    print(f"Testing file {i+1}/{len(files)}: {file.name}")
    try:
      with file.extract("") as path:
        preprocessing.assert_same_parse(path)
        results["passed"] += 1
    except AssertionError as e:
      print(f"  FAIL: {e}")
      results["failed"] += 1
      results["errors"].append((file.name, str(e)))
    except Exception as e:
      print(f"  ERROR: {e}")
      results["failed"] += 1
      results["errors"].append((file.name, str(e)))

  return results


def main(_):
  base_temp_dir = FLAGS.temp_dir or tempfile.gettempdir()
  with tempfile.TemporaryDirectory(dir=base_temp_dir) as temp_dir:
    zip_path = os.path.join(temp_dir, "test_dataset.zip")
    download_file(FLAGS.url, zip_path)

    extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    extract_zip(zip_path, extract_dir)

    results = test_parsing_equality(extract_dir)

    print("\nTest Summary:")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")

    if results["errors"]:
      print("\nFailures:")
      for name, error in results["errors"]:
        print(f"  {name}: {error}")

    if FLAGS.keep_files:
      keep_dir = os.path.join(os.getcwd(), "test_dataset")
      print(f"\nKeeping files in {keep_dir}")
      if not os.path.exists(keep_dir):
        os.makedirs(keep_dir)
      os.system(f"cp -r {extract_dir}/* {keep_dir}/")

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
  app.run(main)
