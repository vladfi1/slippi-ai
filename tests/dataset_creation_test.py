"""Test to verify dataset creation functions in slippi_db."""

import os
import pickle
import tempfile

from absl import app
from absl import flags

from slippi_db import parse_local
from replay_parser_test import TEST_DATASET_URL, download_file, extract_zip

FLAGS = flags.FLAGS
flags.DEFINE_integer("threads", 1, "Number of threads to use for parallel processing")


def setup_dataset_root(temp_dir):
  """Set up the dataset root directory structure."""
  root_dir = os.path.join(temp_dir, "dataset_root")
  raw_dir = os.path.join(root_dir, "Raw")
  parsed_dir = os.path.join(root_dir, "Parsed")

  os.makedirs(raw_dir, exist_ok=True)
  os.makedirs(parsed_dir, exist_ok=True)

  return root_dir, raw_dir, parsed_dir


def download_test_dataset(raw_dir, temp_dir):
  """Download and extract the test dataset to the Raw directory."""
  zip_path = os.path.join(temp_dir, "test_dataset.zip")
  download_file(TEST_DATASET_URL, zip_path)

  extract_dir = os.path.join(temp_dir, "extracted")
  os.makedirs(extract_dir, exist_ok=True)
  extract_zip(zip_path, extract_dir)

  slp_files = []
  for root, _, files in os.walk(extract_dir):
    for file in files:
      if file.endswith('.slp'):
        slp_files.append(os.path.join(root, file))

  if slp_files:
    import zipfile
    test_zip_path = os.path.join(raw_dir, "test_dataset.zip")
    with zipfile.ZipFile(test_zip_path, 'w') as zipf:
      for file in slp_files:
        arcname = os.path.relpath(file, extract_dir)
        zipf.write(file, arcname)

    print(f"Created zip with {len(slp_files)} .slp files at {test_zip_path}")
    return [os.path.basename(test_zip_path)]
  else:
    print("No .slp files found in the extracted dataset")
    return []


def test_dataset_creation(threads=1):
  """Test the dataset creation functions in slippi_db."""
  with tempfile.TemporaryDirectory() as temp_dir:
    root_dir, raw_dir, parsed_dir = setup_dataset_root(temp_dir)

    raw_files = download_test_dataset(raw_dir, temp_dir)
    print(f"Downloaded {len(raw_files)} files to {raw_dir}")

    parse_local.run_parsing(
      root=root_dir,
      num_threads=threads,
      in_memory=True,
      reprocess=False,
      dry_run=False
    )

    parsed_pkl_path = os.path.join(root_dir, "parsed.pkl")
    assert os.path.exists(parsed_pkl_path), f"parsed.pkl not found at {parsed_pkl_path}"

    with open(parsed_pkl_path, "rb") as f:
      parsed_data = pickle.load(f)

    print(f"Parsed data contains {len(parsed_data)} entries")

    invalid_entries = [entry for entry in parsed_data if not entry.get("valid", False)]
    non_training_entries = [entry for entry in parsed_data if not entry.get("is_training", False)]

    assert len(invalid_entries) == 0, f"Found {len(invalid_entries)} invalid entries"
    assert len(non_training_entries) == 0, f"Found {len(non_training_entries)} non-training entries"

    return parsed_data


def test_parallel_dataset_creation():
  """Test dataset creation with parallel processing."""
  threads = 4
  parsed_data = test_dataset_creation(threads=threads)

  print(f"Successfully processed dataset with {threads} threads")
  return parsed_data


def main(_):
  """Main function to run the tests."""
  print("Running single-threaded test...")
  single_thread_data = test_dataset_creation(threads=1)

  print("\nRunning parallel test...")
  parallel_data = test_parallel_dataset_creation()

  assert len(single_thread_data) == len(parallel_data), (
    f"Single-threaded ({len(single_thread_data)} entries) and "
    f"parallel ({len(parallel_data)} entries) results differ"
  )

  print("\nAll tests passed!")
  return 0


if __name__ == "__main__":
  app.run(main)
