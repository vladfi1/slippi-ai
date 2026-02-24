"""Test to verify dataset creation functions in slippi_db."""

import os
import pickle
import shutil
import tempfile

from absl import app
from absl import flags

from slippi_db import parse_local
from replay_parser_test import TEST_DATASET_URL, download_file

FLAGS = flags.FLAGS
flags.DEFINE_integer("threads", 4, "Number of threads to use for parallel processing")
flags.DEFINE_string("dataset_zip", None, "Path to a pre-downloaded test dataset zip file")


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
  zip_path = os.path.join(raw_dir, "test_dataset.zip")
  if FLAGS.dataset_zip:
    shutil.copy2(FLAGS.dataset_zip, zip_path)
  else:
    download_file(TEST_DATASET_URL, zip_path)


def test_dataset_creation(threads=1) -> dict[str, dict]:
  """Test the dataset creation functions in slippi_db."""
  with tempfile.TemporaryDirectory() as temp_dir:
    root_dir, raw_dir, parsed_dir = setup_dataset_root(temp_dir)

    download_test_dataset(raw_dir, temp_dir)

    parse_local.run_parsing(
        root=root_dir,
        num_threads=threads,
        in_memory=True,
        reprocess=False,
        dry_run=False,
    )

    parsed_pkl_path = os.path.join(root_dir, "parsed.pkl")
    assert os.path.exists(parsed_pkl_path), f"parsed.pkl not found at {parsed_pkl_path}"

    with open(parsed_pkl_path, "rb") as f:
      parsed_data = pickle.load(f)

    print(f"Parsed data contains {len(parsed_data)} entries")

    invalid_entries = [entry for entry in parsed_data if not entry.get("valid", False)]
    non_training_entries = [entry for entry in parsed_data if not entry.get("is_training", False)]

    # There is one known invalid entry in the test dataset, with multiple Gecko codes.
    assert len(invalid_entries) == 0, f"Found {len(invalid_entries)} invalid entries"
    assert len(non_training_entries) == 0, f"Found {len(non_training_entries)} non-training entries"

    print(f"Successfully processed dataset with {threads} threads")

    return {row['slp_md5']: row for row in parsed_data}


def main(_):
  """Main function to run the tests."""
  print("Running single-threaded test...")
  single_thread_data = test_dataset_creation(threads=1)

  print("\nRunning parallel test...")
  parallel_data = test_dataset_creation(threads=FLAGS.threads)

  assert single_thread_data == parallel_data, (
      "Data mismatch between single-threaded and multi-threaded processing"
  )

  print("\nAll tests passed!")
  return 0


if __name__ == "__main__":
  app.run(main)
