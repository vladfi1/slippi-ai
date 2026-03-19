import glob as glob_module
import io
import json
import os
import tempfile
import unittest

import webdataset as wds
import numpy as np

from slippi_ai import data, paths


class TrainTestSplitDatasetPathTest(unittest.TestCase):

    def test_train_test_split_with_dataset_path(self):
        config = data.DatasetConfig(
            dataset_path=str(paths.TOY_DATASET),
        )
        train, test = data.train_test_split(config)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        # No overlap between train and test
        train_paths = set(id(r.path) if not isinstance(r.path, str) else r.path for r in train)
        test_paths = set(id(r.path) if not isinstance(r.path, str) else r.path for r in test)
        # Every replay should have metadata
        for r in train + test:
            self.assertNotEqual(r.meta, ())

    def test_train_test_split_with_archive(self):
        config = data.DatasetConfig(
            archive=str(paths.TOY_DATASET / 'Dataset.zip'),
        )
        train, test = data.train_test_split(config)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)


def _make_toy_wds(output_dir: str):
    """Write a small WDS shard from the toy dataset and return its glob path."""

    toy_config = data.DatasetConfig(
        data_dir=str(paths.TOY_DATA_DIR),
        meta_path=str(paths.TOY_META_PATH),
        swap=True,
    )
    data.write_wds_shards(toy_config, output_dir)


class DataSourceWdsTest(unittest.TestCase):

    def test_data_source_wds(self):
        """DataSource(wds_shards=...) produces valid Batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_toy_wds(tmpdir)

            source = data.WebDataSource(
                dataset_path=tmpdir,
                split='train',
                batch_size=2,
                unroll_length=16,
            )

            batch, epoch = next(source)
            self.assertEqual(batch.batch.game.stage.shape, (2, 17))


if __name__ == '__main__':
    unittest.main()
