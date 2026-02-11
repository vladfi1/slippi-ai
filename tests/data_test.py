import unittest

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


if __name__ == '__main__':
    unittest.main()
