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


class TrajectoryManagerPadTest(unittest.TestCase):

    def test_pad_final_chunk(self):
        """Padded chunks cover every frame of a game exactly once."""
        dataset_config = data.DatasetConfig(
            data_dir=str(paths.TOY_DATA_DIR),
            meta_path=str(paths.TOY_META_PATH),
        )
        replays = data.replays_from_meta(dataset_config)
        unroll_length, overlap = 1000, 3
        manager = data.TrajectoryManager(
            iter(replays),
            unroll_length=unroll_length,
            encode_name=lambda _: 0,
            overlap=overlap,
            stagger=False,
            pad_final_chunk=True,
        )

        chunks = []
        while True:
            flat_batch, meta = manager.grab_chunk()
            chunks.append((flat_batch, meta))
            if len(chunks) > 1 and flat_batch[-3][0]:  # is_resetting
                chunks.pop()
                break

        game_length = data.game_len(replays[0].read_game())
        flat_game = data.utils.cached_flatten(data.Game)(
            replays[0].read_game())
        stage = flat_game[0]  # any per-frame leaf
        self.assertGreater(game_length % (unroll_length - overlap), 0)

        covered = []
        for flat_batch, meta in chunks:
            is_valid = flat_batch[-1]
            rewards = flat_batch[-2]
            num_valid = is_valid.sum()
            # Validity is a prefix.
            np.testing.assert_array_equal(
                is_valid, np.arange(unroll_length) < num_valid)
            self.assertGreater(num_valid, overlap)
            chunk_stage = flat_batch[0]
            np.testing.assert_array_equal(
                chunk_stage[:num_valid],
                stage[meta.start:meta.start + num_valid])
            # Padding repeats the last frame and has zero reward.
            np.testing.assert_array_equal(
                chunk_stage[num_valid:], chunk_stage[num_valid - 1])
            np.testing.assert_array_equal(rewards[num_valid:], 0)
            self.assertEqual(rewards.shape, (unroll_length - 1,))
            start = meta.start if not covered else meta.start + overlap
            covered.extend(range(start, meta.start + num_valid))

        np.testing.assert_array_equal(covered, np.arange(game_length))
        self.assertLess(chunks[-1][0][-1].sum(), unroll_length)


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
