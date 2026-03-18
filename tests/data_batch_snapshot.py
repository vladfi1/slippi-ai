"""Snapshot test for data batches.

Loads one batch from a DataSource, then either:
  --write --path PATH: pickles the batch to PATH
  --read --path PATH:  loads the pickle and asserts equality with a fresh batch

Usage:
  python tests/data_batch_snapshot.py --write --path /tmp/batch.pkl
  python tests/data_batch_snapshot.py --read --path /tmp/batch.pkl
"""

import pickle

from absl import app, flags

from slippi_ai import data, paths, utils

FLAGS = flags.FLAGS

flags.DEFINE_string('path', None, 'Path to read/write the batch snapshot.')
flags.DEFINE_boolean('write', False, 'Write mode: save batch to --path.')
flags.DEFINE_boolean('read', False, 'Read mode: compare batch against --path.')
flags.DEFINE_string('dataset_path', str(paths.TOY_DATASET), 'Path to the dataset.')
flags.DEFINE_integer('batch_size', 2, 'Batch size.')
flags.DEFINE_integer('unroll_length', 16, 'Unroll length.')
flags.DEFINE_integer('seed', 0, 'Dataset seed.')


def make_source() -> data.DataSource:
  dataset_config = data.DatasetConfig(
      dataset_path=FLAGS.dataset_path,
      seed=FLAGS.seed,
  )
  replays = data.replays_from_meta(dataset_config)
  return data.DataSource(
      replays=replays,
      batch_size=FLAGS.batch_size,
      unroll_length=FLAGS.unroll_length,
      extra_frames=1,
      compressed=True,
  )


def main(_):
  if not FLAGS.path:
    raise ValueError('--path is required.')
  if FLAGS.write == FLAGS.read:
    raise ValueError('Exactly one of --write or --read must be set.')

  source = make_source()
  batch, epoch = next(source)

  if FLAGS.write:
    with open(FLAGS.path, 'wb') as f:
      pickle.dump(batch, f)
    print(f'Wrote batch snapshot to {FLAGS.path}')
  else:
    with open(FLAGS.path, 'rb') as f:
      expected = pickle.load(f)
    errors = utils.check_same_structure(batch, expected, equal=True)
    if errors:
      for path, msg in errors:
        print(f'  {".".join(str(p) for p in path)}: {msg}')
      raise AssertionError(f'{len(errors)} mismatches found.')
    print('Batch matches snapshot.')


if __name__ == '__main__':
  app.run(main)
