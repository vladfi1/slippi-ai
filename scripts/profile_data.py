import time
from absl import app, flags

from slippi_ai import data, embed

DATA_DIR = flags.DEFINE_string('data_dir', None, 'path to data directory')

BATCH_SIZE = flags.DEFINE_integer('batch_size', 32, 'batch size')
UNROLL_LENGTH = flags.DEFINE_integer('unroll_length', 32, 'unroll length')
RUNTIME = flags.DEFINE_integer('runtime', 15, 'runtime in seconds')

def main(_):
  dataset_config = data.DatasetConfig(data_dir=DATA_DIR.value)
  train, _ = data.train_test_split(dataset_config)

  data_config = data.CONFIG.copy()
  data_config.update(
      batch_size=BATCH_SIZE.value,
      unroll_length=UNROLL_LENGTH.value,
  )
  del data_config['in_parallel']

  source = data.DataSource(
      train,
      embed_controller=embed.get_controller_embedding(axis_spacing=16),
      **data_config,
  )

  start = time.perf_counter()
  while time.perf_counter() - start < RUNTIME.value:
    next(source)

if __name__ == '__main__':
  app.run(main)
