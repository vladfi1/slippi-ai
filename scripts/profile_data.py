import dataclasses
import time

from absl import app, flags

from slippi_ai import data, paths
from slippi_ai import observations

DATA_DIR = flags.DEFINE_string('data_dir', str(paths.TOY_DATA_DIR), 'path to data directory')
META_PATH = flags.DEFINE_string('meta_path', str(paths.TOY_META_PATH), 'path to meta file')

BATCH_SIZE = flags.DEFINE_integer('batch_size', 32, 'batch size')
UNROLL_LENGTH = flags.DEFINE_integer('unroll_length', 32, 'unroll length')
RUNTIME = flags.DEFINE_integer('runtime', 5, 'runtime in seconds')

def main(_):
  dataset_config = data.DatasetConfig(
      data_dir=DATA_DIR.value,
      meta_path=META_PATH.value,
  )
  train, _ = data.train_test_split(dataset_config)

  data_config = data.DataConfig(
      batch_size=BATCH_SIZE.value,
      unroll_length=UNROLL_LENGTH.value,
      num_workers=0,
  )

  source = data.make_source(
      replays=train,
      observation_config=observations.ObservationConfig(),
      **dataclasses.asdict(data_config),
  )

  start = time.perf_counter()
  batches = 0
  while time.perf_counter() - start < RUNTIME.value:
    next(source)
    batches += 1
  run_time = time.perf_counter() - start
  bps = batches / run_time

  frames_per_batch = data_config.unroll_length * data_config.batch_size
  fps = frames_per_batch * bps
  print(f'bps={bps:.2f} fps={fps:.2f}')

if __name__ == '__main__':
  app.run(main)
