import dataclasses
import time
import logging

from absl import app, flags
import fancyflags as ff

from slippi_ai import data as data_lib, paths, flag_utils
from slippi_ai import observations

_field = lambda f: dataclasses.field(default_factory=f)

@dataclasses.dataclass
class Config:
  dataset: data_lib.DatasetConfig = _field(data_lib.DatasetConfig)
  data: data_lib.DataConfig = _field(data_lib.DataConfig)
  observation: observations.ObservationConfig = _field(observations.ObservationConfig)


DEFAULT_CONFIG = Config()
DEFAULT_CONFIG.dataset.dataset_path = str(paths.TOY_DATASET)

CONFIG = ff.DEFINE_dict('config', **flag_utils.get_flags_from_default(DEFAULT_CONFIG))

RUNTIME = flags.DEFINE_integer('runtime', 5, 'runtime in seconds')



def main(_):
  logging.getLogger("dropbox").setLevel(logging.WARNING)

  config = flag_utils.dataclass_from_dict(Config, CONFIG.value)
  train, _ = data_lib.train_test_split(config.dataset)

  source = data_lib.make_source(
      replays=train,
      observation_config=config.observation,
      **dataclasses.asdict(config.data),
  )

  print('Warming up...')
  start = time.perf_counter()
  next(source)  # warmup
  print(f'Warmup took {time.perf_counter() - start:.2f} seconds')

  start = time.perf_counter()
  batches = 0
  while time.perf_counter() - start < RUNTIME.value:
    next(source)
    batches += 1
  run_time = time.perf_counter() - start
  bps = batches / run_time
  spb = 1 / bps

  frames_per_batch = config.data.unroll_length * config.data.batch_size
  fps = frames_per_batch * bps

  frames_per_minute = 60 * 60
  mps = fps / frames_per_minute

  print(f'bps={bps:.2f} spb={spb:.3f} fps={fps:.0f} mps={mps:.1f}')

if __name__ == '__main__':
  app.run(main)
