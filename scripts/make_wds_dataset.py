"""Convert an existing dataset to WebDataset tar shards."""

import os

from absl import app, flags
import fancyflags as ff

from slippi_ai import flag_utils, data

DS_CONFIG = ff.DEFINE_dict(
    'dataset', **flag_utils.get_flags_from_dataclass(data.DatasetConfig))

SHARD_CONFIG = ff.DEFINE_dict(
    'sharding', **flag_utils.get_flags_from_dataclass(data.ShardWriterConfig))

OUTPUT_DIR = flags.DEFINE_string(
  'output_dir', None, 'Directory to write WDS shards to.',
  required=True)

def main(_):
  os.makedirs(OUTPUT_DIR.value, exist_ok=True)

  dataset_config = flag_utils.dataclass_from_dict(
    data.DatasetConfig, DS_CONFIG.value)
  shard_config = flag_utils.dataclass_from_dict(
    data.ShardWriterConfig, SHARD_CONFIG.value)

  data.write_wds_shards(
      config=dataset_config,
      output_dir=OUTPUT_DIR.value,
      shard_writer_config=shard_config,
  )

if __name__ == '__main__':
  app.run(main)
