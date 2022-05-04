from absl import app
import ray

from slippi_db.run_parsing import (
  CLUSTER, ENV, WIPE, SERIAL, PARSER,
  CompressionType,
  process_all,
)

# use maximum compression for each
configurations = {
    'uncompressed': (CompressionType.NONE, None),
    'gzip': (CompressionType.GZIP, 9),
    'snappy': (CompressionType.SNAPPY, None),
    'brotli': (CompressionType.BROTLI, 11),
    'lz4': (CompressionType.LZ4, None),
    'zstd': (CompressionType.ZSTD, 22),
    'zlib': (CompressionType.ZLIB, 9),
}

def main(_):
  if CLUSTER.value:
    ray.init('auto')

  for dataset, (compression, compression_level) in configurations.items():
    print('process_all:', dataset)
    process_all(
        env=ENV.value,
        wipe=WIPE.value,
        serial=SERIAL.value,
        dataset=dataset,
        parser=PARSER.value,
        compression=compression,
        compression_level=compression_level,
    )

if __name__ == '__main__':
  app.run(main)