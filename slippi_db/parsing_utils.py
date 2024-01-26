import enum
import io
from typing import Optional
import zlib

import pyarrow as pa
import pyarrow.parquet as pq

from slippi_ai import types

class CompressionType(enum.Enum):
  # zlib compresses parquet file itself
  # To read it, set compress=True in slippi_ai.data.make_source
  ZLIB = 'zlib'
  SNAPPY = 'snappy'
  GZIP = 'gzip'
  BROTLI = 'brotli'
  LZ4 = 'lz4'
  ZSTD = 'zstd'
  NONE = 'none'

  def for_parquet(self) -> str:
    if self is CompressionType.ZLIB:
      return CompressionType.NONE.value
    return self.value

def convert_game(
    game: types.GAME_TYPE,  # a pyarrow StructArray
    pq_version: str = '2.4',
    compression: CompressionType = CompressionType.NONE,
    compression_level: Optional[int] = None,
) -> bytes:
  table = pa.Table.from_arrays([game], names=['root'])
  pq_file = io.BytesIO()

  if compression == CompressionType.ZLIB:
    pq_compression_level = None
  else:
    pq_compression_level = compression_level

  pq.write_table(
      table, pq_file,
      version=pq_version,
      compression=compression.for_parquet(),
      compression_level=pq_compression_level,
      use_dictionary=False,
  )
  pq_bytes = pq_file.getvalue()

  if compression == CompressionType.ZLIB:
    level = -1 if compression_level is None else compression_level
    pq_bytes = zlib.compress(pq_bytes, level=level)
  return pq_bytes
