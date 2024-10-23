import typing as tp

import numpy as np
import tensorflow as tf

from slippi_ai import embed, nametags
from slippi_ai import data as data_lib

T = tp.TypeVar('T')

def merge(structs: tp.Sequence[T], axis=1) -> T:
  return tf.nest.map_structure(lambda *xs: tf.concat(xs, axis), *structs)

def split(struct: T, axis=1, num_splits=2) -> list[T]:
  flat_struct = tf.nest.flatten(struct)
  flat_splits = [tf.split(x, num_splits, axis=axis) for x in flat_struct]
  split_flats = zip(*flat_splits)
  return [
      tf.nest.pack_sequence_as(struct, split_flat)
      for split_flat in split_flats
  ]

class TwoPlayerBatch(tp.NamedTuple):
  p0_frames: data_lib.Frames
  p1_frames: data_lib.Frames
  needs_reset: bool
  count: int  # For reproducing batches
  meta: data_lib.ChunkMeta

def convert_batch(
    batch: data_lib.Batch,  # batch-major
    batched_encode_name: tp.Callable[[np.ndarray], np.ndarray],
) -> TwoPlayerBatch:
  p1_game = data_lib.swap_players(batch.frames.state_action.state)
  p1_name_codes = batched_encode_name(batch.meta.info.meta.p1.name)
  unroll_length = batch.frames.state_action.state.stage.shape[1]
  p1_name_codes = np.tile(p1_name_codes[:, None], [1, unroll_length])

  p1_state_action = embed.StateAction(
      state=p1_game,
      action=p1_game.p0.controller,
      name=p1_name_codes,
  )
  p1_frames = data_lib.Frames(
      state_action=p1_state_action,
      reward=-batch.frames.reward,  # assume 0-sum
  )

  return TwoPlayerBatch(
      p0_frames=batch.frames,
      p1_frames=p1_frames,
      needs_reset=batch.needs_reset,
      count=batch.count,
      meta=batch.meta,
  )


class TwoPlayerDataSource:
  def __init__(self, source: data_lib.DataSource):
    self.source = source
    self.batch_size = source.batch_size
    encode_name = nametags.name_encoder(source.name_map)
    self.batched_encode_name = np.vectorize(encode_name)

  def __iter__(self):
    return self

  def __next__(self):
    batch, epoch = next(self.source)
    return convert_batch(batch, self.batched_encode_name), epoch

def make_source(**kwargs) -> TwoPlayerDataSource:
  return TwoPlayerDataSource(data_lib.make_source(**kwargs))
