# TODO: maybe just merge this in with data.py

import typing as tp

import numpy as np

from slippi_ai import nametags, utils
from slippi_ai import data as data_lib
from slippi_ai.types import S
from slippi_ai import types

T = tp.TypeVar('T')
Rank2 = tuple[int, int]
Rank3 = tuple[int, int, int]
ZippedFrames = types.Frames[Rank3, types.Controller[Rank3]]

class DoubleBatch(tp.NamedTuple):
  p0_batch: data_lib.Batch[Rank2]
  p1_batch: data_lib.Batch[Rank2]

  @classmethod
  def from_batch_with_meta(
      cls,
      batch_with_meta: data_lib.BatchWithMeta[Rank2],
      encode_name: tp.Callable[[str], types.Int32Array[data_lib.Rank1]],
  ) -> tp.Self:
    batch = batch_with_meta.batch
    p1_game = data_lib.swap_players(batch.game)
    # Note: the name data is a numpy array of strings with shape [B]
    p1_name_code = encode_name(batch_with_meta.meta.meta.p1.name)  # [B]
    full_p1_name_code = np.broadcast_to(
        p1_name_code[:, None], batch.name.shape)  # [B, T]

    p1_batch = data_lib.Batch[Rank2](
        game=p1_game,
        name=full_p1_name_code,
        is_resetting=batch.is_resetting,
        reward=-batch.reward,  # assume 0-sum
    )

    return cls(
        p0_batch=batch,
        p1_batch=p1_batch,
    )

  def to_frames(self, frame_skip: int, discount: float = 1.0) -> ZippedFrames:
    p0_frames = self.p0_batch.to_frames(frame_skip, discount=discount)
    p1_frames = self.p1_batch.to_frames(frame_skip, discount=discount)

    zipped_frames = utils.map_nt(
        lambda *xs: np.stack(xs, axis=1),
        p0_frames, p1_frames,
    )
    return tp.cast(ZippedFrames, zipped_frames)

class TwoPlayerBatch(tp.NamedTuple, tp.Generic[S]):
  p0_frames: types.Frames[S, types.Controller]
  p1_frames: types.Frames[S, types.Controller]
  # is_resetting: types.BoolArray[S]
  meta: data_lib.ChunkMeta

def batch_to_frames(
    batch: TwoPlayerBatch[data_lib.Rank2],
) -> ZippedFrames:
  """Convert [B, T] batch to [B, 2, T] frames."""
  zipped_frames = utils.map_nt(
      lambda *xs: np.stack(xs, axis=1),
      batch.p0_frames, batch.p1_frames,
  )
  return tp.cast(ZippedFrames, zipped_frames)


def convert_batch(
    batch_with_meta: data_lib.BatchWithMeta[data_lib.Rank2],  # batch-major
    encode_name: tp.Callable[[str], types.Int32Array[data_lib.Rank1]],
    frame_skip: int = 1,
    discount: float = 1.0,
) -> TwoPlayerBatch[data_lib.Rank2]:
  batch = batch_with_meta.batch
  p1_game = data_lib.swap_players(batch.game)
  # Note: the name data is a numpy array of strings with shape [B]
  p1_name_code = encode_name(batch_with_meta.meta.meta.p1.name)  # [B]

  full_p1_name_code = np.broadcast_to(
    p1_name_code[:, None], batch.name.shape)  # [B, T]

  p1_batch = data_lib.Batch(
      game=p1_game,
      name=full_p1_name_code,
      is_resetting=batch.is_resetting,
      reward=-batch.reward,  # assume 0-sum
  )

  p0_frames = batch.to_frames(frame_skip, discount=discount)
  p1_frames = p1_batch.to_frames(frame_skip, discount=discount)

  return TwoPlayerBatch(
      p0_frames=p0_frames,
      p1_frames=p1_frames,
      meta=batch_with_meta.meta,
  )


class TwoPlayerDataSource:
  def __init__(
      self,
      source: data_lib.AbstractDataSource,
      name_map: dict[str, int],
      frame_skip: int = 1,
      # TODO: switch to DoubleBatch and defer to_frames to the learners
      discount: float = 1.0,
  ):
    self.source = source
    self.batch_size = source.batch_size
    self.frame_skip = frame_skip
    self.discount = discount
    self.encode_name = nametags.name_encoder(name_map)
    self.batched_encode_name = np.vectorize(self.encode_name)

  def __iter__(self):
    return self

  def __next__(self):
    batch, epoch = next(self.source)
    return convert_batch(batch, self.batched_encode_name, self.frame_skip, self.discount), epoch

  def shutdown(self):
    self.source.shutdown()
