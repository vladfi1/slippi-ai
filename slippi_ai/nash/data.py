# TODO: maybe just merge this in with data.py

import typing as tp

import numpy as np

from slippi_ai import nametags
from slippi_ai import data as data_lib
from slippi_ai.types import S
from slippi_ai import types

T = tp.TypeVar('T')


class TwoPlayerBatch(tp.NamedTuple, tp.Generic[S]):
  p0_frames: types.Frames[S, types.Controller]
  p1_frames: types.Frames[S, types.Controller]
  # is_resetting: types.BoolArray[S]
  meta: data_lib.ChunkMeta

def convert_batch(
    batch: data_lib.Batch[data_lib.Rank2],  # batch-major
    encode_name: tp.Callable[[str], types.Int32Array[data_lib.Rank1]],
) -> TwoPlayerBatch[data_lib.Rank2]:
  p1_game = data_lib.swap_players(batch.game)
  # Note: the name data is a numpy array of strings with shaep [B]
  p1_name_code = encode_name(batch.meta.info.meta.p1.name)  # [B]

  full_p1_name_code = np.broadcast_to(
    p1_name_code[:, None], batch.name.shape)  # [B, T]

  p1_frames = types.Frames(
      state_action=types.StateAction(
          state=p1_game,
          action=p1_game.p0.controller,
          name=full_p1_name_code,
      ),
      is_resetting=batch.is_resetting,
      reward=-batch.reward,  # assume 0-sum
  )

  p0_frames = types.Frames(
      state_action=types.StateAction(
          state=batch.game,
          action=batch.game.p0.controller,
          name=batch.name,
      ),
      is_resetting=batch.is_resetting,
      reward=batch.reward,
  )

  return TwoPlayerBatch(
      p0_frames=p0_frames,
      p1_frames=p1_frames,
      # is_resetting=batch.is_resetting,
      meta=batch.meta,
  )


class TwoPlayerDataSource:
  def __init__(self, source: data_lib.AbstractDataSource, name_map: dict[str, int]):
    self.source = source
    self.batch_size = source.batch_size
    self.encode_name = nametags.name_encoder(name_map)
    self.batched_encode_name = np.vectorize(self.encode_name)

  def __iter__(self):
    return self

  def __next__(self):
    batch, epoch = next(self.source)
    return convert_batch(batch, self.batched_encode_name), epoch
