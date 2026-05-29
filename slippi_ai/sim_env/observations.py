"""MSL buffer to slippi-ai observation adapters.

This module converts melee-sim-light native buffer views into the
`slippi_ai.types.Game` structures consumed by policies. The fast path writes
into reusable `GameBatch` buffers so rollout can avoid rebuilding Python
observation objects every step.
"""

import typing as tp

import jax

import melee
import melee_sim
import numpy as np

from slippi_ai import utils

from slippi_ai.types import (
    Buttons, Controller, FoDPlatforms, Game, Item, Items, Nana, Player, Randall,
    reify_tuple_type
)


_SIM_TO_MELEE_STAGE = {
    int(melee_sim.Stage.FOUNTAIN_OF_DREAMS): melee.Stage.FOUNTAIN_OF_DREAMS.value,
    int(melee_sim.Stage.POKEMON_STADIUM): melee.Stage.POKEMON_STADIUM.value,
    int(melee_sim.Stage.YOSHIS_STORY): melee.Stage.YOSHIS_STORY.value,
    int(melee_sim.Stage.DREAM_LAND_N64): melee.Stage.DREAMLAND.value,
    int(melee_sim.Stage.BATTLEFIELD): melee.Stage.BATTLEFIELD.value,
    int(melee_sim.Stage.FINAL_DESTINATION): melee.Stage.FINAL_DESTINATION.value,
}

# melee_sim exposes native C buffers as NumPy structured-array views. Field
# access like frame["slots"] or slot["percent"] selects dtype field views; these
# are not dicts.
# TODO: add TypedDict signatures for these
FrameView = np.ndarray
SlotsView = np.ndarray
PlayerSlotView = np.ndarray
ItemsView = np.ndarray
ArrayAllocator = tp.Callable[[tuple[int, ...], tp.Any], np.ndarray]

S = tp.TypeVar('S', bound=tuple[int, ...])
Rank1 = tuple[int]

class PlayerArrays(tp.TypedDict, tp.Generic[S]):
  percent: np.ndarray[S, np.dtype[np.uint16]]
  facing: np.ndarray[S, np.dtype[np.bool_]]
  x: np.ndarray[S, np.dtype[np.float32]]
  y: np.ndarray[S, np.dtype[np.float32]]
  action: np.ndarray[S, np.dtype[np.uint16]]
  invulnerable: np.ndarray[S, np.dtype[np.bool_]]
  character: np.ndarray[S, np.dtype[np.uint8]]
  jumps_left: np.ndarray[S, np.dtype[np.uint8]]
  shield_strength: np.ndarray[S, np.dtype[np.float32]]
  on_ground: np.ndarray[S, np.dtype[np.bool_]]


class GameBatch(tp.NamedTuple, tp.Generic[S]):
  """Policy-facing batch laid out as [all port-1 views, all port-2 views]."""
  game: Game[S]
  needs_reset: np.ndarray[S, np.dtype[np.bool_]]


def game_for_port(
    frame: FrameView,
    port: int,
    controllers: tp.Mapping[int, Controller],
) -> Game:
  """Build the ordinary port-keyed Game view used by EnvOutput."""
  slots_by_source = _slots_by_source(frame['slots'])
  self_source = port - 1
  opponent_source = 1 - self_source
  opponent_port = 1 if port == 2 else 2
  batch_size = frame.shape[0]
  stage = np.zeros(frame['stage_id'].shape, dtype=np.uint8)
  for sim_stage, melee_stage in _SIM_TO_MELEE_STAGE.items():
    stage[frame['stage_id'] == sim_stage] = melee_stage
  return Game(
      p0=player_from_slot(slots_by_source[self_source], controllers[port]),
      p1=player_from_slot(slots_by_source[opponent_source], controllers[opponent_port]),
      stage=stage,
      randall=Randall(
          x=frame['stage']['randall']['x'].astype(np.float32, copy=True),
          y=frame['stage']['randall']['y'].astype(np.float32, copy=True),
      ),
      fod_platforms=FoDPlatforms(
          left=np.zeros(batch_size, dtype=np.float32),
          right=np.zeros(batch_size, dtype=np.float32),
      ),
      items=items_from_frame(frame['items']),
  )


def copy_controller_slice(dst: Controller, src: Controller, target: slice):
  dst.main_stick.x[target] = src.main_stick.x
  dst.main_stick.y[target] = src.main_stick.y
  dst.c_stick.x[target] = src.c_stick.x
  dst.c_stick.y[target] = src.c_stick.y
  dst.shoulder[target] = src.shoulder
  for name in Buttons._fields:
    getattr(dst.buttons, name)[target] = getattr(src.buttons, name)


def player_from_slot(slot: PlayerSlotView, controller: Controller) -> Player:
  """Convert one melee_sim source-player slot into a policy-facing Player."""
  return Player(
      percent=slot['percent'].clip(0, np.iinfo(np.uint16).max).astype(np.uint16),
      facing=slot['facing'].astype(np.bool_, copy=True),
      x=slot['pos_x'].astype(np.float32, copy=True),
      y=slot['pos_y'].astype(np.float32, copy=True),
      action=slot['action_id'].astype(np.uint16, copy=True),
      invulnerable=slot['invulnerable'].astype(np.bool_, copy=True),
      character=slot['char_id'].astype(np.uint8, copy=True),
      jumps_left=_libmelee_jumps_left(slot),
      shield_strength=slot['shield_hp'].astype(np.float32, copy=True),
      on_ground=slot['on_ground'].astype(np.bool_, copy=True),
      controller=controller,
      nana=_empty_nana(slot.shape[0]),
  )


def items_from_frame(items: ItemsView) -> Items:
  """Convert melee_sim item slots into the fixed policy-facing item nest."""
  items = _canonical_items(items)
  return Items(**{
      f'item_{i}': Item(
          exists=items[:, i]['exists'].astype(np.bool_, copy=True),
          type=items[:, i]['type'].astype(np.uint16, copy=True),
          state=items[:, i]['state'].astype(np.uint8, copy=True),
          x=items[:, i]['pos_x'].astype(np.float32, copy=True),
          y=items[:, i]['pos_y'].astype(np.float32, copy=True),
      )
      for i in range(len(Items._fields))
  })


def _slots_by_source(slots: SlotsView) -> dict[int, PlayerSlotView]:
  result = {}
  for i in range(slots.shape[1]):
    if not np.any(slots[:, i]['present']):
      continue
    source = slots[:, i]['source_player']
    if np.all(source == source[0]):
      result[int(source[0])] = slots[:, i]
  if 0 not in result or 1 not in result:
    raise RuntimeError('melee_sim gamestate did not contain source players 0 and 1')
  return result


def _empty_nana(
    batch_size: int,
    allocate_array: ArrayAllocator | None = None,
) -> Nana:
  if allocate_array is None:
    allocate_array = lambda shape, dtype: np.zeros(shape, dtype=dtype)
  shape = (batch_size,)

  def zeros(dtype):
    array = allocate_array(shape, dtype)
    array[...] = 0
    return array

  return Nana(
      exists=zeros(np.bool_),
      percent=zeros(np.uint16),
      facing=zeros(np.bool_),
      x=zeros(np.float32),
      y=zeros(np.float32),
      action=zeros(np.uint16),
      invulnerable=zeros(np.bool_),
      character=zeros(np.uint8),
      jumps_left=zeros(np.uint8),
      shield_strength=zeros(np.float32),
      on_ground=zeros(np.bool_),
  )


def _libmelee_jumps_left(slot: PlayerSlotView) -> np.ndarray:
  raw = np.asarray(slot['jumps_left'], dtype=np.int16)
  airborne_with_ground_jump_available = (np.asarray(slot['on_ground']) == 0) & (raw > 1)
  values = np.where(airborne_with_ground_jump_available, raw - 1, raw)
  return np.maximum(values, 0).astype(np.uint8)


def _canonical_items(items: ItemsView) -> ItemsView:
  # Native item slots are storage slots. Sort into a stable policy-facing order
  # so observations do not depend on item allocator history.
  exists_key = -items['exists'].astype(np.int16)
  type_key = -items['type'].astype(np.int32)
  index_key = np.broadcast_to(
      np.arange(items.shape[1], dtype=np.int16),
      items.shape,
  )
  order = np.lexsort((index_key, type_key, exists_key), axis=1)
  return np.take_along_axis(items, order, axis=1)


class GameBatchBuffer:
  """Reusable Game storage for batched policy calls.

  The policy sees each env twice: first from port 1's perspective, then from
  port 2's perspective. Keeping this storage live lets rollout code fill arrays
  in place instead of allocating a fresh Game nest for every frame.
  """

  def __init__(
      self,
      game_batch: GameBatch[Rank1],
  ):
    leaves: list[np.ndarray] = jax.tree.leaves(game_batch)
    (num_players,) = leaves[0].shape
    for leaf in leaves:
      if leaf.shape != leaves[0].shape:
        raise ValueError('game_batch must have uniform shape')

    self.game_batch = game_batch
    self.game = game_batch.game
    self.needs_reset = game_batch.needs_reset
    self._item_arrays = self.game.items

    self.num_players = num_players
    self.num_envs, r = divmod(num_players, 2)
    assert r == 0

  def fill(
      self,
      frame: FrameView,
      needs_reset: np.ndarray,
      controllers: tp.Mapping[int, Controller] | None = None,
  ):
    self.fill_slice(frame, needs_reset, slice(0, self.num_envs), controllers)

  def fill_slice(
      self,
      frame: FrameView,
      needs_reset: np.ndarray,
      env_slice: slice,
      controllers: tp.Mapping[int, Controller] | None = None,
  ):
    # TODO: we should try to reuse the the native env's own time-major buffers

    # Each native env contributes two policy examples: port 1 perspective in the
    # first half and port 2 perspective in the second. p0 is the controlled
    # player and p1 is the opponent in both halves.
    first = env_slice
    start = env_slice.start
    assert isinstance(start, int)
    stop = env_slice.stop
    assert isinstance(stop, int)
    second = slice(self.num_envs + start, self.num_envs + stop)
    self.needs_reset[first] = needs_reset
    self.needs_reset[second] = needs_reset

    slots_by_source = _slots_by_source(frame['slots'])
    src0 = slots_by_source[0]
    src1 = slots_by_source[1]
    self._fill_player(self.game.p0, first, src0)
    self._fill_player(self.game.p0, second, src1)
    self._fill_player(self.game.p1, first, src1)
    self._fill_player(self.game.p1, second, src0)
    if controllers is not None:
      # Controller history is perspective-local: p0 sees its own previous
      # controller and p1 sees the opponent's previous controller.
      copy_controller_slice(self.game.p0.controller, controllers[1], first)
      copy_controller_slice(self.game.p0.controller, controllers[2], second)
      copy_controller_slice(self.game.p1.controller, controllers[2], first)
      copy_controller_slice(self.game.p1.controller, controllers[1], second)

    self._fill_stage_like(frame, first)
    self._fill_stage_like(frame, second)
    self._fill_items(frame['items'], first)
    self._fill_items(frame['items'], second)

  def _fill_player(self, dst: Player, target: slice, slot: PlayerSlotView):
    # percent_tmp = self._percent_tmp[:slot.shape[0]]
    np.clip(slot['percent'], 0, np.iinfo(np.uint16).max, out=dst.percent[target], casting='unsafe')
    # dst['percent'][target] = percent_tmp
    dst.facing[target] = slot['facing']
    dst.x[target] = slot['pos_x']
    dst.y[target] = slot['pos_y']
    dst.action[target] = slot['action_id']
    dst.invulnerable[target] = slot['invulnerable']
    dst.character[target] = slot['char_id']
    dst.jumps_left[target] = _libmelee_jumps_left(slot)
    dst.shield_strength[target] = slot['shield_hp']
    dst.on_ground[target] = slot['on_ground']

  def _fill_stage_like(self, frame: FrameView, target: slice):
    stage = self.game.stage[target]
    stage[:] = 0
    for sim_stage, melee_stage in _SIM_TO_MELEE_STAGE.items():
      stage[frame['stage_id'] == sim_stage] = melee_stage
    self.game.randall.x[target] = frame['stage']['randall']['x']
    self.game.randall.y[target] = frame['stage']['randall']['y']
    self.game.fod_platforms.left[target] = 0
    self.game.fod_platforms.right[target] = 0

  def _fill_items(self, items: ItemsView, target: slice):
    items = _canonical_items(items)
    for i, arrays in enumerate(self._item_arrays):
      src = items[:, i]
      arrays.exists[target] = src['exists']
      arrays.type[target] = src['type']
      arrays.state[target] = src['state']
      arrays.x[target] = src['pos_x']
      arrays.y[target] = src['pos_y']


T = tp.TypeVar('T')

class TrajectoryBuffer(tp.NamedTuple, tp.Generic[T]):
  """Time-major storage for a structure.

  Slots are numpy views into the contiguous memory region.
  """

  time_major: T
  slots: list[T]  # T+1 views into states

  def __len__(self):
    return len(self.slots)

def trajectory_buffer_from_struct(struct: T) -> TrajectoryBuffer[T]:
  """Initialize from an existing time-major struct."""

  leaves = jax.tree.leaves(struct)
  first = leaves[0]
  assert isinstance(first, np.ndarray)

  for leaf in leaves:
    assert isinstance(leaf, np.ndarray)
    assert leaf.shape == first.shape

  slots = [
      utils.map_nt(lambda leaf, i=i: leaf[i], struct)
      for i in range(first.shape[0])
  ]

  return TrajectoryBuffer(struct, slots)

def build_trajectory_buffer(
    type_: type[T],
    batch_size: int,
    rollout_length: int,
) -> TrajectoryBuffer[T]:
  struct = utils.map_single_structure(
      lambda dtype: np.empty((rollout_length, batch_size), dtype=dtype),
      reify_tuple_type(type_),
  )
  return trajectory_buffer_from_struct(struct)
