"""MSL buffer to slippi-ai observation adapters.

This module converts melee-sim-light native buffer views into the
`slippi_ai.types.Game` structures consumed by policies. The fast path writes
into reusable `GameBatch` buffers so rollout can avoid rebuilding Python
observation objects every step.
"""

import typing as tp

import melee
import melee_sim
import numpy as np

from slippi_ai.types import (
    Buttons, Controller, FoDPlatforms, Game, Item, Items, Nana, Player, Randall,
    Stick,
)


_SIM_TO_MELEE_STAGE = {
    int(melee_sim.Stage.FOUNTAIN_OF_DREAMS): melee.Stage.FOUNTAIN_OF_DREAMS.value,
    int(melee_sim.Stage.POKEMON_STADIUM): melee.Stage.POKEMON_STADIUM.value,
    int(melee_sim.Stage.YOSHIS_STORY): melee.Stage.YOSHIS_STORY.value,
    int(melee_sim.Stage.DREAM_LAND_N64): melee.Stage.DREAMLAND.value,
    int(melee_sim.Stage.BATTLEFIELD): melee.Stage.BATTLEFIELD.value,
    int(melee_sim.Stage.FINAL_DESTINATION): melee.Stage.FINAL_DESTINATION.value,
}


class GameBatch(tp.NamedTuple):
  """Policy-facing batch laid out as [all port-1 views, all port-2 views]."""
  game: Game
  needs_reset: np.ndarray


def game_for_port(
    frame: np.ndarray,
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


def copy_controller_slice(dst: Controller, src: Controller, target: slice, source: slice):
  dst.main_stick.x[target] = src.main_stick.x[source]
  dst.main_stick.y[target] = src.main_stick.y[source]
  dst.c_stick.x[target] = src.c_stick.x[source]
  dst.c_stick.y[target] = src.c_stick.y[source]
  dst.shoulder[target] = src.shoulder[source]
  for name in Buttons._fields:
    getattr(dst.buttons, name)[target] = getattr(src.buttons, name)[source]


def player_from_slot(slot: np.ndarray, controller: Controller) -> Player:
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


def items_from_frame(items: np.ndarray) -> Items:
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


def _slots_by_source(slots: np.ndarray) -> dict[int, np.ndarray]:
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


def _empty_nana(batch_size: int) -> Nana:
  zeros_bool = np.zeros(batch_size, dtype=np.bool_)
  zeros_f32 = np.zeros(batch_size, dtype=np.float32)
  zeros_u16 = np.zeros(batch_size, dtype=np.uint16)
  zeros_u8 = np.zeros(batch_size, dtype=np.uint8)
  return Nana(
      exists=zeros_bool.copy(),
      percent=zeros_u16.copy(),
      facing=zeros_bool.copy(),
      x=zeros_f32.copy(),
      y=zeros_f32.copy(),
      action=zeros_u16.copy(),
      invulnerable=zeros_bool.copy(),
      character=zeros_u8.copy(),
      jumps_left=zeros_u8.copy(),
      shield_strength=zeros_f32.copy(),
      on_ground=zeros_bool.copy(),
  )


def _libmelee_jumps_left(slot: np.ndarray) -> np.ndarray:
  raw = np.asarray(slot['jumps_left'], dtype=np.int16)
  airborne_with_ground_jump_available = (np.asarray(slot['on_ground']) == 0) & (raw > 1)
  values = np.where(airborne_with_ground_jump_available, raw - 1, raw)
  return np.maximum(values, 0).astype(np.uint8)


def _canonical_items(items: np.ndarray) -> np.ndarray:
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


class GameBatchBuffers:
  """Reusable Game storage for batched policy calls.

  The policy sees each env twice: first from port 1's perspective, then from
  port 2's perspective. Keeping this storage live lets rollout code fill arrays
  in place instead of allocating a fresh Game nest for every frame.
  """

  def __init__(self, batch_size: int):
    self.batch_size = int(batch_size)
    self.num_players = self.batch_size * 2
    self.needs_reset = np.zeros(self.num_players, dtype=np.bool_)
    self._percent_tmp = np.zeros(self.batch_size, dtype=np.float32)
    self._p0_arrays = _player_arrays(self.num_players)
    self._p1_arrays = _player_arrays(self.num_players)
    self._item_arrays = [
        {
            'exists': np.zeros(self.num_players, dtype=np.bool_),
            'type': np.zeros(self.num_players, dtype=np.uint16),
            'state': np.zeros(self.num_players, dtype=np.uint8),
            'x': np.zeros(self.num_players, dtype=np.float32),
            'y': np.zeros(self.num_players, dtype=np.float32),
        }
        for _ in Items._fields
    ]
    self._items = Items(**{
        f'item_{i}': Item(**arrays)
        for i, arrays in enumerate(self._item_arrays)
    })
    self._p0_controller = _controller_buffers(self.num_players)
    self._p1_controller = _controller_buffers(self.num_players)
    empty_nana = _empty_nana(self.num_players)
    self.game = Game(
        p0=Player(**self._p0_arrays, controller=self._p0_controller, nana=empty_nana),
        p1=Player(**self._p1_arrays, controller=self._p1_controller, nana=empty_nana),
        stage=np.zeros(self.num_players, dtype=np.uint8),
        randall=Randall(
            x=np.zeros(self.num_players, dtype=np.float32),
            y=np.zeros(self.num_players, dtype=np.float32),
        ),
        fod_platforms=FoDPlatforms(
            left=np.zeros(self.num_players, dtype=np.float32),
            right=np.zeros(self.num_players, dtype=np.float32),
        ),
        items=self._items,
    )

  def fill(
      self,
      frame: np.ndarray,
      needs_reset: np.ndarray,
      controllers: tp.Mapping[int, Controller] | None = None,
  ):
    self.fill_slice(frame, needs_reset, slice(0, self.batch_size), controllers)

  def fill_slice(
      self,
      frame: np.ndarray,
      needs_reset: np.ndarray,
      env_slice: slice,
      controllers: tp.Mapping[int, Controller] | None = None,
      controller_slice: slice | None = None,
  ):
    # Each native env contributes two policy examples: port 1 perspective in the
    # first half and port 2 perspective in the second. p0 is the controlled
    # player and p1 is the opponent in both halves.
    first = env_slice
    second = slice(
        self.batch_size + int(env_slice.start or 0),
        self.batch_size + int(env_slice.stop),
    )
    local_batch = int(env_slice.stop) - int(env_slice.start or 0)
    if self._percent_tmp.shape[0] != local_batch:
      self._percent_tmp = np.zeros(local_batch, dtype=np.float32)
    self.needs_reset[first] = needs_reset
    self.needs_reset[second] = needs_reset

    slots_by_source = _slots_by_source(frame['slots'])
    src0 = slots_by_source[0]
    src1 = slots_by_source[1]
    self._fill_player(self._p0_arrays, first, src0)
    self._fill_player(self._p0_arrays, second, src1)
    self._fill_player(self._p1_arrays, first, src1)
    self._fill_player(self._p1_arrays, second, src0)
    if controllers is not None:
      # Controller history is perspective-local: p0 sees its own previous
      # controller and p1 sees the opponent's previous controller.
      source = env_slice if controller_slice is None else controller_slice
      copy_controller_slice(self._p0_controller, controllers[1], first, source)
      copy_controller_slice(self._p0_controller, controllers[2], second, source)
      copy_controller_slice(self._p1_controller, controllers[2], first, source)
      copy_controller_slice(self._p1_controller, controllers[1], second, source)

    self._fill_stage_like(frame, first)
    self._fill_stage_like(frame, second)
    self._fill_items(frame['items'], first)
    self._fill_items(frame['items'], second)

  def _fill_player(self, dst: dict[str, np.ndarray], target: slice, slot: np.ndarray):
    np.clip(slot['percent'], 0, np.iinfo(np.uint16).max, out=self._percent_tmp)
    dst['percent'][target] = self._percent_tmp
    dst['facing'][target] = slot['facing']
    dst['x'][target] = slot['pos_x']
    dst['y'][target] = slot['pos_y']
    dst['action'][target] = slot['action_id']
    dst['invulnerable'][target] = slot['invulnerable']
    dst['character'][target] = slot['char_id']
    dst['jumps_left'][target] = _libmelee_jumps_left(slot)
    dst['shield_strength'][target] = slot['shield_hp']
    dst['on_ground'][target] = slot['on_ground']

  def _fill_stage_like(self, frame: np.ndarray, target: slice):
    stage = self.game.stage[target]
    stage[:] = 0
    for sim_stage, melee_stage in _SIM_TO_MELEE_STAGE.items():
      stage[frame['stage_id'] == sim_stage] = melee_stage
    self.game.randall.x[target] = frame['stage']['randall']['x']
    self.game.randall.y[target] = frame['stage']['randall']['y']
    self.game.fod_platforms.left[target] = 0
    self.game.fod_platforms.right[target] = 0

  def _fill_items(self, items: np.ndarray, target: slice):
    items = _canonical_items(items)
    for i, arrays in enumerate(self._item_arrays):
      src = items[:, i]
      arrays['exists'][target] = src['exists']
      arrays['type'][target] = src['type']
      arrays['state'][target] = src['state']
      arrays['x'][target] = src['pos_x']
      arrays['y'][target] = src['pos_y']


def _player_arrays(batch_size: int) -> dict[str, np.ndarray]:
  arrays = {
      'percent': np.zeros(batch_size, dtype=np.uint16),
      'facing': np.zeros(batch_size, dtype=np.bool_),
      'x': np.zeros(batch_size, dtype=np.float32),
      'y': np.zeros(batch_size, dtype=np.float32),
      'action': np.zeros(batch_size, dtype=np.uint16),
      'invulnerable': np.zeros(batch_size, dtype=np.bool_),
      'character': np.zeros(batch_size, dtype=np.uint8),
      'jumps_left': np.zeros(batch_size, dtype=np.uint8),
      'shield_strength': np.zeros(batch_size, dtype=np.float32),
      'on_ground': np.zeros(batch_size, dtype=np.bool_),
  }
  return arrays


def _controller_buffers(batch_size: int) -> Controller:
  controller = Controller(
      main_stick=Stick(
          x=np.zeros(batch_size, dtype=np.float32),
          y=np.zeros(batch_size, dtype=np.float32),
      ),
      c_stick=Stick(
          x=np.zeros(batch_size, dtype=np.float32),
          y=np.zeros(batch_size, dtype=np.float32),
      ),
      shoulder=np.zeros(batch_size, dtype=np.float32),
      buttons=Buttons(**{
          name: np.zeros(batch_size, dtype=np.bool_)
          for name in Buttons._fields
      }),
  )
  controller.main_stick.x[:] = 0.5
  controller.main_stick.y[:] = 0.5
  controller.c_stick.x[:] = 0.5
  controller.c_stick.y[:] = 0.5
  return controller
