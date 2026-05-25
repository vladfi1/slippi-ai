"""Single-process melee-sim-light environment adapter.

This module is the direct bridge from `melee_sim.EnvBatch` native buffers to the
slippi-ai `Game`/`Controller` structures used by policies. It owns the Python
API for a batched sim env, including stage and character-pool setup, reset
handling, terminal side-channel views, and conversion from native SoA buffers to
the libmelee-shaped observation nest.

There are two use modes. `current_state`/`step` preserve the familiar
port-keyed Dolphin env interface for tests and small tools. The high-throughput
path uses `write_current_game` and `step_encoded`: one reusable `GameBatch`
stores all port-1 perspectives followed by all port-2 perspectives, and policy
action buckets are decoded straight into the native action ring. That path avoids
rebuilding Python game objects during rollout and is what the JAX sim pipeline
uses.
"""

import collections
import contextlib
import itertools
import typing as tp

import melee
import melee_sim
import numpy as np

from slippi_ai import dolphin
from slippi_ai.envs import EnvOutput
from slippi_ai.sim_env.observations import (
    GameBatchBuffers, copy_controller_slice as _copy_controller_slice,
    game_for_port,
)
from slippi_ai.types import Buttons, Controller, Stick


Port = int
Controllers = tp.Mapping[Port, Controller]

_SUPPORTED_PORTS = (1, 2)
_MELEE_TO_SIM_STAGE = {
    melee.Stage.FOUNTAIN_OF_DREAMS: melee_sim.Stage.FOUNTAIN_OF_DREAMS,
    melee.Stage.POKEMON_STADIUM: melee_sim.Stage.POKEMON_STADIUM,
    melee.Stage.YOSHIS_STORY: melee_sim.Stage.YOSHIS_STORY,
    melee.Stage.DREAMLAND: melee_sim.Stage.DREAM_LAND_N64,
    melee.Stage.BATTLEFIELD: melee_sim.Stage.BATTLEFIELD,
    melee.Stage.FINAL_DESTINATION: melee_sim.Stage.FINAL_DESTINATION,
}
SUPPORTED_STAGES = tuple(_MELEE_TO_SIM_STAGE)
SUPPORTED_CHARACTERS = (
    melee.Character.FOX,
    melee.Character.FALCO,
)
_CHARACTER_BY_NAME = {
    character.name.lower(): character for character in SUPPORTED_CHARACTERS
}

class SimStepInfo(tp.NamedTuple):
  terminal: np.ndarray
  step_t: int


PlayerConfigs = tp.Mapping[int, dolphin.Player]
CharacterPool = str | tp.Sequence[melee.Character | str | int]


class SimBatchedEnvironment:
  """Batched melee_sim-backed environment using slippi-ai Game objects.

  The public env shape mirrors the Dolphin-backed envs: callers pass controller
  actions and receive `EnvOutput` objects. High-throughput paths should use
  `write_current_game` and `step_encoded` to avoid rebuilding per-port Python
  objects while still feeding the same policy-facing fields.
  """

  def __init__(
      self,
      num_envs: int,
      players: PlayerConfigs | tp.Sequence[PlayerConfigs] | None = None,
      *,
      frame_buffer_length: int = 128,
      stage: melee.Stage | tp.Sequence[melee.Stage] = melee.Stage.FINAL_DESTINATION,
      character_pool: CharacterPool | None = None,
      max_frame_id: int = -1,
      data_dir: str | None = None,
      fake: bool = False,
  ):
    self._num_envs = int(num_envs)
    self._frame_buffer_length = int(frame_buffer_length)
    self._max_frame_id = int(max_frame_id)
    self._fake = fake
    self.num_steps = 1

    # The sim adapter currently exposes the singles shape used by the policy:
    # port 1 and port 2, no teams, one controlled character per port.
    if players is None:
      default_players = {
          1: dolphin.AI(melee.Character.FOX),
          2: dolphin.AI(melee.Character.FOX),
      }
      self._players_by_env = (default_players,) * self._num_envs
    elif isinstance(players, collections.abc.Mapping):
      self._players_by_env = (players,) * self._num_envs
    else:
      self._players_by_env = tuple(players)
      if len(self._players_by_env) != self._num_envs:
        raise ValueError(f'players sequence must have length num_envs={self._num_envs}')
    self._players = self._players_by_env[0]
    self._ports = tuple(sorted(self._players))
    for player_map in self._players_by_env:
      if tuple(sorted(player_map)) != _SUPPORTED_PORTS:
        raise ValueError('SimBatchedEnvironment currently supports ports 1 and 2.')

    # `stage` can be one value for every lane or an explicit per-env list.
    if isinstance(stage, melee.Stage):
      stages = [stage] * self._num_envs
    else:
      stages = list(stage)
      if len(stages) != self._num_envs:
        raise ValueError(f'stage sequence must have length num_envs={self._num_envs}')
    for item in stages:
      if item not in SUPPORTED_STAGES:
        raise ValueError(f'SimBatchedEnvironment currently supports {SUPPORTED_STAGES}.')
    self._stage_by_env = np.asarray(stages, dtype=object)

    # Default to the explicit player matchup for each lane. If a pool is
    # supplied, cycle lanes through the ordered singles matchup matrix. Resets
    # keep each lane's initial matchup until we intentionally add randomized
    # reset-time assignment later.
    if character_pool is None:
      character_assignments = tuple(
          (
              _coerce_character(player_map[1].character),
              _coerce_character(player_map[2].character),
          )
          for player_map in self._players_by_env
      )
    else:
      if isinstance(character_pool, str):
        characters = tuple(
            _coerce_character(name.strip())
            for name in character_pool.split(',')
            if name.strip())
      else:
        characters = tuple(_coerce_character(character) for character in character_pool)
      pool = tuple(dict.fromkeys(characters))
      if not pool:
        raise ValueError('character pool must not be empty')
      unsupported = tuple(
          character for character in pool if character not in SUPPORTED_CHARACTERS)
      if unsupported:
        raise ValueError(
            f'unsupported sim character pool {pool!r}; '
            f'supported characters are {SUPPORTED_CHARACTERS}')
      matrix = tuple(itertools.product(pool, repeat=2))
      character_assignments = tuple(
          matrix[i % len(matrix)] for i in range(self._num_envs))

    match_configs = [
        melee_sim.MatchConfig(
            stage=_MELEE_TO_SIM_STAGE[stage],
            players=(
                melee_sim.PlayerConfig(character=int(char_pair[0].value)),
                melee_sim.PlayerConfig(character=int(char_pair[1].value)),
            ),
        )
        for stage, char_pair in zip(self._stage_by_env, character_assignments)
    ]

    # EnvBatch owns and binds its native rollout ring. Python keeps views into
    # that storage and only writes controller actions / reads observations at
    # the current cursor.
    self._env = melee_sim.EnvBatch(
        batch_size=self._num_envs,
        length=self._frame_buffer_length,
        num_players=2,
        data_dir=data_dir,
    )
    self._env.configure_matches(match_configs)
    self._env.reset_all()

    # Previous-controller observations are policy-visible, so they live beside
    # the native env and are reset for lanes as games finish.
    self._last_controllers = {
        port: neutral_controllers(self._num_envs) for port in self._ports
    }

    self._last_step_info = SimStepInfo(
        terminal=np.zeros(self._num_envs, dtype=melee_sim.terminal_dtype()),
        step_t=-1,
    )
    self._episode_ids = np.zeros(self._num_envs, dtype=np.int64)
    self._completed_games: list[dict[str, tp.Any]] = []

    # Reusable policy-facing observation tree for the high-throughput path.
    self._game_batch = GameBatchBuffers(self._num_envs)

    # Match the existing env push/pop contract by seeding an initial observation.
    self._output_queue = collections.deque([
        self.current_state(needs_reset=np.ones(self._num_envs, dtype=np.bool_))
    ])

  def stop(self):
    self._env.close()

  @contextlib.contextmanager
  def run(self):
    try:
      yield self
    finally:
      self.stop()

  def current_state(self, needs_reset: np.ndarray | None = None) -> EnvOutput:
    needs_reset = np.zeros(self._num_envs, dtype=np.bool_) if needs_reset is None else needs_reset
    return EnvOutput(
        gamestates={
            port: game_for_port(
                self._env.current_frame, port, self._last_controllers)
            for port in self._ports
        },
        needs_reset=np.asarray(needs_reset, dtype=np.bool_),
    )

  @property
  def game_batch_buffers(self) -> GameBatchBuffers:
    return self._game_batch

  def write_current_game(
      self,
      game_batch: GameBatchBuffers,
      needs_reset: np.ndarray | None = None,
  ):
    """Write the current sim observation into reusable policy input buffers."""
    needs_reset = np.zeros(self._num_envs, dtype=np.bool_) if needs_reset is None else needs_reset
    game_batch.fill(
        self._env.current_frame, needs_reset, self._last_controllers)

  def reset(self, env_ids: tp.Sequence[int] | np.ndarray | None = None) -> EnvOutput:
    ids = np.arange(self._num_envs, dtype=np.int64) if env_ids is None else np.asarray(env_ids, dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= self._num_envs):
      raise ValueError('env_ids contains an out-of-range env index')
    self._ensure_cursor_room()
    reset_mask = self._env.current_reset_mask
    reset_mask[:] = 0
    reset_mask[ids] = 1
    self._env.reset_masked()
    reset_mask[ids] = 0
    self._episode_ids[ids] += 1
    if ids.size:
      neutral = neutral_controllers(ids.size)
      for port in self._ports:
        _copy_controller_slice(
            self._last_controllers[port], neutral, ids, slice(None))
    needs_reset = np.zeros(self._num_envs, dtype=np.bool_)
    needs_reset[ids] = True
    return self.current_state(needs_reset=needs_reset)

  def push(self, controllers: Controllers):
    self._output_queue.append(self._advance(controllers))

  def pop(self) -> EnvOutput:
    return self._output_queue.popleft()

  def peek(self) -> EnvOutput:
    return self._output_queue[0]

  def step(self, controllers: Controllers) -> EnvOutput:
    return self._advance(controllers)

  def step_encoded(
      self,
      controller_state: Controller,
      *,
      axis_spacing: int,
      shoulder_spacing: int,
  ) -> np.ndarray:
    """Step from encoded default-controller buckets shaped by port perspective."""
    return self.step_encoded_slices(
        controller_state,
        player_slices=(
            slice(0, self._num_envs),
            slice(self._num_envs, 2 * self._num_envs),
        ),
        axis_spacing=axis_spacing,
        shoulder_spacing=shoulder_spacing,
    )

  # The JAX rollout worker stores both port perspectives in one controller
  # batch. Multiprocess workers own contiguous env shards, so they pass the
  # source slices for port 1 and port 2 explicitly instead of assuming the
  # default [all p1 views, all p2 views] layout.
  def step_encoded_slices(
      self,
      controller_state: Controller,
      *,
      player_slices: tuple[slice, slice],
      axis_spacing: int,
      shoulder_spacing: int,
  ) -> np.ndarray:
    """Step from encoded controller buckets with explicit source slices."""
    self._ensure_cursor_room()

    action = self._env.current_action_frame
    # Decode policy buckets straight into the native action ring, and mirror the
    # same decoded values into previous-controller state for the next Game view.
    for player_index, port in enumerate(self._ports):
      write_encoded_controller_action(
          action,
          controller_state,
          player_index=player_index,
          source_slice=player_slices[player_index],
          axis_spacing=axis_spacing,
          shoulder_spacing=shoulder_spacing,
      )
      copy_encoded_controller(
          self._last_controllers[port],
          controller_state,
          source_slice=player_slices[player_index],
          axis_spacing=axis_spacing,
          shoulder_spacing=shoulder_spacing,
      )
    if self._fake:
      return np.zeros(self._num_envs, dtype=np.bool_)

    step_t = self._env.t
    self._env.step(max_frame_id=self._max_frame_id)
    needs_reset = self._env.done_at(step_t).astype(np.bool_, copy=True)
    terminal = self._env.terminal_at(step_t).copy()
    self._last_step_info = SimStepInfo(terminal=terminal, step_t=step_t)
    self._record_completed_games(terminal, self._env.current_frame)
    self._reset_finished_lanes_for_next_observation(needs_reset)
    return needs_reset

  def multi_step(self, controllers: list[Controllers]) -> list[EnvOutput]:
    return [self.step(c) for c in controllers]

  @property
  def buffers(self):
    return self._env.buffers

  @property
  def cursor(self) -> int:
    return self._env.t

  @property
  def stages(self) -> np.ndarray:
    return self._stage_by_env.copy()

  @property
  def last_step_info(self) -> SimStepInfo:
    return self._last_step_info

  def pop_completed_games(self) -> list[dict[str, tp.Any]]:
    completed_games = self._completed_games
    self._completed_games = []
    return completed_games

  def active_games(self) -> list[dict[str, int | str]]:
    return [
        {
            'env_id': env_id,
            'episode_id': int(self._episode_ids[env_id]),
            'stage': self._stage_by_env[env_id].name,
            'stage_id': int(self._stage_by_env[env_id].value),
        }
        for env_id in range(self._num_envs)
    ]

  def _advance(self, controllers: Controllers) -> EnvOutput:
    self._ensure_cursor_room()

    action = self._env.current_action_frame
    for player_index, port in enumerate(self._ports):
      controller = controllers[port]
      player = action['p'][:, int(player_index)]
      player['main_stick_x'][:] = controller.main_stick.x
      player['main_stick_y'][:] = controller.main_stick.y
      player['c_stick_x'][:] = controller.c_stick.x
      player['c_stick_y'][:] = controller.c_stick.y
      player['shoulder'][:] = controller.shoulder
      for name in Buttons._fields:
        player['buttons'][name][:] = getattr(controller.buttons, name)
      _copy_controller_slice(
          self._last_controllers[port],
          controller,
          slice(None),
          slice(None),
      )
    if self._fake:
      return self.current_state(needs_reset=np.zeros(self._num_envs, dtype=np.bool_))

    step_t = self._env.t
    self._env.step(max_frame_id=self._max_frame_id)
    needs_reset = self._env.done_at(step_t).astype(np.bool_, copy=True)
    terminal = self._env.terminal_at(step_t).copy()
    self._last_step_info = SimStepInfo(terminal=terminal, step_t=step_t)
    self._record_completed_games(terminal, self._env.current_frame)
    self._reset_finished_lanes_for_next_observation(needs_reset)
    return self.current_state(needs_reset=needs_reset)

  def _record_completed_games(self, terminal: np.ndarray, frame: np.ndarray):
    done_ids = np.flatnonzero(terminal['done'])
    if done_ids.size == 0:
      return
    slot0 = _slot_for_source(frame['slots'], 0)
    slot1 = _slot_for_source(frame['slots'], 1)
    for env_id in done_ids:
      env_id = int(env_id)
      env_slot0 = slot0[env_id]
      env_slot1 = slot1[env_id]
      stocks = (int(env_slot0['stocks']), int(env_slot1['stocks']))
      percents = (float(env_slot0['percent']), float(env_slot1['percent']))
      if stocks[0] != stocks[1]:
        winner_port = 1 if stocks[0] > stocks[1] else 2
      elif percents[0] != percents[1]:
        winner_port = 1 if percents[0] < percents[1] else 2
      else:
        winner_port = None
      terminal_row = terminal[env_id]
      stage = self._stage_by_env[env_id]
      self._completed_games.append({
          'env_id': env_id,
          'episode_id': int(self._episode_ids[env_id]),
          'stage': stage.name,
          'stage_id': int(stage.value),
          'frame_id': int(terminal_row['frame_id']),
          'frames': max(0, int(terminal_row['frame_id']) + 123),
          'winner_port': winner_port,
          'stocks': stocks,
          'percents': percents,
          'match_ended': bool(terminal_row['match_ended']),
          'stockout': bool(terminal_row['stockout']),
          'max_frame_reached': bool(terminal_row['max_frame_reached']),
      })

  def _ensure_cursor_room(self):
    # Native frame ring size. Cursor wrap does not reset the match.
    if self._env.t >= self._frame_buffer_length:
      self._env.reset_cursor()

  def _reset_finished_lanes_for_next_observation(self, needs_reset: np.ndarray):
    if np.any(needs_reset):
      self.reset(np.flatnonzero(needs_reset))

def neutral_controllers(batch_size: int) -> Controller:
  shape = (int(batch_size),)
  return Controller(
      main_stick=Stick(
          x=np.full(shape, 0.5, dtype=np.float32),
          y=np.full(shape, 0.5, dtype=np.float32),
      ),
      c_stick=Stick(
          x=np.full(shape, 0.5, dtype=np.float32),
          y=np.full(shape, 0.5, dtype=np.float32),
      ),
      shoulder=np.zeros(shape, dtype=np.float32),
      buttons=Buttons(**{
          name: np.zeros(shape, dtype=np.bool_)
          for name in Buttons._fields
      }),
  )


def write_encoded_controller_action(
    action_frame: np.ndarray,
    controller: Controller,
    *,
    player_index: int,
    source_slice: slice,
    axis_spacing: int,
    shoulder_spacing: int,
):
  """Decode discretized policy controller buckets into melee_sim actions."""
  player = action_frame['p'][:, int(player_index)]
  scale_axis = np.float32(1.0 / float(axis_spacing))
  scale_shoulder = np.float32(1.0 / float(shoulder_spacing))
  player['main_stick_x'][:] = np.asarray(controller.main_stick.x)[source_slice] * scale_axis
  player['main_stick_y'][:] = np.asarray(controller.main_stick.y)[source_slice] * scale_axis
  player['c_stick_x'][:] = np.asarray(controller.c_stick.x)[source_slice] * scale_axis
  player['c_stick_y'][:] = np.asarray(controller.c_stick.y)[source_slice] * scale_axis
  player['shoulder'][:] = np.asarray(controller.shoulder)[source_slice] * scale_shoulder
  for name in Buttons._fields:
    player['buttons'][name][:] = np.asarray(getattr(controller.buttons, name))[source_slice]


def copy_encoded_controller(
    dst: Controller,
    src: Controller,
    *,
    source_slice: slice,
    axis_spacing: int,
    shoulder_spacing: int,
):
  """Decode discretized policy controller buckets into policy-observation state."""
  scale_axis = np.float32(1.0 / float(axis_spacing))
  scale_shoulder = np.float32(1.0 / float(shoulder_spacing))
  dst.main_stick.x[:] = np.asarray(src.main_stick.x)[source_slice] * scale_axis
  dst.main_stick.y[:] = np.asarray(src.main_stick.y)[source_slice] * scale_axis
  dst.c_stick.x[:] = np.asarray(src.c_stick.x)[source_slice] * scale_axis
  dst.c_stick.y[:] = np.asarray(src.c_stick.y)[source_slice] * scale_axis
  dst.shoulder[:] = np.asarray(src.shoulder)[source_slice] * scale_shoulder
  for name in Buttons._fields:
    getattr(dst.buttons, name)[:] = np.asarray(getattr(src.buttons, name))[source_slice]


def _slot_for_source(slots: np.ndarray, source_player: int) -> np.ndarray:
  for slot_index in range(slots.shape[1]):
    slot = slots[:, slot_index]
    present = slot['present'].astype(np.bool_)
    source = slot['source_player']
    if np.all((~present) | (source == int(source_player))) and np.any(present):
      return slot
  raise RuntimeError(f'melee_sim gamestate did not contain source player {source_player}')


def _coerce_character(character) -> melee.Character:
  if isinstance(character, melee.Character):
    return character
  if isinstance(character, str):
    try:
      return _CHARACTER_BY_NAME[character.lower()]
    except KeyError as exc:
      raise ValueError(f'unsupported character {character!r}') from exc
  return melee.Character(character)
