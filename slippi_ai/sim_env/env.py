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

from slippi_ai import dolphin, utils
from slippi_ai.envs import EnvOutput
from slippi_ai.sim_env.observations import (
    GameBatchBuffer, copy_controller_slice as _copy_controller_slice,
    game_for_port, GameBatch, build_trajectory_buffer
)
from slippi_ai.types import Buttons, Controller, Stick, reify_tuple_type


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


PlayerConfigs = tp.Mapping[int, dolphin.AI]
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
      seed_offset: int = 0,
  ):
    self._num_envs = num_envs
    self._frame_buffer_length = frame_buffer_length
    self._max_frame_id = max_frame_id
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
    elif isinstance(players, tp.Mapping):
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

    # Seed with the global env index so a process-sharded batch plays out
    # identically to a single-process batch with the same configs. (Without an
    # explicit seed the native env seeds each match with its local lane index,
    # which differs across shardings.)
    match_configs = [
        melee_sim.MatchConfig(
            stage=_MELEE_TO_SIM_STAGE[stage],
            players=(
                melee_sim.PlayerConfig(character=int(char_pair[0].value)),
                melee_sim.PlayerConfig(character=int(char_pair[1].value)),
            ),
            max_frame=self._max_frame_id,
            seed=seed_offset + i,
        )
        for i, (stage, char_pair) in enumerate(
            zip(self._stage_by_env, character_assignments))
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
    # TODO: this is only used in test_sim_env.py and should be removed
    game_batch = utils.map_single_structure(
        lambda dtype: np.empty([self._num_envs * 2], dtype=dtype),
        reify_tuple_type(GameBatch),
    )
    self._game_batch_buffer = GameBatchBuffer(game_batch)

  def stop(self):
    self._env.close()

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
  def game_batch_buffers(self) -> GameBatchBuffer:
    return self._game_batch_buffer

  def write_current_game(
      self,
      game_batch: GameBatchBuffer,
      needs_reset: np.ndarray | None = None,
      *,
      env_slice: slice | None = None,  # overridden when inside an MP env
  ):
    """Write the current sim observation into reusable policy input buffers."""
    needs_reset = np.zeros(self._num_envs, dtype=np.bool_) if needs_reset is None else needs_reset
    if env_slice is None:
      env_slice = slice(0, self._num_envs)
    game_batch.fill_slice(
        self._env.current_frame,
        needs_reset,
        env_slice,
        self._last_controllers,
    )

  def reset(self, env_ids: tp.Sequence[int] | np.ndarray | None = None) -> np.ndarray:
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
            self._last_controllers[port], neutral, ids)
    needs_reset = np.zeros(self._num_envs, dtype=np.bool_)
    needs_reset[ids] = True
    return needs_reset

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

  @property
  def episode_ids(self) -> np.ndarray:
    return self._episode_ids

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

  def advance(self, controllers: Controllers):
    self._ensure_cursor_room()

    action = self._env.current_action_frame
    for player_index, port in enumerate(self._ports):
      controller = controllers[port]
      player = action['players'][:, int(player_index)]
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
      )
    if self._fake:
      needs_reset = np.zeros(self._num_envs, dtype=np.bool_)
      return needs_reset

    step_t = self._env.t
    self._env.step()
    needs_reset = self._env.done_at(step_t).astype(np.bool_, copy=True)
    terminal = self._env.terminal_at(step_t).copy()
    self._last_step_info = SimStepInfo(terminal=terminal, step_t=step_t)
    self._record_completed_games(terminal, self._env.current_frame)
    self._reset_finished_lanes_for_next_observation(needs_reset)
    return needs_reset

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

class CompatSimBatchedEnvironment(SimBatchedEnvironment):
  """Presents an interface compatible with slippi_ai.envs.BatchedEnvironment"""

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
    super().__init__(
        num_envs=num_envs,
        players=players,
        frame_buffer_length=frame_buffer_length,
        stage=stage,
        character_pool=character_pool,
        max_frame_id=max_frame_id,
        data_dir=data_dir,
        fake=fake,
    )

    # Old (slower) async interface.
    # Match the existing env push/pop contract by seeding an initial observation.
    self._output_queue = collections.deque([
        self.current_state(needs_reset=np.ones(self._num_envs, dtype=np.bool_))
    ])

  @contextlib.contextmanager
  def run(self):
    try:
      yield self
    finally:
      self.stop()

  def reset(self, env_ids: tp.Sequence[int] | np.ndarray | None = None) -> EnvOutput:
    needs_reset = super().reset(env_ids)
    return self.current_state(needs_reset=needs_reset)

  def push(self, controllers: Controllers):
    self._output_queue.append(self.step(controllers))

  def pop(self) -> EnvOutput:
    return self._output_queue.popleft()

  def peek(self) -> EnvOutput:
    return self._output_queue[0]

  def step(self, controllers: Controllers) -> EnvOutput:
    needs_reset = self.advance(controllers)
    return self.current_state(needs_reset=needs_reset)

  def multi_step(self, controllers: list[Controllers]) -> list[EnvOutput]:
    return [self.step(c) for c in controllers]


# We use two different subclasses to avoid type signature clash on push/pop
class AsyncSimBatchedEnvironment(SimBatchedEnvironment):
  """Synchronously implements new async interface."""

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
      runahead: int = 0,  # number of extra actions that can be pushed
  ):
    super().__init__(
        num_envs=num_envs,
        players=players,
        frame_buffer_length=frame_buffer_length,
        stage=stage,
        character_pool=character_pool,
        max_frame_id=max_frame_id,
        data_dir=data_dir,
        fake=fake,
    )

    self._needs_reset = np.ones(self._num_envs, dtype=np.bool_)

    # New async interface
    self._capacity = runahead + 1
    self._read_index = 0
    self._write_index = 0
    self._pushed_minus_popped = 1
    self._trajectory_buffer = build_trajectory_buffer(
        GameBatch,
        batch_size=num_envs * 2,
        rollout_length=self._capacity,
    )
    self._game_batch_buffers = [
        GameBatchBuffer(game_batch)
        for game_batch in self._trajectory_buffer.slots
    ]
    self.write_current_game(
        self._game_batch_buffers[self._write_index], self._needs_reset)

  def pop(self) -> GameBatch:
    self._pushed_minus_popped -= 1
    if self._pushed_minus_popped < 0:
      raise RuntimeError('not enough actions pushed')

    obs = self._trajectory_buffer.slots[self._read_index]
    self._read_index = (self._read_index + 1) % self._capacity
    return obs

  def push(self, controllers: Controllers):
    self._pushed_minus_popped += 1
    if self._pushed_minus_popped > self._capacity:
      raise RuntimeError('too many actions pushed')

    self._needs_reset = self.advance(controllers)
    self._write_index = (self._write_index + 1) % self._capacity
    self.write_current_game(
        self._game_batch_buffers[self._write_index], self._needs_reset)

  def peek(self) -> GameBatch:
    if not (self._pushed_minus_popped > 0):
      raise RuntimeError('no observations available')
    return self._trajectory_buffer.slots[self._read_index]


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
