import abc
import dataclasses
import typing as tp

import numpy as np

import melee
from melee.enums import Action

from slippi_ai.types import Game, Rank1, BoolArray, S, Player
from slippi_ai import utils

Game0 = Game[tuple[()]]
Game1 = Game[Rank1]

class ObservationFilter(abc.ABC, tp.Generic[S]):

  def reset(self):
    """Reset the filter state."""
    pass

  @abc.abstractmethod
  def filter(self, game: Game0) -> Game0:
    """Filter a single unbatched frame."""

  @abc.abstractmethod
  def filter_time(self, game: Game1) -> Game1:
    """Filter a time-batched game."""

  def reset_batched(self, needs_reset: BoolArray[S]):
    """Reset the filter state for a batch of games."""
    raise NotImplementedError()

  def filter_batched(self, game: Game[S], env_slice: tp.Optional[slice] = None):
    """Filter a batched game in-place."""
    raise NotImplementedError()

class ChainObservationFilter(ObservationFilter):

  def __init__(self, filters: list[ObservationFilter]):
    self.filters = filters

  def reset(self):
    for filter in self.filters:
      filter.reset()

  def filter_batched(self, game: Game[S], env_slice: tp.Optional[slice] = None):
    for filter in self.filters:
      filter.filter_batched(game, env_slice)

  def reset_batched(self, needs_reset: BoolArray[S]):
    for filter in self.filters:
      filter.reset_batched(needs_reset)

  def filter(self, game: Game0) -> Game0:
    for filter in self.filters:
      game = filter.filter(game)
    return game

  def filter_time(self, game: Game1) -> Game1:
    for filter in self.filters:
      game = filter.filter_time(game)
    return game

ActionDType = np.uint16
assert Player.__annotations__['action'].__args__[1].__args__[0] is ActionDType

# TODO: what about missed tech?
N_TECH, F_TECH, B_TECH = [
    ActionDType(action.value) for action in
    [Action.NEUTRAL_TECH, Action.FORWARD_TECH, Action.BACKWARD_TECH]
]

# A set of actions indistinguishable for the first N frames.
ActionSet = tuple[tuple[ActionDType, ...], int]

# Every character has a list of ActionSets from less to more specific.
# Note: the frame count is 1-indexed.
INDISTINGUISHABLE_ACTIONS: dict[int, list[ActionSet]] = {
    # https://www.fightcore.gg/characters/224/fox/moves/1901/neutraltech/
    melee.Character.FOX.value: [
        ((N_TECH, F_TECH, B_TECH), 4),
        ((N_TECH, F_TECH), 7),
    ],
    # https://www.fightcore.gg/characters/218/falco/moves/1902/neutraltech/
    melee.Character.FALCO.value: [
        ((N_TECH, F_TECH, B_TECH), 4),
        ((N_TECH, F_TECH), 7),
    ],
    # https://www.fightcore.gg/characters/222/marth/moves/1900/neutraltech/
    melee.Character.MARTH.value: [
        ((N_TECH, F_TECH, B_TECH), 7),
    ],
    # https://www.fightcore.gg/characters/260/sheik/moves/1897/neutraltech/
    melee.Character.SHEIK.value: [
        ((N_TECH, F_TECH, B_TECH), 3),
        ((N_TECH, B_TECH), 7),
    ],
    # https://www.fightcore.gg/characters/227/captainfalcon/moves/1879/neutraltech/
    melee.Character.CPTFALCON.value: [
        ((N_TECH, F_TECH, B_TECH), 7),
    ],
    # https://www.fightcore.gg/characters/214/jigglypuff/moves/1899/neutraltech/
    melee.Character.JIGGLYPUFF.value: [
        ((N_TECH, F_TECH, B_TECH), 7),
    ],
    # https://www.fightcore.gg/characters/240/peach/moves/1896/neutraltech/
    # TODO: the data looks a bit weird, some frames are exact duplicates
    melee.Character.PEACH.value: [
        ((N_TECH, F_TECH, B_TECH), 2),
        ((N_TECH, F_TECH), 8),
    ],
    # https://www.fightcore.gg/characters/209/iceclimbers/moves/1894/neutraltech/
    melee.Character.POPO.value: [
        ((N_TECH, F_TECH, B_TECH), 7),
    ],
    # https://www.fightcore.gg/characters/211/yoshi/moves/1893/neutraltech/
    melee.Character.YOSHI.value: [
        ((N_TECH, F_TECH, B_TECH), 8),
    ],
    # https://www.fightcore.gg/characters/212/samus/moves/1898/neutraltech/
    # The one could be 7 or 8.
    melee.Character.SAMUS.value: [
        ((N_TECH, F_TECH, B_TECH), 7),
    ],
    # https://www.fightcore.gg/characters/237/pikachu/moves/1895/neutraltech/
    melee.Character.DK.value: [
        ((N_TECH, F_TECH, B_TECH), 7),
        ((N_TECH, F_TECH), 9),
    ],
    # https://www.fightcore.gg/characters/215/drmario/moves/1891/neutraltech/
    melee.Character.DOC.value: [
        ((N_TECH, F_TECH, B_TECH), 7),
        ((N_TECH, F_TECH), 8),
    ],
}

DEFAULT_TECH_MASK_WINDOW = 7
TECH_ACTIONS = (N_TECH, F_TECH, B_TECH)
_TECH_ACTIONS_NP = np.array(TECH_ACTIONS, dtype=ActionDType)

# TODO: fill in data for all characters
DEFAULT_ACTION_DATA = [(TECH_ACTIONS, DEFAULT_TECH_MASK_WINDOW)]

def get_masks(action_data: list[ActionSet]):
  action_to_masked: dict[ActionDType, list[ActionDType]] = {}

  for action_set, num_frames in action_data:
    for action in action_set:
      masked = action_to_masked.setdefault(action, [])
      masked.extend([action_set[0]] * (num_frames - len(masked)))

  return action_to_masked

DEFAULT_MASKS = get_masks(DEFAULT_ACTION_DATA)

ACTION_MASKS = {
    char: get_masks(action_data) for char, action_data
    in INDISTINGUISHABLE_ACTIONS.items()
}
ACTION_MASKS = {}

def mask_tech_action_with_char(char: int, action: ActionDType, frame: int) -> ActionDType:
  """Mask actions that are indistinguishable.

  For every set of indistinguishable actions, we return an arbitrary member of
  the set (for now, this is always NEUTRAL_TECH). We could also randomize or
  create a new unique action id for each set.
  """
  masks = ACTION_MASKS.get(char, DEFAULT_MASKS).get(action, None)

  if masks is None or frame >= len(masks):
    return action

  return masks[frame]

class AnimationFilter(ObservationFilter, tp.Generic[S]):
  """Obscures tech animations that look the same for the first N frames."""

  def __init__(
      self,
      shape: S = (),
      tech_mask_window: int = DEFAULT_TECH_MASK_WINDOW,
      fast: bool = True,
  ):
    self.shape = shape
    self.tech_mask_window = tech_mask_window
    self.fast = fast
    if not fast:
      assert shape == (), 'Non-fast mode only supported for unbatched games'
    self.reset()

  def reset(self):
    self.prev_action = np.zeros(self.shape, dtype=ActionDType)
    # Number of times the current action has been repeated.
    self.count = np.zeros(self.shape, dtype=int)

  def filter_batched(self, game: Game[S], env_slice: tp.Optional[slice] = None):
    """Updates in-place."""
    if env_slice is not None:
      actions = game.p0.action[env_slice]
    else:
      actions = game.p0.action
    del game

    same_action = actions == self.prev_action
    self.count[:] += 1
    self.count[~same_action] = 0
    self.prev_action[:] = actions

    needs_mask = self.count < self.tech_mask_window & np.isin(actions, _TECH_ACTIONS_NP)
    actions[needs_mask] = _TECH_ACTIONS_NP[0]

  def reset_batched(self, needs_reset: BoolArray[S]):
    self.prev_action[needs_reset] = 0
    self.count[needs_reset] = 0

  def update(self, char: int, action: ActionDType) -> ActionDType:
    if action == self.prev_action:
      self.count += 1
    else:
      self.count[...] = 0
      self.prev_action[...] = action

    if not self.fast:
      return mask_tech_action_with_char(char, action, self.count.item())

    is_tech_action = action in TECH_ACTIONS
    should_mask = is_tech_action and (self.count < self.tech_mask_window)
    if should_mask:
      return TECH_ACTIONS[0]

    return action

  def filter(self, game: Game0) -> Game0:
    assert self.shape == ()

    masked_action = self.update(game.p1.character, game.p1.action)
    if masked_action != game.p1.action:
      return utils.replace_nt(game, ['p1', 'action'], masked_action)
    return game

  def filter_time(self, game: Game1) -> Game1:
    assert self.shape == ()

    actions = np.concatenate([np.full([1], self.prev_action), game.p1.action])

    same_action = actions[:-1] == actions[1:]
    cumsum = np.cumsum(same_action) + self.count
    reset_values = np.maximum.accumulate(np.where(same_action, 0, cumsum))
    action_frames = cumsum - reset_values
    self.count = action_frames[-1]

    if not self.fast:
      raise NotImplementedError('Non-fast mode not implemented')

    # TODO: re-enable per-character masking
    is_tech_action = np.isin(game.p1.action, TECH_ACTIONS)
    should_mask = is_tech_action & (action_frames <= self.tech_mask_window)

    masked_actions = np.where(
        should_mask, TECH_ACTIONS[0], game.p1.action)

    return utils.replace_nt(game, ['p1', 'action'], masked_actions)

def floor_mod(a: np.ndarray, b: int) -> np.ndarray:
  """Replaces each x with the largest multiple of b less than or equal to x."""
  return a - np.mod(a, b)

class FrameSkipFilter(ObservationFilter):
  """Keeps only every N-th frame, except for the controller."""

  def __init__(self, skip: int):
    self.skip = skip
    self.reset()

  def reset(self):
    self.index = 0

  def filter(self, game: Game) -> Game:

    if self.index % self.skip == 0:
      self.last_state = game
      self.index += 1
      return game

    self.index += 1
    return utils.replace_nt(
        self.last_state, ['p0', 'controller'], game.p0.controller)

  def filter_time(self, game: Game) -> Game:
    game_len = len(game.stage)
    indices = floor_mod(np.arange(game_len), self.skip)
    frame_skipped = utils.map_nt(lambda arr: arr[indices], game)
    return utils.replace_nt(
        frame_skipped, ['p0', 'controller'], game.p0.controller)

field = lambda x: dataclasses.field(default_factory=x)

@dataclasses.dataclass
class AnimationConfig:
  mask: bool = True
  tech_mask_window: int = DEFAULT_TECH_MASK_WINDOW

@dataclasses.dataclass
class FrameSkipConfig:
  skip: int = 0

@dataclasses.dataclass
class ObservationConfig:
  animation: AnimationConfig = field(AnimationConfig)
  frame_skip: FrameSkipConfig = field(FrameSkipConfig)

# Mimics behavior pre-observation filtering
NULL_OBSERVATION_CONFIG = ObservationConfig(
    animation=AnimationConfig(mask=False),
)


def build_observation_filter(
    config: ObservationConfig,
    shape: S = (),
) -> ObservationFilter[S]:
  filters = []
  if config.animation.mask:
    filters.append(AnimationFilter[S](shape=shape, tech_mask_window=config.animation.tech_mask_window))
  if config.frame_skip.skip > 1:
    filters.append(FrameSkipFilter(skip=config.frame_skip.skip))
  return ChainObservationFilter(filters)
