import abc
import dataclasses
import numpy as np

import melee
from melee.enums import Action

from slippi_ai import types
from slippi_ai import utils


class ObservationFilter(abc.ABC):

  def reset(self):
    """Reset the filter state."""
    pass

  @abc.abstractmethod
  def filter(self, game: types.Game) -> types.Game:
    """Filter a single unbatched frame."""

  @abc.abstractmethod
  def filter_time(self, game: types.Game) -> types.Game:
    """Filter a time-batched game."""

class ChainObservationFilter(ObservationFilter):

  def __init__(self, filters: list[ObservationFilter]):
    self.filters = filters

  def reset(self):
    for filter in self.filters:
      filter.reset()

  def filter(self, game: types.Game) -> types.Game:
    for filter in self.filters:
      game = filter.filter(game)
    return game

  def filter_time(self, game: types.Game) -> types.Game:
    for filter in self.filters:
      game = filter.filter_time(game)
    return game

ActionDType = np.uint16
assert types.Player.__annotations__['action'] is ActionDType

# TODO: what about missed tech?
N_TECH, F_TECH, B_TECH = [
    ActionDType(action.value) for action in
    [Action.NEUTRAL_TECH, Action.FORWARD_TECH, Action.BACKWARD_TECH]
]

# A set of actions indistinguishable for a certain number of frames.
ActionSet = tuple[tuple[ActionDType, ...], int]

# Every character has a list of ActionSets from less to more specific.
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
}

# TODO: fill in data for all characters
DEFAULT_ACTION_DATA = [((N_TECH, F_TECH, B_TECH), 7)]

def mask_tech_action(char: int, action: ActionDType, frame: int) -> ActionDType:
  """Mask actions that are indistinguishable.

  For every set of indistinguishable actions, we return an arbitrary member of
  the set (for now, this is always NEUTRAL_TECH). We could also randomize or
  create a new unique action id for each set.
  """
  action_data = INDISTINGUISHABLE_ACTIONS.get(char, DEFAULT_ACTION_DATA)

  for action_set, num_frames in action_data:
    if frame <= num_frames and action in action_set:
      return action_set[0]

  return action

class AnimationFilter(ObservationFilter):
  """Obscures tech animations that look the same for the first N frames."""

  def __init__(self):
    self.reset()

  def reset(self):
    self.prev_action = ActionDType(0)
    self.count = 0

  def update(self, char: int, action: ActionDType) -> ActionDType:
    if action == self.prev_action:
      self.count += 1
    else:
      self.count = 1
      self.prev_action = action

    return mask_tech_action(char, action, self.count)

  def filter(self, game: types.Game) -> types.Game:
    masked_action = self.update(game.p1.character, game.p1.action)
    if masked_action != game.p1.action:
      return utils.replace_nt(game, ['p1', 'action'], masked_action)
    return game

  def filter_time(self, game: types.Game) -> types.Game:
    masked_actions = np.copy(game.p1.action)

    for i, (char, action) in enumerate(zip(game.p1.character, game.p1.action)):
      masked_actions[i] = self.update(char, action)

    return utils.replace_nt(game, ['p1', 'action'], masked_actions)

@dataclasses.dataclass
class AnimationConfig:
  mask: bool = True

@dataclasses.dataclass
class ObservationConfig:
  animation: AnimationConfig = AnimationConfig()

# Mimics behavior pre-observation filtering
NULL_OBSERVATION_CONFIG = ObservationConfig(
    animation=AnimationConfig(mask=False),
)


def build_observation_filter(
    config: ObservationConfig,
) -> ObservationFilter:
  filters = []
  if config.animation.mask:
    filters.append(AnimationFilter())
  return ChainObservationFilter(filters)
