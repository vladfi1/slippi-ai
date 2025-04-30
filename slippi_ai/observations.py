import abc
import dataclasses
import numpy as np

import melee

from slippi_ai import types
from slippi_ai import utils


class ObservationFilter(abc.ABC):

  def reset(self):
    """Reset the filter state."""
    pass

  @abc.abstractmethod
  def filter(self, game: types.Game) -> types.Game:
    """Filter a single unbatched game."""

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

# TODO: what about missed tech?
# TODO: condition on the character
# TODO: not all three are indistinguishable for the same number of frames
TECH_ACTIONS = (
    melee.Action.NEUTRAL_TECH,
    melee.Action.FORWARD_TECH,
    melee.Action.BACKWARD_TECH,
)

TECH_ACTION_VALUES = tuple(action.value for action in TECH_ACTIONS)


class TechAnimationFilter(ObservationFilter):
  """Obscures tech animations that look the same for the first N frames."""

  def __init__(self, num_frames: int):
    self.num_frames = num_frames
    self.reset()

  def reset(self):
    self.prev_action = np.uint16(0)
    self.count = 0

  def update(self, action: np.uint16) -> np.uint16:
    if action == self.prev_action:
      self.count += 1
    else:
      self.count = 1
      self.prev_action = action

    if action in TECH_ACTION_VALUES and self.count <= self.num_frames:
      return TECH_ACTION_VALUES[0]
    return action

  def filter(self, game: types.Game) -> types.Game:
    masked_action = self.update(game.p1.action)
    if masked_action != game.p1.action:
      return utils.replace_nt(game, ['p1', 'action'], masked_action)
    return game

  def filter_time(self, game: types.Game) -> types.Game:
    masked_actions = np.copy(game.p1.action)

    for i, action in enumerate(game.p1.action):
      masked_actions[i] = self.update(action)

    return utils.replace_nt(game, ['p1', 'action'], masked_actions)

@dataclasses.dataclass
class TechAnimationConfig:
  num_frames: int = 0


@dataclasses.dataclass
class ObservationConfig:
  tech_animation: TechAnimationConfig = TechAnimationConfig()

# Mimics behavior pre-observation filtering
NULL_OBSERVATION_CONFIG = ObservationConfig()


def build_observation_filter(
    config: ObservationConfig,
) -> ObservationFilter:
  filters = []
  if config.tech_animation.num_frames > 0:
    filters.append(TechAnimationFilter(config.tech_animation.num_frames))
  return ChainObservationFilter(filters)
