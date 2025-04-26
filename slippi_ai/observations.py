import abc
import dataclasses
import numpy as np

import melee

from slippi_ai import embed
from slippi_ai import utils

class ObservationFilter(abc.ABC):
  @abc.abstractmethod
  def filter(self, game: embed.Game) -> embed.Game:
    pass

class ChainObservationFilter(ObservationFilter):
  def __init__(self, filters: list[ObservationFilter]):
    self.filters = filters

  def filter(self, game: embed.Game) -> embed.Game:
    for filter in self.filters:
      game = filter.filter(game)
    return game

class ActionStateTracker:
  def __init__(self, batch_size: int):
    self.batch_size = batch_size
    self.actions = np.zeros((batch_size,), dtype=np.int32)
    self.counts = np.zeros((batch_size,), dtype=np.uint32)

  def update(self, actions: np.ndarray):
    is_same = actions == self.actions
    self.actions = actions
    self.counts = np.where(is_same, self.counts + 1, 1)


# TODO: what about missed tech?
# TODO: condition on the character
# TODO: not all three are indistinguishable for the same number of frames
TECH_ACTIONS = (
    melee.Action.NEUTRAL_TECH,
    melee.Action.FORWARD_TECH,
    melee.Action.BACKWARD_TECH,
)

def is_tech_action(actions: np.ndarray) -> np.ndarray:
  mask = actions == TECH_ACTIONS[0].value
  for action in TECH_ACTIONS[1:]:
    mask = np.logical_or(mask, actions == action.value)
  return mask

class TechAnimationFilter:
  """Obscures tech animations that look the same for the first N frames."""

  def __init__(self, batch_size: int, num_frames: int):
    self.batch_size = batch_size
    self.num_frames = num_frames
    self.action_states = ActionStateTracker(batch_size)

  def filter(self, game: embed.Game) -> embed.Game:
    self.action_states.update(game.p1.action)

    to_mask = np.logical_and(
        is_tech_action(self.action_states.actions),
        self.action_states.counts <= self.num_frames)

    # maybe randomize instead of using a fixed action?
    masked = np.where(to_mask, TECH_ACTIONS[0].value, game.p1.action)

    return utils.replace_nt(game, ['p1', 'action'], masked)


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
    batch_size: int,
) -> ObservationFilter:
  filters = []
  if config.tech_animation.num_frames > 0:
    filters.append(TechAnimationFilter(batch_size, config.tech_animation.num_frames))
  return ChainObservationFilter(filters)
