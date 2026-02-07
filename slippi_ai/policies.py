import abc
import typing as tp
import enum

from slippi_ai.controller_heads import (
    ControllerHead,
    ControllerType,
)
from slippi_ai.types import Game
from slippi_ai.agents import BasicAgent, RecurrentState

PolicyState = tp.Any

# TODO: find a better place for this
class Platform(enum.Enum):
  TF = 'tf'
  JAX = 'jax'


# Ideally we'd replace Array with Unknown, see https://github.com/python/typing/issues/2169
class Policy(abc.ABC, tp.Generic[ControllerType, RecurrentState]):

  @property
  @abc.abstractmethod
  def platform(self) -> Platform:
    pass

  @property
  @abc.abstractmethod
  def delay(self) -> int:
    pass

  @property
  @abc.abstractmethod
  def controller_head(self) -> ControllerHead[ControllerType]:
    pass

  @abc.abstractmethod
  def encode_game(self, game: Game) -> Game:
    pass

  @abc.abstractmethod
  def initial_state(self, batch_size: int) -> RecurrentState:
    pass

  @abc.abstractmethod
  def build_agent(self, batch_size: int, **kwargs) -> BasicAgent[ControllerType, RecurrentState]:
    """Builds an agent for this policy."""

  @abc.abstractmethod
  def get_state(self) -> PolicyState:
    """Returns the policy parameters."""

  @abc.abstractmethod
  def set_state(self, state: PolicyState):
    """Sets the policy parameters."""
