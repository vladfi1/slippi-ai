import abc
import typing as tp

import numpy as np

from slippi_ai.data import Game, NAME_DTYPE
from slippi_ai.controller_heads import SampleOutputs, ControllerType

RecurrentState = tp.TypeVar('RecurrentState')

BoolArray = np.ndarray[tuple[int], np.dtype[np.bool]]

class BasicAgent(abc.ABC, tp.Generic[ControllerType, RecurrentState]):
  """Wraps a Policy to track hidden state."""

  @property
  @abc.abstractmethod
  def name_code(self) -> np.ndarray[tuple[int], np.dtype[NAME_DTYPE]]:
    """The (possibly batched) player name code used by this agent."""

  def set_name_code(self, name_code: tp.Union[int, tp.Sequence[int]]):
    raise NotImplementedError()

  def warmup(self):
    """Warm up the agent so that step is fast."""

  @abc.abstractmethod
  def hidden_state(self) -> RecurrentState:
    """Returns the current hidden state."""

  @abc.abstractmethod
  def step(
      self,
      game: Game,
      needs_reset: BoolArray,
  ) -> SampleOutputs[ControllerType]:
    """Doesn't take into account delay."""

  @abc.abstractmethod
  def multi_step(
      self,
      states: list[tuple[Game, BoolArray]],
  ) -> list[SampleOutputs[ControllerType]]:
    """Do multiple steps at once"""
