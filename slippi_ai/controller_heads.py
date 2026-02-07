import abc
import typing as tp

from slippi_ai.types import Controller

ControllerType = tp.TypeVar('ControllerType')

# TODO: don't expose SampleOutputs
class SampleOutputs(tp.NamedTuple, tp.Generic[ControllerType]):
  controller_state: ControllerType
  logits: ControllerType

# This is still here mainly to preserve compatibility with tests/unroll_agent.py,
# which loads pickled DistanceOutputs objects that reference this file.
class DistanceOutputs(tp.NamedTuple, tp.Generic[ControllerType]):
  distance: ControllerType
  logits: ControllerType


class ControllerHead(abc.ABC, tp.Generic[ControllerType]):

  @abc.abstractmethod
  def dummy_controller(self, shape: tp.Sequence[int]) -> ControllerType:
    """Returns a dummy controller state for the given batch shape."""

  @abc.abstractmethod
  def dummy_sample_outputs(
      self,
      shape: tp.Sequence[int],
  ) -> SampleOutputs[ControllerType]:
    """Returns dummy sample outputs for the given batch shape."""

  @abc.abstractmethod
  def decode_controller(
      self,
      controller_state: ControllerType,
  ) -> Controller:
    """Decodes a controller state into a Controller object."""
