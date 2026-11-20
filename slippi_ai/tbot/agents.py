"""Agents that the bot can play as."""

import abc
import dataclasses
import os
import typing as tp
from typing import Optional, Union

import melee

from slippi_ai import saving
from slippi_ai import eval_lib

Agent = Union[eval_lib.Agent, eval_lib.EnsembleAgent]

auto_prefix = 'auto-'
imitation_prefix = 'basic-'

@dataclasses.dataclass
class AgentInfo:
  character: melee.Character
  opponents: frozenset[melee.Character]
  # For "special" agents named <group>-<character> (multi-character models
  # and imitation agents), the group prefix; None for regular agents.
  group: Optional[str] = None
  # Whether the underlying model is an imitation model.
  imitation: bool = False

def names_to_str(names: tp.Sequence[str]) -> str:
  if len(names) > 1:
    return "[" + ",".join(names) + "]"
  return names[0]

def group_special_agents(pairs: tp.Iterable[tuple[str, str]]) -> list[str]:
  """Compresses (group, character) pairs into "[groups]-[characters]" strings.

  Characters that share the same set of groups are merged into one entry.
  """
  chars_by_group: dict[str, list[str]] = {}
  for group, char_name in pairs:
    chars_by_group.setdefault(group, []).append(char_name)

  groups_by_chars: dict[tuple[str, ...], list[str]] = {}
  for group, char_names in chars_by_group.items():
    key = tuple(sorted(char_names))
    groups_by_chars.setdefault(key, []).append(group)

  return [
      names_to_str(sorted(groups)) + '-' + names_to_str(char_tuple)
      for char_tuple, groups in groups_by_chars.items()
  ]

def parse_auto_char(agent: str) -> Optional[melee.Character]:
  if not agent.startswith(auto_prefix):
    return None

  char_str = agent.removeprefix(auto_prefix)
  return eval_lib.data.name_to_character.get(char_str)

def parse_imitation_char(agent: str) -> Optional[melee.Character]:
  if not agent.startswith(imitation_prefix):
    return None

  char_str = agent.removeprefix(imitation_prefix)
  return eval_lib.data.name_to_character.get(char_str)

class AgentConfig(abc.ABC):

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """This agent's name."""

  @property
  @abc.abstractmethod
  def delay(self) -> int:
    """The agent's delay."""

  @property
  @abc.abstractmethod
  def batch_steps(self) -> int:
    """Number of (time)steps to batch during inference."""

  @property
  def target_console_delay(self) -> int:
    # Leave a local delay of 1 for async inference.
    target_delay = max(self.delay - self.batch_steps, 0)
    return min(target_delay, MAX_DOLPHIN_DELAY)

  @abc.abstractmethod
  def build(
      self,
      controller: melee.Controller,
      # Actual (netplay) port may different from Controller's (local) port
      port: int,
      opponent_port: int,
      console_delay: int,
  ) -> Agent:
    """Builds an agent for the given port and opponent port."""

  @property
  @abc.abstractmethod
  def character(self) -> melee.Character:
    """The character associated with this agent."""

# Depends on gecko code. TODO: configure
MAX_DOLPHIN_DELAY = 24

class SingleAgent(AgentConfig):

  def __init__(
      self,
      agent_kwargs: dict[str, tp.Any],
      # For models trained to play multiple characters, specify
      # the preferred one for this session.
      character: Optional[melee.Character] = None,
      name: Optional[str] = None,
  ):
    self.agent_kwargs = agent_kwargs
    self._loaded = False
    self._character = character
    self._name = name or os.path.basename(agent_kwargs['path'])

  @property
  def name(self) -> str:
    return self._name

  def _load(self):
    # can't use functools.cache because it isn't serializable
    if self._loaded:
      return
    self.state = saving.load_state_from_disk(self.agent_kwargs['path'])
    self._loaded = True

  @property
  def delay(self) -> int:
    self._load()
    # TODO: find a more robust way of doing this.
    # Note that this could be a tf or jax state/config.
    return self.state['config']['policy']['delay']

  @property
  def batch_steps(self) -> int:
    return self.agent_kwargs['batch_steps'] or 1

  def build(
      self,
      controller: melee.Controller,
      port: int,
      opponent_port: int,
      console_delay: int,
  ) -> eval_lib.Agent:
    self._load()
    return eval_lib.build_agent(
        controller=controller,
        port=port,
        opponent_port=opponent_port,
        console_delay=console_delay,
        state=self.state,
        **self.agent_kwargs,
    )

  @property
  def character(self) -> melee.Character:
    self._load()
    chars = eval_lib.allowed_characters(self.state['config'])

    if self._character is not None:
      assert self._character in chars
      return self._character

    return chars[0]  # Default to the first character in the list.

class AutoAgent(AgentConfig):

  def __init__(
      self,
      character: melee.Character,
      models_path: str,
      opponent_table: dict[melee.Character, str],
      delay: int,
      agent_kwargs: dict = {},
  ):
    """
    Args:
      opponent_table: Maps opponent character to model name.
      delay: The smallest delay of the models in opponent_table, which bounds
        the console delay since that is fixed before the opponent is known.
    """
    self._character = character
    self._delay = delay
    self.models_path = models_path
    self.opponent_table = opponent_table
    self.agent_kwargs = agent_kwargs

  @property
  def name(self) -> str:
    return auto_prefix + self.character.name.lower()

  @property
  def character(self) -> melee.Character:
    return self._character

  @property
  def delay(self) -> int:
    return self._delay

  @property
  def batch_steps(self) -> int:
    return self.agent_kwargs['batch_steps'] or 1

  def build(
      self,
      controller: melee.Controller,
      port: int,
      opponent_port: int,
      console_delay: int,
  ) -> eval_lib.EnsembleAgent:
    return eval_lib.EnsembleAgent(
        models_path=self.models_path,
        opponent_table=self.opponent_table,
        port=port,
        opponent_port=opponent_port,
        controller=controller,
        console_delay=console_delay,
        **self.agent_kwargs,
    )
