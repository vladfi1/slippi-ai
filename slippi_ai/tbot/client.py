"""Frontends' access to the SessionManager.

The manager runs as a named Ray actor (see scripts/tbot_backend.py), so the
frontends can be started and stopped without it. Tests use a manager in the
same process instead.
"""

import asyncio
import inspect
import logging
import typing as tp
from typing import Optional

import melee
import ray

from slippi_ai.tbot.api import (
    AgentRow, Announcements, AutoPick, CharacterRating, ManagerInfo,
    ManagerStatus, PlayResult, SelectResult, SessionLogs, User, UserError,
)

if tp.TYPE_CHECKING:
  from slippi_ai.tbot.manager import SessionManager

NAMESPACE = 'tbot'
ACTOR_NAME = 'session_manager'

# Calls a manager method by name.
Call = tp.Callable[..., tp.Awaitable[tp.Any]]
Listener = tp.Callable[[str], tp.Awaitable[None]]

def _local_call(manager: 'SessionManager') -> Call:
  async def call(method: str, *args, **kwargs) -> tp.Any:
    result = getattr(manager, method)(*args, **kwargs)
    if inspect.isawaitable(result):
      result = await result
    return result
  return call

class _RemoteCall:
  """Calls the manager actor, finding it again if it has restarted."""

  def __init__(self, actor: ray.actor.ActorHandle, name: str, namespace: str):
    self._actor = actor
    self._name = name
    self._namespace = namespace

  async def __call__(self, method: str, *args, **kwargs) -> tp.Any:
    try:
      return await getattr(self._actor, method).remote(*args, **kwargs)
    except ray.exceptions.RayTaskError as e:
      if isinstance(e.cause, UserError):
        raise e.cause from None
      raise
    except ray.exceptions.RayActorError:
      logging.warning('The manager actor is gone; looking it up again.')

    try:
      self._actor = await asyncio.to_thread(
          ray.get_actor, self._name, namespace=self._namespace)
    except ValueError:
      raise UserError('The bot is down; try again later.') from None
    raise UserError('The bot restarted; try again.')

class ManagerClient:
  """Async access to a SessionManager, in this process or in a Ray actor."""

  def __init__(self, call: Call):
    self._call = call

  @classmethod
  def local(cls, manager: 'SessionManager') -> 'ManagerClient':
    return cls(_local_call(manager))

  @classmethod
  def connect(
      cls, name: str = ACTOR_NAME, namespace: str = NAMESPACE,
  ) -> 'ManagerClient':
    """Finds the manager actor; ray.init must have joined its cluster."""
    try:
      actor = ray.get_actor(name, namespace=namespace)
    except ValueError:
      raise RuntimeError(
          f'No {name} actor in namespace {namespace}; is the backend'
          ' (scripts/tbot_backend.py) running?') from None
    return cls(_RemoteCall(actor, name, namespace))

  async def listen(self, listener: Listener):
    """Passes each new announcement to listener, forever."""
    since: Optional[int] = None
    while True:
      try:
        result = await self.announcements(since)
      except Exception:
        logging.exception('Failed to get announcements')
        await asyncio.sleep(5)
        continue
      if since is not None:
        for message in result.messages:
          try:
            await listener(message)
          except Exception:
            logging.exception('Listener failed')
      since = result.next

  # The manager's methods.

  async def info(self) -> ManagerInfo:
    return await self._call('info')

  async def announcements(
      self, since: Optional[int], timeout: float = 30) -> Announcements:
    return await self._call('announcements', since, timeout)

  async def reload(self) -> int:
    return await self._call('reload')

  async def agent_names(self) -> list[str]:
    return await self._call('agent_names')

  async def special_agents(self) -> list[str]:
    return await self._call('special_agents')

  async def find_agents(
      self,
      char: Optional[melee.Character] = None,
      opponent: Optional[melee.Character] = None,
  ) -> list[str]:
    return await self._call('find_agents', char, opponent)

  async def agent_table(self) -> list[AgentRow]:
    return await self._call('agent_table')

  async def model_names(self) -> list[str]:
    return await self._call('model_names')

  async def model_config(self, model: str) -> dict:
    return await self._call('model_config', model)

  async def auto_agent_character(self, name: str) -> Optional[melee.Character]:
    return await self._call('auto_agent_character', name)

  async def ratings(self, name: str) -> tuple[str, list[CharacterRating]]:
    return await self._call('ratings', name)

  async def auto_pick(
      self, char: melee.Character, opponent: melee.Character) -> AutoPick:
    return await self._call('auto_pick', char, opponent)

  async def selected_agent(self, user: User) -> str:
    return await self._call('selected_agent', user)

  async def select_agent(self, user: User, agent_name: str) -> SelectResult:
    return await self._call('select_agent', user, agent_name)

  async def touch(self, user: User):
    await self._call('touch', user)

  async def connect_code(self, user: User) -> Optional[str]:
    return await self._call('connect_code', user)

  async def set_connect_code(self, user: User, connect_code: str):
    await self._call('set_connect_code', user, connect_code)

  async def stages(self, user: User) -> Optional[list[melee.Stage]]:
    return await self._call('stages', user)

  async def set_stages(self, user: User, stages: Optional[list[melee.Stage]]):
    await self._call('set_stages', user, stages)

  async def sd_stocks(self, user: User) -> Optional[int]:
    return await self._call('sd_stocks', user)

  async def set_sd_stocks(self, user: User, stocks: int):
    await self._call('set_sd_stocks', user, stocks)

  async def play(self, user: User) -> PlayResult:
    return await self._call('play', user)

  async def stop(self, user: User):
    await self._call('stop', user)

  async def reset(self, user: User) -> PlayResult:
    return await self._call('reset', user)

  async def kick(self, name: str) -> User:
    return await self._call('kick', name)

  async def botmatch(
      self, user: User, agent1_name: str, agent2_name: str) -> str:
    return await self._call('botmatch', user, agent1_name, agent2_name)

  async def session_id(self, user: User) -> Optional[int]:
    return await self._call('session_id', user)

  async def session_logs(
      self, user: User, since: int) -> Optional[SessionLogs]:
    return await self._call('session_logs', user, since)

  async def start_bot_session(self) -> bool:
    return await self._call('start_bot_session')

  async def stop_bot_session(self) -> bool:
    return await self._call('stop_bot_session')

  async def set_bots(
      self, user: User, bot1: str, bot2: str, is_mod: bool = False):
    await self._call('set_bots', user, bot1, bot2, is_mod)

  async def relinquish(self, user: User, is_mod: bool = False):
    await self._call('relinquish', user, is_mod)

  async def status(self) -> ManagerStatus:
    return await self._call('status')

  async def gc(self) -> list[User]:
    return await self._call('gc')
