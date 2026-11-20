"""Types shared by the SessionManager and its frontends.

Frontends get these from the manager over Ray, so this module stays light: no
TF or JAX imports.
"""

import dataclasses
import datetime
from typing import Optional

import melee

NAME_TO_STAGE = {
    'fd': melee.Stage.FINAL_DESTINATION,
    'fod': melee.Stage.FOUNTAIN_OF_DREAMS,
    'dl': melee.Stage.DREAMLAND,
    'ys': melee.Stage.YOSHIS_STORY,
    'ps': melee.Stage.POKEMON_STADIUM,
    'bf': melee.Stage.BATTLEFIELD,
    'all': melee.Stage.RANDOM_STAGE,
}

MAX_STOCKS = 4

# Convenience aliases on top of the lowercased melee.Character names.
CHARACTER_ALIASES = {
    'puff': melee.Character.JIGGLYPUFF,
    'falcon': melee.Character.CPTFALCON,
    'ics': melee.Character.POPO,
    'icies': melee.Character.POPO,
    'ganon': melee.Character.GANONDORF,
    'gnw': melee.Character.GAMEANDWATCH,
}

_NAME_TO_CHARACTER = {c.name.lower(): c for c in melee.Character}

def parse_character(name: str) -> Optional[melee.Character]:
  name = name.lower()
  if name in CHARACTER_ALIASES:
    return CHARACTER_ALIASES[name]
  return _NAME_TO_CHARACTER.get(name)

class UserError(Exception):
  """A request that can't be done; the message is shown to the user."""

@dataclasses.dataclass(frozen=True)
class User:
  """A user of one of the frontends; users are equal if their ids are."""
  frontend: str  # e.g. 'twitch' or 'web'
  key: str  # unique within the frontend; may be secret
  name: str = dataclasses.field(compare=False)  # shown to other users

  @property
  def id(self) -> str:
    return f'{self.frontend}:{self.key}'

@dataclasses.dataclass
class ManagerInfo:
  bot_connect_code: str
  max_sessions: int
  botmatch_max_games: int
  bots_lock_minutes: float

@dataclasses.dataclass
class PlayResult:
  connect_code: str
  agent: str
  on_stream: bool

@dataclasses.dataclass
class SelectResult:
  # Set if the user was playing and their session was restarted.
  restarted: Optional[PlayResult] = None

@dataclasses.dataclass
class AgentRow:
  """An agent as listed by the web frontend."""
  name: str
  kind: str  # 'auto', 'basic' or 'model'
  character: melee.Character
  opponents: list[melee.Character]  # trained against, or picked for (auto)
  delay: int
  model: Optional[str]  # None for auto agents
  rating: Optional[float]

@dataclasses.dataclass
class SessionLogs:
  session_id: int
  next: int  # number of the next log line
  lines: list[str]

@dataclasses.dataclass
class CharacterRating:
  character: melee.Character
  rating: float
  rank: int  # among the models that play this character
  count: int  # number of rated models that play this character

@dataclasses.dataclass
class AutoPick:
  agent: str
  model: str
  trained: bool  # whether the model was trained against the opponent
  rating: Optional[float]

@dataclasses.dataclass
class BotsStatus:
  bot1: str
  bot2: str
  set_by: Optional[str]
  playing_for: Optional[datetime.timedelta]
  lock_remaining: Optional[datetime.timedelta]

@dataclasses.dataclass
class SessionStatusInfo:
  session_id: int
  user: User
  agent: str
  duration: datetime.timedelta
  # Bot matches have no connect code and count games instead of menu time.
  connect_code: Optional[str]
  on_stream: bool
  # None while the session is starting up (or otherwise not responding).
  menu_time: Optional[datetime.timedelta]
  games_completed: Optional[int]

@dataclasses.dataclass
class ManagerStatus:
  bots: Optional[BotsStatus]  # set if a bot session is on stream
  sessions: list[SessionStatusInfo]

@dataclasses.dataclass
class Announcements:
  """Messages for everyone, e.g. when idle sessions are stopped."""
  next: int  # number of the next announcement
  messages: list[str]

def format_td(td: datetime.timedelta) -> str:
  """Chop off microseconds."""
  return str(td).split('.')[0]
