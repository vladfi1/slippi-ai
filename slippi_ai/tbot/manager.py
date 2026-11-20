"""Sessions and state shared by the bot's frontends (twitch, web)."""

import asyncio
import dataclasses
import datetime
import json
import logging
import os
import sqlite3
import time
import typing as tp
from typing import Optional

import portpicker
import ray

import melee

from slippi_ai import saving
from slippi_ai import eval_lib
from slippi_ai import dolphin as dolphin_lib
from slippi_ai.eval_db import db as eval_db_lib
from slippi_ai.tbot import sessions as sessions_lib
from slippi_ai.tbot.agents import (
    AgentConfig, AgentInfo, AutoAgent, SingleAgent,
    auto_prefix, imitation_prefix,
    group_special_agents, parse_imitation_char,
)
from slippi_ai.tbot.persistence import (
    LoadedModel, ModelCache, UserPrefs, trim_model_state,
)

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

# Keys into UserPrefs.
AGENT_PREF = 'agent'
CONNECT_CODE_PREF = 'connect_code'

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
class SessionInfo:
  id: int  # unique among the sessions this manager has started
  session: sessions_lib.Session  # actually a RemoteSession or RemoteBotSession
  start_time: datetime.datetime
  user: User
  connect_code: Optional[str]  # None for bot matches
  agent: str

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

# Called with messages for everyone, e.g. when idle sessions are stopped.
Listener = tp.Callable[[str], tp.Awaitable[None]]

class SessionManager:
  """Owns the sessions, the stream slot and the models.

  Public async methods take the lock; private methods assume it is held.
  """

  def __init__(
      self,
      dolphin_config: dolphin_lib.DolphinConfig,
      agent_kwargs: dict,
      models_path: str,
      max_sessions: int = 4,  # Includes stream session.
      menu_timeout: float = 3,  # in minutes
      stream: bool = True,
      bot_session_interval: float = 1, # in minutes
      bots_lock_minutes: float = 15,
      bot: Optional[str] = None,
      bot2: Optional[str] = None,
      auto_delay: int = 18,
      eval_db_path: Optional[str] = None,
      rating_offset: float = 2000,
      default_agent: Optional[str] = None,
      costume: Optional[int] = None,
      botmatch_replay_dir: str = 'Replays/BotMatch',
      botmatch_max_games: int = 10,
      state_dir: Optional[str] = None,
  ):
    """
    Args:
      auto_delay: Delay of the basic-* agents, and of the auto-* agents when
        there are no eval ratings.
      eval_db_path: Eval database whose ratings pick the auto-* agents.
      rating_offset: Added to the eval ratings, which have mean zero.
      state_dir: Directory for state that persists across restarts (model
        config cache, user preferences). None disables persistence.
    """
    self.dolphin_config = dolphin_config
    self.agent_kwargs = agent_kwargs
    self.run_on_cpu = agent_kwargs['run_on_cpu']

    if self.run_on_cpu:
      eval_lib.disable_gpus()
    else:
      sessions_lib.set_memory_growth()

    self.max_sessions = max_sessions
    self._menu_timeout = menu_timeout
    self._stream = stream
    self._bot_session_interval = bot_session_interval
    self.bots_lock_minutes = bots_lock_minutes

    self._auto_delay = auto_delay
    self._costume = costume
    self._botmatch_replay_dir = botmatch_replay_dir
    self.botmatch_max_games = botmatch_max_games

    # Keyed by User.id.
    self._sessions: dict[str, SessionInfo] = {}
    self._next_session_id = 0
    self._streaming_against: Optional[User] = None
    self._last_stream_time: Optional[float] = None
    self._bot_session: Optional[sessions_lib.BotSession] = None  # actually RemoteBotSession
    # Who last set the bots and when (for status), and when their lock expires.
    self._bots_set_by: Optional[User] = None
    self._bots_set_time: Optional[datetime.datetime] = None
    self._bots_lock_until: Optional[datetime.datetime] = None

    self._lock = asyncio.Lock()
    self._listeners: list[Listener] = []

    with open(dolphin_config.user_json_path) as f:
      user_json = json.load(f)
    self.bot_connect_code: str = user_json['connectCode']

    self._models_path = models_path
    self._eval_db_path = eval_db_path
    self._rating_offset = rating_offset
    self._default_agent = default_agent
    # Cache of loaded model states, keyed by model name. Each entry holds the
    # file mtime it was loaded at, the (trimmed) state, and its summary.
    self._loaded_models: dict[str, LoadedModel] = {}
    # Optional on-disk cache of the trimmed states, to speed up startup.
    self._model_cache: Optional[ModelCache] = None
    user_prefs_path: Optional[str] = None
    if state_dir:
      self._model_cache = ModelCache(os.path.join(state_dir, 'cached_models'))
      user_prefs_path = os.path.join(state_dir, 'user_prefs.json')
    self._reload_models()

    # Persisted per-user preferences, keyed by User.id.
    self._user_prefs = UserPrefs(user_prefs_path)
    # Prefs used to be keyed by twitch name, which can't contain ':'.
    self._user_prefs.rename_users(
        lambda user: user if ':' in user else f'twitch:{user}')

    if bot is None:
      bot1_config = self._default_agent_config
    else:
      bot1_config = self._single_agent(path=bot)

    if bot2 is None:
      bot2_config = bot1_config
    else:
      bot2_config = self._single_agent(path=bot2)

    self._bot_configs: dict[int, AgentConfig] = {
        1: bot1_config,
        2: bot2_config,
    }

    # User-specific state (not persisted), keyed by User.id.
    self._stages: dict[str, list[melee.Stage]] = {}
    # Number of stocks the bot should SD down to at the start of each game.
    self._sd_stocks: dict[str, int] = {}

  def add_listener(self, listener: Listener):
    self._listeners.append(listener)

  async def _announce(self, message: str):
    for listener in self._listeners:
      try:
        await listener(message)
      except Exception:
        logging.exception('Listener failed')

  # Agents

  def _single_agent(
      self,
      model: Optional[str] = None,
      path: Optional[str] = None,
      char: Optional[melee.Character] = None,
      name: Optional[str] = None,
  ) -> SingleAgent:
    if model is not None:
      if path is not None:
        raise ValueError('Cannot specify both model and path for SingleAgent')
      path = os.path.join(self._models_path, model)
    elif path is None:
      raise ValueError('Must specify either model or path for SingleAgent')

    return SingleAgent(
        agent_kwargs=dict(self.agent_kwargs, path=path),
        character=char,
        name=name or model,
    )

  def _auto_agent(
      self,
      char: melee.Character,
      opponent_table: dict[melee.Character, str],
      agent_summaries: dict[str, eval_lib.AgentSummary],
  ) -> AutoAgent:
    delay = min(agent_summaries[model].delay
                for model in set(opponent_table.values()))
    return AutoAgent(
        character=char,
        models_path=self._models_path,
        opponent_table=opponent_table,
        delay=delay,
        agent_kwargs=self.agent_kwargs,
    )

  def _load_ratings(self) -> dict[tuple[str, melee.Character], float]:
    """Latest eval ratings keyed by (model, character); empty if unavailable."""
    if not self._eval_db_path or not os.path.exists(self._eval_db_path):
      return {}
    try:
      conn = eval_db_lib.connect_readonly(self._eval_db_path)
      try:
        by_name = eval_db_lib.latest_ratings_by_name(conn)
      finally:
        conn.close()
    except sqlite3.Error:
      logging.exception(f'Could not read ratings from {self._eval_db_path}')
      return {}
    return {
        (name, melee.Character[char]): rating + self._rating_offset
        for (name, char), rating in by_name.items()
    }

  def _refresh_loaded_models(self) -> int:
    """Loads models that are new or have changed since they were loaded.

    Models whose files have been removed are dropped from the in-memory (and
    on-disk) cache. Returns the number of models (re)loaded from disk, not
    counting those served from the on-disk cache.
    """
    num_loaded = 0
    num_cached = 0

    current_models = set(os.listdir(self._models_path))
    for model in list(self._loaded_models):
      if model not in current_models:
        logging.info(f'Model {model} removed from disk')
        del self._loaded_models[model]
    if self._model_cache is not None:
      self._model_cache.prune(current_models)

    for model in sorted(current_models):
      path = os.path.join(self._models_path, model)
      stat = os.stat(path)
      mtime = stat.st_mtime
      size = stat.st_size

      loaded = self._loaded_models.get(model)
      if loaded is not None and (loaded.mtime, loaded.size) == (mtime, size):
        continue

      state = None
      if self._model_cache is not None:
        state = self._model_cache.get(model, mtime, size)
        if state is not None:
          num_cached += 1

      if state is None:
        logging.info(f'Loading model {model}')
        state = trim_model_state(saving.load_state_from_disk(path))
        num_loaded += 1
        if self._model_cache is not None:
          self._model_cache.put(model, mtime, size, state)

      self._loaded_models[model] = LoadedModel(
          mtime=mtime,
          size=size,
          state=state,
          summary=eval_lib.AgentSummary.from_state(state),
      )

    if num_cached:
      logging.info(f'Loaded {num_cached} models from the cache.')

    return num_loaded

  def _reload_models(self) -> int:
    """Rebuilds the agent tables, loading only new or updated model files.

    Returns the number of models (re)loaded from disk.
    """
    num_loaded = self._refresh_loaded_models()

    self._special_agents: list[str] = []
    self._agents: dict[str, AgentConfig] = {}
    # Character played, opponents trained against, and special-agent group,
    # for filtering and grouping in find_agents.
    self._agent_infos: dict[str, AgentInfo] = {}

    # For inspection by model_config.
    self._model_configs: dict[str, dict] = {
        model: loaded.state for model, loaded in self._loaded_models.items()
    }
    summaries: dict[str, eval_lib.AgentSummary] = {
        model: loaded.summary for model, loaded in self._loaded_models.items()
    }

    def add_agent(
        agent_config: AgentConfig,
        char: melee.Character,
        opponents: tp.Iterable[melee.Character],
        group: Optional[str] = None,
        imitation: bool = False,
    ):
      if agent_config.name in self._agents:
        logging.warning(f'Duplicate agents named {agent_config.name}')
      self._agents[agent_config.name] = agent_config
      self._agent_infos[agent_config.name] = AgentInfo(
          character=char, opponents=frozenset(opponents), group=group,
          imitation=imitation)

    # (group, character name) pairs for the special agents.
    special_pairs: list[tuple[str, str]] = []

    # Regular agents
    for model, summary in summaries.items():
      imitation = summary.type is eval_lib.AgentType.IMITATION
      if len(summary.characters) == 1:
        add_agent(
            self._single_agent(model=model),
            char=summary.characters[0],
            opponents=summary.opponents,
            imitation=imitation)
      else:
        for char in summary.characters:
          agent_config = self._single_agent(
              model=model, char=char,
              name=model + '-' + char.name.lower())
          add_agent(
              agent_config, char=char, opponents=summary.opponents,
              group=model, imitation=imitation)
          special_pairs.append((model, char.name.lower()))

    # imitation agents
    imitation_models = eval_lib.get_imitation_agents_from_summaries(
        summaries, delay=self._auto_delay)

    imitation_group = imitation_prefix.rstrip('-')
    for char, model in imitation_models.items():
      agent_config = self._single_agent(
          model=model,
          char=char,
          name=imitation_prefix + char.name.lower(),
      )
      add_agent(
          agent_config, char=char, opponents=summaries[model].opponents,
          group=imitation_group, imitation=True)
      special_pairs.append((imitation_group, char.name.lower()))

    self._special_agents.extend(group_special_agents(special_pairs))

    # auto agents
    self._auto_agents: dict[melee.Character, AutoAgent] = {}
    # Eval ratings of the loaded models, for ratings, auto picks and agents.
    self._ratings: dict[tuple[str, melee.Character], float] = {
        key: rating for key, rating in self._load_ratings().items()
        if key[0] in summaries
    }
    matchup_table = eval_lib.build_matchup_table_from_ratings(
        summaries, self._ratings)
    if matchup_table:
      logging.info('Picked auto agents using eval ratings.')
    else:
      logging.warning(
          'No eval ratings; picking auto agents by training matchups.')
      matchup_table = eval_lib.build_matchup_table_from_summaries(
          summaries, delay=self._auto_delay)
    auto_names = []
    for character, opponent_table in matchup_table.items():
      agent_config = self._auto_agent(character, opponent_table, summaries)
      self._auto_agents[character] = agent_config
      add_agent(agent_config, char=character, opponents=opponent_table.keys())
      auto_names.append(character.name.lower())

    if len(auto_names) > 0:
      self._special_agents.append(
          f'{auto_prefix}[{",".join(auto_names)}]')

    # Set default agent
    if self._default_agent is not None:
      if self._default_agent not in self._agents:
        raise ValueError(f'Invalid default agent {self._default_agent}')
      self._default_agent_config = self._agents[self._default_agent]
    else:
      if melee.Character.FOX in self._auto_agents:
        default_char = melee.Character.FOX
      else:
        default_char = next(iter(self._auto_agents))

      self._default_agent_config = self._auto_agents[default_char]

    return num_loaded

  async def reload(self) -> int:
    """Returns the number of new or updated models."""
    async with self._lock:
      return self._reload_models()

  def agent_names(self) -> list[str]:
    return sorted(self._agents)

  def special_agents(self) -> list[str]:
    """The auto/basic/multi-character agents, grouped as [groups]-[chars]."""
    return sorted(self._special_agents)

  def find_agents(
      self,
      char: Optional[melee.Character] = None,
      opponent: Optional[melee.Character] = None,
  ) -> list[str]:
    """Agents that play char vs. opponent, either of which may be None.

    Special agents are grouped as "[groups]-[characters]". Auto agents are
    meta agents and raw imitation models are already listed as "basic", so
    both are excluded.
    """
    def matches(info: AgentInfo) -> bool:
      if char is not None and info.character != char:
        return False
      if opponent is not None and opponent not in info.opponents:
        return False
      return True

    special_pairs: list[tuple[str, str]] = []
    regular_names: list[str] = []
    for name, info in self._agent_infos.items():
      # Auto agents are meta agents; leave them out of filtered lists.
      if isinstance(self._agents[name], AutoAgent):
        continue
      if not matches(info):
        continue
      if info.group is not None:
        special_pairs.append((info.group, info.character.name.lower()))
      elif info.imitation:
        # Imitation models are already listed under the "basic" group.
        continue
      else:
        regular_names.append(name)

    return sorted(group_special_agents(special_pairs)) + sorted(regular_names)

  def agent_table(self) -> list[AgentRow]:
    """All agents except the multi-character models' per-character ones."""
    rows = []
    for name, agent_config in self._agents.items():
      info = self._agent_infos[name]
      if isinstance(agent_config, AutoAgent):
        rows.append(AgentRow(
            name=name, kind='auto', character=info.character,
            opponents=sorted(info.opponents, key=lambda c: c.name),
            delay=agent_config.delay, model=None, rating=None))
        continue

      assert isinstance(agent_config, SingleAgent)
      if info.imitation and info.group is None:
        # Imitation models are already listed as basic-<char>.
        continue

      model = os.path.basename(agent_config.agent_kwargs['path'])
      summary = self._loaded_models[model].summary
      rows.append(AgentRow(
          name=name,
          kind='basic' if name.startswith(imitation_prefix) else 'model',
          character=info.character,
          opponents=sorted(info.opponents, key=lambda c: c.name),
          delay=summary.delay,
          model=model,
          rating=self._ratings.get((model, info.character)),
      ))
    return rows

  def model_names(self) -> list[str]:
    return list(self._model_configs)

  def model_config(self, model: str) -> dict:
    if model not in self._model_configs:
      raise UserError(f'{model} is not a valid agent')
    return self._model_configs[model]

  def auto_agent_character(self, name: str) -> Optional[melee.Character]:
    """The character of an auto agent, or None if name isn't one."""
    agent_config = self._agents.get(name)
    if isinstance(agent_config, AutoAgent):
      return agent_config.character
    return None

  def _rating_rank(self, model: str, char: melee.Character) -> tuple[int, int]:
    """A rated model's rank among the models playing char, and their count."""
    rating = self._ratings[(model, char)]
    peers = [r for (_, c), r in self._ratings.items() if c == char]
    return 1 + sum(r > rating for r in peers), len(peers)

  def ratings(self, name: str) -> tuple[str, list[CharacterRating]]:
    """The model behind a (non-auto) agent and its ratings, maybe empty."""
    agent_config = self._agents.get(name)
    if isinstance(agent_config, AutoAgent):
      raise UserError(f'{name} picks a model per opponent.')

    if isinstance(agent_config, SingleAgent):
      model = os.path.basename(agent_config.agent_kwargs['path'])
      chars = [self._agent_infos[name].character]
    elif name in self._loaded_models:
      # A multi-character model, listed as <model>-<char>.
      model = name
      chars = self._loaded_models[name].summary.characters
    else:
      raise UserError(f'{name} is not a valid agent')

    ratings = []
    for char in chars:
      if (model, char) not in self._ratings:
        continue
      rank, count = self._rating_rank(model, char)
      ratings.append(CharacterRating(
          character=char, rating=self._ratings[(model, char)],
          rank=rank, count=count))
    return model, ratings

  def auto_pick(
      self, char: melee.Character, opponent: melee.Character) -> AutoPick:
    auto_agent = self._auto_agents.get(char)
    if auto_agent is None:
      raise UserError(f'There is no {auto_prefix}{char.name.lower()} agent.')

    model = auto_agent.opponent_table[opponent]
    summary = self._loaded_models[model].summary
    return AutoPick(
        agent=auto_agent.name,
        model=model,
        trained=opponent in summary.opponents,
        rating=self._ratings.get((model, char)),
    )

  # User preferences

  def _get_opponent_config(self, user: User) -> AgentConfig:
    agent_name = self._user_prefs.get(user.id, AGENT_PREF)
    if agent_name is None:
      return self._default_agent_config
    # Agents are looked up by name at use time so that the selection survives
    # restarts and reloads, even if the model has since gone away.
    agent_config = self._agents.get(agent_name)
    if agent_config is None:
      logging.warning(
          f'{user.id} selected agent {agent_name} which no longer exists.')
      return self._default_agent_config
    return agent_config

  def selected_agent(self, user: User) -> str:
    return self._get_opponent_config(user).name

  async def select_agent(self, user: User, agent_name: str) -> SelectResult:
    """Selects an agent, restarting the user's session if they are playing."""
    if agent_name not in self._agents:
      raise UserError(f'{agent_name} is not a valid agent')

    async with self._lock:
      self._user_prefs.set(user.id, AGENT_PREF, agent_name)

      if user.id not in self._sessions:
        return SelectResult()

      await self._stop_sessions([self._sessions[user.id]])
      return SelectResult(restarted=await self._play(user))

  def connect_code(self, user: User) -> Optional[str]:
    return self._user_prefs.get(user.id, CONNECT_CODE_PREF)

  def set_connect_code(self, user: User, connect_code: str):
    connect_code = connect_code.upper()
    if '#' not in connect_code:
      raise UserError(f'{connect_code} is not a valid connect code')
    self._user_prefs.set(user.id, CONNECT_CODE_PREF, connect_code)

  def stages(self, user: User) -> Optional[list[melee.Stage]]:
    """The user's stages, or None for all stages."""
    return self._stages.get(user.id)

  def set_stages(self, user: User, stages: Optional[list[melee.Stage]]):
    """Sets the user's stages; None means all stages."""
    if stages is None:
      self._stages.pop(user.id, None)
    else:
      self._stages[user.id] = stages

  def sd_stocks(self, user: User) -> Optional[int]:
    """Stocks the bot SDs down to, or None to play with all its stocks."""
    return self._sd_stocks.get(user.id)

  def set_sd_stocks(self, user: User, stocks: int):
    if not 1 <= stocks <= MAX_STOCKS:
      raise UserError(f'Stocks must be between 1 and {MAX_STOCKS}.')

    if stocks == MAX_STOCKS:
      self._sd_stocks.pop(user.id, None)
    else:
      self._sd_stocks[user.id] = stocks

  # Sessions

  async def play(self, user: User) -> PlayResult:
    """Has the bot connect to the user's saved connect code."""
    async with self._lock:
      return await self._play(user)

  async def _play(self, user: User) -> PlayResult:
    connect_code = self.connect_code(user)
    if connect_code is None:
      raise UserError('You must specify a connect code')

    if user.id in self._sessions:
      raise UserError(f'{user.name}, you are already playing')

    await self._gc_sessions()

    for info in self._sessions.values():
      if info.connect_code == connect_code:
        raise UserError(
            f'{connect_code} is already playing (as {info.user.name}).')

    if len(self._sessions) == self.max_sessions:
      raise UserError('Sorry, too many sessions already active.')

    agent_config = self._get_opponent_config(user)
    is_weak_agent = parse_imitation_char(agent_config.name) is not None

    is_stream = self._stream and (self._streaming_against is None)
    if is_weak_agent:
      is_stream = False

    if is_stream:
      await self._stop_bot_session()

    on_off = "on" if is_stream else "off"
    logging.info(f"Connecting to {user.id} ({connect_code}) {on_off} stream.")

    session = self._start_session(
        connect_code,
        agent_config=agent_config,
        render=is_stream,
        stages=self._stages.get(user.id, None),
        save_replays=not is_weak_agent,
        sd_stocks=self._sd_stocks.get(user.id, None),
    )
    self._sessions[user.id] = SessionInfo(
        id=self._new_session_id(),
        session=session,
        start_time=datetime.datetime.now(),
        user=user,
        connect_code=connect_code,
        agent=agent_config.name,
    )
    if is_stream:
      self._streaming_against = user

    return PlayResult(
        connect_code=connect_code,
        agent=agent_config.name,
        on_stream=is_stream,
    )

  async def stop(self, user: User):
    async with self._lock:
      if user.id not in self._sessions:
        raise UserError(f"{user.name}, you're not playing right now.")
      await self._stop_sessions([self._sessions[user.id]])

  async def reset(self, user: User) -> PlayResult:
    """Stops the user's session and starts a new one with the same agent."""
    async with self._lock:
      if user.id not in self._sessions:
        raise UserError(f"{user.name}, you're not playing right now.")
      await self._stop_sessions([self._sessions[user.id]])
      return await self._play(user)

  async def kick(self, name: str) -> User:
    """Stops the session of the user with the given name."""
    async with self._lock:
      for info in self._sessions.values():
        if info.user.name == name:
          await self._stop_sessions([info])
          return info.user
      raise UserError(f'"{name}" isn\'t playing right now')

  async def botmatch(
      self, user: User, agent1_name: str, agent2_name: str) -> str:
    """Starts an off-stream bot match; returns its label."""
    async with self._lock:
      agent1_config = self._agents.get(agent1_name)
      if agent1_config is None:
        raise UserError(f'Unknown agent: {agent1_name}')

      agent2_config = self._agents.get(agent2_name)
      if agent2_config is None:
        raise UserError(f'Unknown agent: {agent2_name}')

      if user.id in self._sessions:
        raise UserError(f'{user.name}, you already have an active session.')

      await self._gc_sessions()

      if len(self._sessions) == self.max_sessions:
        raise UserError('Sorry, too many sessions already active.')

      agent_configs = {1: agent1_config, 2: agent2_config}
      session = self._start_user_bot_session(
          agent_configs, max_games=self.botmatch_max_games)

      agent_label = f'{agent1_name} vs {agent2_name}'
      self._sessions[user.id] = SessionInfo(
          id=self._new_session_id(),
          session=session,
          start_time=datetime.datetime.now(),
          user=user,
          connect_code=None,
          agent=agent_label,
      )
      return agent_label

  def _new_session_id(self) -> int:
    self._next_session_id += 1
    return self._next_session_id

  def session_id(self, user: User) -> Optional[int]:
    """The id of the user's session, or None if they aren't playing."""
    info = self._sessions.get(user.id)
    return None if info is None else info.id

  async def session_logs(self, user: User, since: int) -> Optional[SessionLogs]:
    """The user's session's log lines from since on; None if not playing."""
    info = self._sessions.get(user.id)
    if info is None:
      return None
    next_line, lines = await info.session.logs.remote(since)
    return SessionLogs(session_id=info.id, next=next_line, lines=lines)

  def _start_session(
      self,
      connect_code: str,
      agent_config: AgentConfig,
      stages: Optional[list[melee.Stage]],
      render: bool = False,
      save_replays: bool = True,
      sd_stocks: Optional[int] = None,
  ) -> sessions_lib.Session:
    config = dataclasses.replace(self.dolphin_config)
    config.slippi_port = portpicker.pick_unused_port()
    config.connect_code = connect_code
    config.render = render
    config.headless = not render
    config.replay_dir = os.path.join(config.replay_dir, agent_config.name)
    os.makedirs(config.replay_dir, exist_ok=True)
    extra_dolphin_kwargs = {}
    if render:
      # TODO: don't hardcode this
      extra_dolphin_kwargs['env_vars'] = dict(DISPLAY=":99")

    config.save_replays = save_replays

    if self.run_on_cpu:
      session_class = sessions_lib.RemoteSession
    else:
      session_class = sessions_lib.RemoteGpuSession

    return session_class.remote(
        config,
        agent_config,
        extra_dolphin_kwargs=extra_dolphin_kwargs,
        stages=stages,
        run_on_cpu=self.run_on_cpu,
        costume=self._costume,
        sd_stocks=sd_stocks,
    )

  def _start_bot_session(self, render: bool = True) -> sessions_lib.BotSession:
    config = dataclasses.replace(self.dolphin_config)
    config.slippi_port = portpicker.pick_unused_port()
    config.connect_code = None
    config.render = render
    config.headless = not render
    config.save_replays = False
    extra_dolphin_kwargs = {}
    if render:
      config.emulation_speed = 1
      # TODO: don't hardcode this
      extra_dolphin_kwargs['env_vars'] = dict(DISPLAY=":99")

    if self.run_on_cpu:
      session_class = sessions_lib.RemoteBotSession
    else:
      session_class = sessions_lib.RemoteGpuBotSession

    return session_class.remote(
        config, self._bot_configs,
        extra_dolphin_kwargs=extra_dolphin_kwargs,
        run_on_cpu=self.run_on_cpu,
    )

  def _start_user_bot_session(
      self,
      agent_configs: dict[int, AgentConfig],
      max_games: int,
  ) -> sessions_lib.BotSession:
    config = dataclasses.replace(self.dolphin_config)
    config.slippi_port = portpicker.pick_unused_port()
    config.connect_code = None
    config.render = False
    config.headless = True
    config.save_replays = True

    sorted_names = sorted(ac.name for ac in agent_configs.values())
    config.replay_dir = os.path.join(
        self._botmatch_replay_dir,
        f'{sorted_names[0]}_vs_{sorted_names[1]}')
    config.replay_monthly_folders = False
    os.makedirs(config.replay_dir, exist_ok=True)

    config.online_delay = min(
        ac.target_console_delay for ac in agent_configs.values())

    if self.run_on_cpu:
      session_class = sessions_lib.RemoteBotSession
    else:
      session_class = sessions_lib.RemoteGpuBotSession

    return session_class.remote(
        config, agent_configs,
        run_on_cpu=self.run_on_cpu,
        max_games=max_games,
    )

  async def _stop_sessions(self, infos: list[SessionInfo]):
    """All (non-bot) session stops go through this method."""
    if not infos:
      return

    await asyncio.gather(*[info.session.stop.remote() for info in infos])

    logging.info(f'Stopped {len(infos)} human sessions.')

    for info in infos:
      del self._sessions[info.user.id]
      if self._streaming_against == info.user:
        self._streaming_against = None
        self._last_stream_time = time.time()

  # The bot session on stream

  def _bot_names(self) -> tuple[str, str]:
    return self._bot_configs[1].name, self._bot_configs[2].name

  async def _maybe_start_bot_session(self) -> bool:
    if self._streaming_against is not None:
      return False

    if self._bot_session is not None:
      return False

    if self._bot_session_interval < 0:
      return False

    if self._last_stream_time is not None:
      time_since_last_stream = time.time() - self._last_stream_time
      if time_since_last_stream < self._bot_session_interval * 60:
        return False

    self._bot_session = self._start_bot_session()
    logging.info('Started bot session.')
    bot1, bot2 = self._bot_names()
    await self._announce(f'Started {bot1} vs. {bot2} on stream.')
    return True

  async def start_bot_session(self) -> bool:
    async with self._lock:
      return await self._maybe_start_bot_session()

  async def _stop_bot_session(self):
    if not self._bot_session:
      return
    logging.info('Stopping bot session.')
    await self._bot_session.stop.remote()
    self._bot_session = None

  async def stop_bot_session(self) -> bool:
    """Returns whether a bot session was running."""
    async with self._lock:
      if not self._bot_session:
        return False
      await self._stop_bot_session()
      return True

  def _bots_lock_remaining(self) -> Optional[datetime.timedelta]:
    """Time left on the bots lock, or None if the bots aren't locked."""
    if self._bots_lock_until is None:
      return None
    remaining = self._bots_lock_until - datetime.datetime.now()
    if remaining <= datetime.timedelta(0):
      return None
    return remaining

  async def set_bots(
      self, user: User, bot1: str, bot2: str, is_mod: bool = False):
    """Sets the agents that play on stream while no one else is."""
    async with self._lock:
      if not is_mod and user != self._bots_set_by:
        remaining = self._bots_lock_remaining()
        if remaining is not None:
          raise UserError(
              f'{self._bots_set_by.name} has the bots for another'
              f' {format_td(remaining)} (or until they relinquish them).')

      bot_configs: dict[int, AgentConfig] = {}
      for port, name in {1: bot1, 2: bot2}.items():
        agent_config = self._agents.get(name)
        if agent_config is None:
          raise UserError(f'Unknown agent: {name}')
        bot_configs[port] = agent_config

      self._bot_configs = bot_configs
      now = datetime.datetime.now()
      # Only a new initiator gets a fresh timer. The same user setting the
      # bots again keeps their existing timer, and can't regain the lock once
      # it has expired (or been relinquished) by changing the matchup again;
      # someone else has to take the bots first.
      if user != self._bots_set_by:
        self._bots_set_by = user
        self._bots_lock_until = now + datetime.timedelta(
            minutes=self.bots_lock_minutes)
      self._bots_set_time = now

      await self._stop_bot_session()
      await self._maybe_start_bot_session()

  async def relinquish(self, user: User, is_mod: bool = False):
    async with self._lock:
      if self._bots_lock_remaining() is None:
        raise UserError('The bots aren\'t locked.')

      if not is_mod and user != self._bots_set_by:
        raise UserError(
            f'Only {self._bots_set_by.name} or a mod can relinquish the bots.')

      self._bots_lock_until = None

  # Status and chores

  async def status(self) -> ManagerStatus:
    # Doesn't take the lock, so that it doesn't wait behind slow stops.
    now = datetime.datetime.now()

    bots = None
    if self._bot_session:
      bot1, bot2 = self._bot_names()
      bots = BotsStatus(
          bot1=bot1,
          bot2=bot2,
          set_by=self._bots_set_by.name if self._bots_set_by else None,
          playing_for=(
              now - self._bots_set_time if self._bots_set_time else None),
          lock_remaining=self._bots_lock_remaining(),
      )

    async def get_status(
        info: SessionInfo) -> Optional[sessions_lib.SessionStatus]:
      # Sessions don't answer until they have loaded their agents.
      try:
        return await asyncio.wait_for(info.session.status.remote(), timeout=1)
      except asyncio.TimeoutError:
        return None

    infos = list(self._sessions.values())
    statuses = await asyncio.gather(*map(get_status, infos))

    sessions = []
    for info, status in zip(infos, statuses):
      sessions.append(SessionStatusInfo(
          session_id=info.id,
          user=info.user,
          agent=info.agent,
          duration=now - info.start_time,
          connect_code=info.connect_code,
          on_stream=info.user == self._streaming_against,
          menu_time=(
              None if status is None else
              datetime.timedelta(seconds=status.num_menu_frames / 60)),
          games_completed=None if status is None else status.games_completed,
      ))

    return ManagerStatus(bots=bots, sessions=sessions)

  async def _gc_sessions(self) -> list[User]:
    """Stop sessions that have died or been in the menu for too long."""
    if self._bot_session is not None:
      status: sessions_lib.SessionStatus = (
          await self._bot_session.status.remote())
      if not status.is_alive:
        logging.info('Bot session died.')
        await self._stop_bot_session()

    infos = list(self._sessions.values())
    statuses: list[sessions_lib.SessionStatus] = await asyncio.gather(
        *[info.session.status.remote() for info in infos])

    to_gc: list[SessionInfo] = []
    for info, status in zip(infos, statuses):
      menu_minutes = status.num_menu_frames / (60 * 60)
      if not status.is_alive or menu_minutes > self._menu_timeout:
        to_gc.append(info)

    await self._stop_sessions(to_gc)
    if to_gc:
      names = ", ".join([info.user.name for info in to_gc])
      logging.info(f'GCed sessions: {names}')
      await self._announce(f'Stopped idle session against {names}')
    return [info.user for info in to_gc]

  async def gc(self) -> list[User]:
    async with self._lock:
      return await self._gc_sessions()

  async def run_chores(self, interval: float = 60):
    """Runs forever, cleaning up sessions and starting the bot session."""
    while True:
      try:
        async with self._lock:
          await self._gc_sessions()
          await self._maybe_start_bot_session()
      except Exception:
        logging.exception('Chores failed')
      await asyncio.sleep(interval)

  def shutdown(self):
    """Stops all sessions. Blocks, so call it once the event loop is done."""
    logging.info('Shutting down')
    refs = [info.session.stop.remote() for info in self._sessions.values()]
    if self._bot_session is not None:
      refs.append(self._bot_session.stop.remote())
    if refs:
      ray.wait(refs, num_returns=len(refs))
    self._sessions.clear()
    self._bot_session = None

def format_td(td: datetime.timedelta) -> str:
  """Chop off microseconds."""
  return str(td).split('.')[0]
