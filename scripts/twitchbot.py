"""Bot that runs on twitch and lets people play against phillip 2."""

import abc
import dataclasses
import datetime
import functools
import json
import logging
import os
import random
import time
import threading
import typing as tp
from typing import Optional, Union

from absl import app, flags
import fancyflags as ff
from twitchio.ext import commands, routines
import portpicker
import ray

import melee

from slippi_ai import train_lib
from slippi_ai import saving
from slippi_ai import flag_utils, eval_lib
from slippi_ai import dolphin as dolphin_lib

NAME_TO_STAGE = {
    'fd': melee.Stage.FINAL_DESTINATION,
    'fod': melee.Stage.FOUNTAIN_OF_DREAMS,
    'dl': melee.Stage.DREAMLAND,
    'ys': melee.Stage.YOSHIS_STORY,
    'ps': melee.Stage.POKEMON_STADIUM,
    'bf': melee.Stage.BATTLEFIELD,
    'all': melee.Stage.RANDOM_STAGE,
}

# Twitch settings
default_access_token = os.environ.get('TWITCHBOT_ACCESS_TOKEN')
ACCESS_TOKEN = flags.DEFINE_string(
    'token', default_access_token, 'Access token for the twitch bot.',
    required=default_access_token is None)
CHANNEL = flags.DEFINE_string('channel', 'x_pilot', 'twitch channel')

STREAM = flags.DEFINE_boolean('stream', True, 'Stream one of the sessions.')

BOT_SESSION_INTERVAL = flags.DEFINE_float(
    'bot_session_interval', 1,
    'minutes before starting a bot session, negative means disabled')

# Bot settings
_DOLPHIN_CONFIG = dolphin_lib.DolphinConfig(
    infinite_time=False,
    save_replays=True,
    replay_dir='Replays/Twitchbot',
    console_timeout=120,  # allow opponent to pause
)
# STAGES = flags.DEFINE_multi_enum(
#     'stage', ['all'], stages.keys(), 'Stages to play on.')
DOLPHIN = ff.DEFINE_dict(
    'dolphin', **flag_utils.get_flags_from_default(_DOLPHIN_CONFIG))

MODELS_PATH = flags.DEFINE_string('models', 'pickled_models', 'Path to models')

MAX_SESSIONS = flags.DEFINE_integer('max_sessions', 4, 'Maximum number of concurrent sessions.')

# Serves as the default agent people play against, and the "screensaver" agent.
agent_flags = eval_lib.AGENT_FLAGS.copy()
agent_flags.update(
    async_inference=ff.Boolean(True),
    jit_compile=ff.Boolean(False),
)
AGENT = ff.DEFINE_dict('agent', **agent_flags)

BOT = flags.DEFINE_string('bot', None, 'Screensaver agent.')
BOT2 = flags.DEFINE_string('bot2', None, 'Second screensaver agent.')

MEDIUM_AGENT = flags.DEFINE_string('medium_agent', None, 'Path to medium agent.')

@dataclasses.dataclass
class SessionStatus:
  num_menu_frames: int
  is_alive: bool

class BotSession:
  """Session between two bots playing locally."""

  def __init__(
      self,
      dolphin_config: dolphin_lib.DolphinConfig,
      agent_kwargs: dict[int, dict],
      extra_dolphin_kwargs: dict = {},
  ):
    eval_lib.disable_gpus()
    logging.basicConfig(level=logging.INFO)
    self.dolphin_config = dolphin_config
    self.stop_requested = threading.Event()

    ports = agent_kwargs.keys()
    dolphin_kwargs = dolphin_config.to_kwargs()
    dolphin_kwargs.update(extra_dolphin_kwargs)

    players = {p: dolphin_lib.AI() for p in ports}
    dolphin = dolphin_lib.Dolphin(
        players=players,
        **dolphin_kwargs,
    )

    agents: dict[int, eval_lib.Agent] = {}
    for port, opponent_port in zip(ports, reversed(ports)):
      agents[port] = eval_lib.build_agent(
          controller=dolphin.controllers[port],
          opponent_port=opponent_port,
          run_on_cpu=True,
          console_delay=dolphin.console.online_delay,
          **agent_kwargs[port],
      )

      eval_lib.update_character(players[port], agents[port].config)

    def run():
      # Don't block in the menu so that we can stop if asked to.
      gamestates = dolphin.iter_gamestates(skip_menu_frames=False)

      # Main loop
      for agent in agents.values():
        agent.start()
      try:
        while not self.stop_requested.is_set():
          gamestate = next(gamestates)
          if not dolphin_lib.is_menu_state(gamestate):
            for agent in agents.values():
              agent.step(gamestate)
      finally:
        for agent in agents.values():
          agent.stop()
        dolphin.stop()

    self._thread = threading.Thread(target=run)
    self._thread.start()

  def stop(self):
    self.stop_requested.set()
    self._thread.join()

  def status(self) -> SessionStatus:
    return SessionStatus(
        num_menu_frames=0,
        is_alive=self._thread.is_alive(),
    )


RemoteBotSession = ray.remote(BotSession)

def get_ports(gamestate: melee.GameState, display_name: str):
  name_to_port = {
      player.displayName: port for port, player in gamestate.players.items()
  }
  actual_port = name_to_port[display_name]
  ports = list(gamestate.players)
  ports.remove(actual_port)
  opponent_port = ports[0]
  return actual_port, opponent_port

# Depends on gecko code. TODO: configure
MAX_DOLPHIN_DELAY = 24

class AgentConfig(abc.ABC):

  @property
  @abc.abstractmethod
  def delay(self) -> int:
    """The agent's delay."""

  @property
  def console_delay(self) -> int:
    # Leave a local delay of 1 for async inference.
    target_delay = max(self.delay - 1, 0)
    return min(target_delay, MAX_DOLPHIN_DELAY)

  @abc.abstractmethod
  def build(
      self,
      controller: melee.Controller,
      # Actual (netplay) port may different from Controller's (local) port
      port: int,
      opponent_port: int,
  ) -> Union[eval_lib.Agent, eval_lib.EnsembleAgent]:
    """Builds an agent for the given port and opponent port."""

  @property
  @abc.abstractmethod
  def character(self) -> melee.Character:
    """The character associated with this agent."""

class SingleAgent(AgentConfig):

  def __init__(
      self,
      agent_kwargs: dict[str, tp.Any],
      character: Optional[melee.Character] = None,
  ):
    self.agent_kwargs = agent_kwargs
    self._loaded = False
    self._character = character

  def _load(self):
    # can't use functools.cache because it isn't serializable
    if self._loaded:
      return
    self.state = saving.load_state_from_disk(self.agent_kwargs['path'])
    self.config = flag_utils.dataclass_from_dict(
        train_lib.Config, self.state['config'])
    self._loaded = True

  @property
  def delay(self) -> int:
    self._load()
    return self.config.policy.delay

  def build(
      self,
      controller: melee.Controller,
      port: int,
      opponent_port: int,
  ) -> eval_lib.Agent:
    self._load()
    return eval_lib.build_agent(
        controller=controller,
        port=port,
        opponent_port=opponent_port,
        console_delay=self.console_delay,
        run_on_cpu=True,
        state=self.state,
        **self.agent_kwargs,
    )

  @property
  def character(self) -> melee.Character:
    self._load()
    chars = eval_lib.allowed_characters(self.state['config'])
    if chars is None:
      return self._character or melee.Character.FOX

    if self._character is not None:
      assert self._character in chars
      return self._character

    return chars[0]  # Default to the first character in the list.

class AutoAgent(AgentConfig):

  def __init__(
      self,
      character: melee.Character,
      models_path: str,
      delay: int = 18,
      agent_kwargs: dict = {},
  ):
    self._character = character
    self._delay = delay
    self.models_path = models_path
    self.agent_kwargs = agent_kwargs

  @property
  def character(self) -> melee.Character:
    return self._character

  @property
  def delay(self) -> int:
    return self._delay

  def build(
      self,
      controller: melee.Controller,
      port: int,
      opponent_port: int,
  ) -> eval_lib.EnsembleAgent:
    return eval_lib.EnsembleAgent(
        character=self.character,
        delay=self.delay,
        models_path=self.models_path,
        port=port,
        opponent_port=opponent_port,
        controller=controller,
        console_delay=self.console_delay,
        run_on_cpu=True,
        **self.agent_kwargs,
    )

class Session:

  def __init__(
      self,
      dolphin_config: dolphin_lib.DolphinConfig,
      agent_config: AgentConfig,
      extra_dolphin_kwargs: dict = {},
      stages: Optional[list[melee.Stage]] = None,
  ):
    eval_lib.disable_gpus()
    self.dolphin_config = dolphin_config
    self.agent_config = agent_config
    self.stop_requested = threading.Event()
    stages = stages or [dolphin_config.stage]

    dolphin_config.online_delay = agent_config.console_delay
    logging.info(f'Setting console delay to {agent_config.console_delay}')

    port = 1
    dolphin_kwargs = dolphin_config.to_kwargs()
    dolphin_kwargs.update(extra_dolphin_kwargs)

    player = dolphin_lib.AI(character=agent_config.character)
    dolphin = dolphin_lib.Dolphin(
        players={port: player},
        **dolphin_kwargs,
    )
    controller = dolphin.controllers[port]

    agent = agent_config.build(
        controller=controller,
        # Ports will be set once we're in game.
        port=0, opponent_port=0,
    )

    # In netplay the ports are going to be randomized. We use the displayName
    # to figure out which port we actually are. The connect code would be better
    # but doesn't appear to work :(
    with open(dolphin_config.user_json_path) as f:
      user_json = json.load(f)
    display_name = user_json['displayName']

    self._num_menu_frames = 0

    def run():
      # Don't block in the menu so that we can stop if asked to.
      gamestates = dolphin.iter_gamestates(skip_menu_frames=False)

      # Main loop
      agent.start()
      try:
        while not self.stop_requested.is_set():
          gamestate = next(gamestates)

          if not dolphin_lib.is_menu_state(gamestate):
            # Ports may change if opponent disconnects and reconnects.
            if gamestate.frame == -123:
              agent.set_ports(*get_ports(gamestate, display_name))

            agent.step(gamestate)
            self._num_menu_frames = 0
          else:
            controller.release_all()

            if self._num_menu_frames == 0:
              dolphin.stage = random.choice(stages)
              logging.info(f'Setting stage to {dolphin.stage.name}')

            self._num_menu_frames += 1
      finally:
        agent.stop()
        dolphin.stop()

    self._thread = threading.Thread(target=run)
    self._thread.start()

  def num_menu_frames(self) -> int:
    return self._num_menu_frames

  def status(self) -> SessionStatus:
    # Return current_model from EnsembleAgent
    return SessionStatus(
        num_menu_frames=self._num_menu_frames,
        is_alive=self._thread.is_alive(),
    )

  def stop(self):
    self.stop_requested.set()
    self._thread.join()

RemoteSession = ray.remote(Session)

HELP_MESSAGE = """
!play <code>: Have the bot connect to you. Connect to the bot with code {bot_code}.
!agents[_full]: List available agents to play against. The auto-* agents will pick the strongest agent based the matchup. The basic-* agents are much weaker.
!agent <name>: Select an agent to play against.
!more: Show extra commands.
At most {max_players} players can be active at once, with one player on stream. If no one is playing, bots may be on stream.
""".strip()

EXTRA_HELP_MESSAGE = """
!status: Displays selected agent and current sessions.
!stop: Stop the bot after you are done. Doesn't work if the game is paused.
!reset: Stop the bot and start a new game with the same agent.
!stages: Specify a space-separated list of stages to play on.
!bots <agent1> [<agent2>]: Set one or two "screensaver" agents to play while no one is on stream.
!about: Print some info about the this AI.
"""

ABOUT_MESSAGE = """
Melee AI trained with a combination of imitation learning from slippi replays and self-play reinforcement learning.
 Replays are mostly from ranked, with some tournaments and personal dumps.
 Agents by default have a reaction time of 18 frames (300 ms).
 Agents cannot see Randall, FoD platforms, Nana, and items or projectiles.
 Code: https://github.com/vladfi1/slippi-ai. Discord: https://discord.gg/hfVTXGu.
""".strip()

@dataclasses.dataclass
class SessionInfo:
  session: Session  # actually a RemoteSession
  start_time: datetime.datetime
  twitch_name: str
  connect_code: str
  agent: str

def format_td(td: datetime.timedelta) -> str:
  """Chop off microseconds."""
  return str(td).split('.')[0]


auto_prefix = 'auto-'
imitation_prefix = 'basic-'
medium_prefix = 'medium-'

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

def parse_medium_char(agent: str) -> Optional[melee.Character]:
  if not agent.startswith(medium_prefix):
    return None

  char_str = agent.removeprefix(medium_prefix)
  return eval_lib.data.name_to_character.get(char_str)

def tokens(message: str) -> list[str]:
  return [x for x in message.split(' ') if x != '']

class Bot(commands.Bot):

  def __init__(
      self, token: str, prefix: str, channel: str,
      dolphin_config: dolphin_lib.DolphinConfig,
      agent_kwargs: dict,
      models_path: str,
      max_sessions: int = 4,  # Includes stream session.
      menu_timeout: float = 3,  # in minutes
      stream: bool = True,
      bot_session_interval: float = 1, # in minutes
      bot: Optional[str] = None,
      bot2: Optional[str] = None,
      auto_delay: int = 18,
      medium_agent: Optional[str] = None,
  ):
    super().__init__(token=token, prefix=prefix, initial_channels=[channel])
    self.owner = channel

    self.dolphin_config = dolphin_config
    self.agent_kwargs = agent_kwargs
    self._max_sessions = max_sessions
    self._menu_timeout = menu_timeout
    self._stream = stream
    self._bot_session_interval = bot_session_interval

    bot1 = bot or agent_kwargs['path']
    bot2 = bot2 or bot1
    self._bot_agent_kwargs = {
        1: dict(agent_kwargs, path=bot1),
        2: dict(agent_kwargs, path=bot2),
    }
    self._bot1 = bot1
    self._bot2 = bot2

    self._auto_delay = auto_delay

    self._sessions: dict[str, SessionInfo] = {}
    self._streaming_against: Optional[str] = None
    self._last_stream_time: Optional[float] = None
    self._bot_session: Optional[BotSession] = None  # actually RemoteBotSession

    self.lock = threading.RLock()

    with open(dolphin_config.user_json_path) as f:
      user_json = json.load(f)

    self.help_message = HELP_MESSAGE.format(
        max_players=max_sessions,
        bot_code=user_json['connectCode'],
    )

    self._models_path = models_path
    self._medium_agent_path = medium_agent
    self._reload_models()

    self._default_agent_name = 'auto-fox'
    self._default_agent_config = self._auto_agent(melee.Character.FOX)

    # User-specific state.
    self._requested_agents: dict[str, str] = {}
    self._requested_agent_configs: dict[str, AgentConfig] = {}
    self._play_codes: dict[str, str] = {}
    self._stages: dict[str, list[melee.Stage]] = {}

    self._do_chores.start()

  def _auto_agent(self, char: melee.Character) -> AgentConfig:
    return AutoAgent(
        character=char,
        models_path=self._models_path,
        delay=self._auto_delay,
        agent_kwargs=self.agent_kwargs,
    )

  @commands.command()
  async def help(self, ctx: commands.Context):
    for line in self.help_message.split('\n'):
      await ctx.send(line)

  @commands.command()
  async def more(self, ctx: commands.Context):
    for line in EXTRA_HELP_MESSAGE.split('\n'):
      await ctx.send(line)

  @commands.command()
  async def about(self, ctx: commands.Context):
    await ctx.send(ABOUT_MESSAGE)

  def _reload_models(self):
    self._models: dict[str, dict] = {}
    agents = os.listdir(self._models_path)

    keys = ['step', 'config', 'rl_config', 'agent_config']

    for agent in agents:
      path = os.path.join(self._models_path, agent)
      state = saving.load_state_from_disk(path)
      state = {k: state[k] for k in keys if k in state}
      self._models[agent] = state

    self._imitation_agents = eval_lib.get_imitation_agents(
        self._models_path, delay=self._auto_delay)

    self._matchup_table = eval_lib.build_matchup_table(
        self._models_path, delay=self._auto_delay)

    if self._medium_agent_path is not None:
      self._medium_agent_summary = eval_lib.AgentSummary.from_checkpoint(
          self._medium_agent_path)
      # TODO: handle None = all agents
      self._medium_characters = self._medium_agent_summary.characters or []

      # Make medium the default agent if it exists.
      char = self._medium_characters[0]
      self._default_agent_name = medium_prefix + char.name.lower()
      self._default_agent_config = SingleAgent(
          dict(self.agent_kwargs, path=self._medium_agent_path),
          character=char,
      )

  @commands.command()
  async def reload(self, ctx: commands.Context):
    with self.lock:
      self._reload_models()
    await self.agents_full(ctx)

  @commands.command()
  async def config(self, ctx: commands.Context):
    words = ctx.message.content.split(' ')
    if len(words) != 2:
      await ctx.send('You must specify an agent.')
      return

    agent = words[1]
    if agent not in self._models:
      await ctx.send(f'{agent} is not a valid agent')
      models_str = ", ".join(self._models)
      await ctx.send(f'Available agents: {models_str}')
      return

    message = json.dumps(self._models[agent])
    chunk_size = 500

    for i in range(0, len(message), chunk_size):
      await ctx.send(message[i:i+chunk_size])

  @commands.command()
  async def agents(self, ctx: commands.Context):
    agents = [auto_prefix + c.name.lower() for c in self._matchup_table]
    for c in self._imitation_agents:
      agents.append(imitation_prefix + c.name.lower())

    if self._medium_agent_path is not None:
      for char in self._medium_characters:
        agents.append(medium_prefix + char.name.lower())

    await ctx.send(' '.join(agents))

  @commands.command()
  async def agents_full(self, ctx: commands.Context):
    # agents = [auto_prefix + c.name.lower() for c in self._matchup_table]
    # for c in self._imitation_agents:
    #   agents.append(imitation_prefix + c.name.lower())
    # agents.extend(self._models)
    agents = self._models.keys()

    max_chars = 500
    chunks = [[]]
    chunk_size = 0
    for model in sorted(agents):
      chunk = chunks[-1]
      new_chunk_size = chunk_size + len(model) + 1
      if new_chunk_size > max_chars:
        chunk = []
        chunks.append(chunk)
        chunk_size = len(model)
      else:
        chunk_size = new_chunk_size
      chunk.append(model)

    for chunk in chunks:
      message = " ".join(chunk)
      assert len(message) <= max_chars
      await ctx.send(message)

  @commands.command()
  async def agent(self, ctx: commands.Context):
    words = ctx.message.content.split(' ')

    if len(words) != 2:
      await ctx.send('You must specify a single agent.')
      return

    agent: str = words[1]
    agent_config = None

    auto_char = parse_auto_char(agent)
    if auto_char:
      if auto_char not in self._matchup_table:
        valid_chars = ', '.join(c.name.lower() for c in self._matchup_table)
        await ctx.send(
            f'No agents trained to play as {auto_char.name.lower()}.'
            f' Available characters are {valid_chars}.')
        return

      agent_config = self._auto_agent(auto_char)

    imitation_char = parse_imitation_char(agent)
    if imitation_char:
      if imitation_char not in self._imitation_agents:
        valid_chars = ', '.join(c.name.lower() for c in self._imitation_agents)
        await ctx.send(
            f'No basic agents trained to play as {imitation_char.name.lower()}.'
            f' Available basic characters are {valid_chars}.')
        return

      model = self._imitation_agents[imitation_char]
      agent_config = SingleAgent(dict(
          self.agent_kwargs,
          path=os.path.join(self._models_path, model)
      ))

    medium_char = parse_medium_char(agent)
    if medium_char:
      if self._medium_agent_path is None:
        await ctx.send('No medium agent set.')
        return

      if medium_char not in self._medium_characters:
        valid_chars = ', '.join(
            c.name.lower() for c in self._medium_characters)
        await ctx.send(
            f'No medium agents trained to play as {medium_char.name.lower()}.'
            f' Available medium characters are {valid_chars}.')
        return

      agent_config = SingleAgent(
          agent_kwargs=dict(self.agent_kwargs, path=self._medium_agent_path),
          character=medium_char,
      )

    # Interpret agent a model in the models dir
    if agent_config is None:
      if agent not in self._models:
        await ctx.send(f'{agent} is not a valid agent')
        # models_str might be too big for Twitch :(
        # models_str = ", ".join(self._models)
        # await ctx.send(f'Available agents: {models_str}')
        return

      agent_config = SingleAgent(dict(
          self.agent_kwargs,
          path=os.path.join(self._models_path, agent),
      ))

    name = ctx.author.name
    assert isinstance(name, str)
    self._requested_agents[name] = agent
    self._requested_agent_configs[name] = agent_config
    await ctx.send(f'{name} has selected {agent}')

    # Auto-restart if the person is already playing
    if name in self._sessions:
      with self.lock:
        await ctx.send(f'Restarting session with {name}')
        self._stop_sessions([self._sessions[name]])
        await self._play(ctx)

  async def event_ready(self):
    # Notify us when everything is ready!
    # We are logged in and ready to chat and use commands...
    print(f'Logged in as | {self.nick}')
    print(f'User id is | {self.user_id}')

  @commands.command()
  async def stages(self, ctx: commands.Context):
    name = ctx.author.name
    assert isinstance(name, str)
    words = ctx.message.content.split(' ')

    stage_names = words[1:]
    if len(stage_names) == 0:
      stages = self._stages.get(name)
      if stages is None:
        stages_string = 'all'
      else:
        stages_string = ", ".join([s.name.lower() for s in stages])
      await ctx.send(
          f'{name} has selected {stages_string}.'
          f' Available stages: {", ".join(NAME_TO_STAGE)}')
      return

    stages: list[melee.Stage] = []
    for stage_name in stage_names:
      stage_name = stage_name.lower()
      if stage_name not in NAME_TO_STAGE:
          await ctx.send(
              f'{name}, {stage_name} is not a valid stage.'
              f' Available stages: {", ".join(NAME_TO_STAGE)}')
          return
      stages.append(NAME_TO_STAGE[stage_name])

    self._stages[name] = stages
    await ctx.send(f'{name} has set the stages to {", ".join(stage_names)}')

  @commands.command()
  async def stop(self, ctx: commands.Context):
    with self.lock:
      name = ctx.author.name
      if name not in self._sessions:
        await ctx.send(f"{name}, you're not playing right now.")
        return

      self._stop_sessions([self._sessions[name]])
      await ctx.send(f'Stopped playing against {name}')

  async def _play(self, ctx: commands.Context):
    with self.lock:
      name = ctx.author.name
      assert isinstance(name, str)
      connect_code = self._play_codes[name]
      assert connect_code

      if name in self._sessions:
        await ctx.send(f'{name}, you are already playing')
        return

      await self._gc_sessions()

      if len(self._sessions) == self._max_sessions:
        await ctx.send('Sorry, too many sessions already active.')
        return

      is_stream = self._stream and (self._streaming_against is None)
      if is_stream:
        self._stop_bot_session()

      on_off = "on" if is_stream else "off"
      message = f"Connecting to {name} ({connect_code}) {on_off} stream."
      logging.info(message)
      await ctx.send(message)

      agent = self._get_opponent(name)
      session = self._start_session(
          connect_code,
          agent_name=agent,
          agent_config=self._get_opponent_config(name),
          render=is_stream,
          stages=self._stages.get(name, None),
      )
      self._sessions[name] = SessionInfo(
          session=session,
          start_time=datetime.datetime.now(),
          twitch_name=name,
          connect_code=connect_code,
          agent=agent,
      )
      if is_stream:
        self._streaming_against = name

  @commands.command()
  async def play(self, ctx: commands.Context):
    name = ctx.author.name
    assert isinstance(name, str)
    words = ctx.message.content.split(' ')

    if len(words) == 1:
      connect_code = self._play_codes.get(name)
      if connect_code is None:
        await ctx.send('You must specify a connect code')
        return
    else:
      connect_code = words[1].upper()
      if '#' not in connect_code:
        await ctx.send(f'{connect_code} is not a valid connect code')
        return
      self._play_codes[name] = connect_code

    await self._play(ctx)

  @commands.command()
  async def reset(self, ctx: commands.Context):
    with self.lock:
      name = ctx.author.name

      if name not in self._sessions:
        await ctx.send(f"{name}, you're not playing right now.")
        return

      self._stop_sessions([self._sessions[name]])

      # Reusing the context here is a bit of a hack.
      await self._play(ctx)

  def _start_session(
      self,
      connect_code: str,
      agent_name: str,
      agent_config: AgentConfig,
      stages: Optional[list[melee.Stage]],
      render: bool = False,
  ) -> Session:
    config = dataclasses.replace(self.dolphin_config)
    config.slippi_port = portpicker.pick_unused_port()
    config.connect_code = connect_code
    config.render = render
    config.headless = not render
    config.replay_dir = os.path.join(config.replay_dir, agent_name)
    os.makedirs(config.replay_dir, exist_ok=True)
    extra_dolphin_kwargs = {}
    if render:
      # TODO: don't hardcode this
      extra_dolphin_kwargs['env_vars'] = dict(DISPLAY=":99")

    imitation_character = parse_imitation_char(agent_name)

    if imitation_character:
      config.save_replays = False

    return RemoteSession.remote(
        config,
        agent_config,
        extra_dolphin_kwargs=extra_dolphin_kwargs,
        stages=stages,
    )

  def _start_bot_session(self, render: bool = True) -> BotSession:
    config = dataclasses.replace(self.dolphin_config)
    config.slippi_port = portpicker.pick_unused_port()
    config.connect_code = None
    config.render = render
    config.headless = not render
    config.save_replays = False
    extra_dolphin_kwargs = {}
    if render:
      extra_dolphin_kwargs['emulation_speed'] = 1
      # TODO: don't hardcode this
      extra_dolphin_kwargs['env_vars'] = dict(DISPLAY=":99")

    return RemoteBotSession.remote(
        config, self._bot_agent_kwargs,
        extra_dolphin_kwargs=extra_dolphin_kwargs,
    )

  async def _maybe_start_bot_session(self) -> bool:
    with self.lock:
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
      chan = self.get_channel(self.owner)
      # Might be None if we haven't logged in yet
      if chan:
        bot1 = os.path.basename(self._bot1)
        bot2 = os.path.basename(self._bot2)
        await chan.send(f'Started {bot1} vs. {bot2} on stream.')
      return True

  @commands.command()
  async def start_bot_session(self, ctx: commands.Context):
    started = await self._maybe_start_bot_session()
    if started:
      await ctx.send('Started bot session on stream.')
    else:
      await ctx.send('Did not start bot session on stream.')

  @commands.command()
  async def bots(self, ctx: commands.Context):
    words = ctx.message.content.split(' ')[1:]

    if len(words) == 1:
      bot1 = bot2 = words[0]
    elif len(words) == 2:
      bot1, bot2 = words
    else:
      await ctx.send('Must specify one or two agent names')
      return

    for agent in (bot1, bot2):
      if agent not in self._models:
        await ctx.send(f'{agent} is not a valid agent')
        return

    with self.lock:
      self._bot1 = os.path.join(self._models_path, bot1)
      self._bot2 = os.path.join(self._models_path, bot2)
      self._bot_agent_kwargs[1]['path'] = self._bot1
      self._bot_agent_kwargs[2]['path'] = self._bot2

      self._stop_bot_session()
      await self._maybe_start_bot_session()

  def _get_opponent(self, name: str) -> str:
    return self._requested_agents.get(name, self._default_agent_name)

  def _get_opponent_config(self, name: str) -> AgentConfig:
    return self._requested_agent_configs.get(name, self._default_agent_config)

  @commands.command()
  async def status(self, ctx: commands.Context):
    with self.lock:

      agent_name = self._get_opponent(ctx.author.name)
      await ctx.send(f'Selected agent: {agent_name}')

      if self._bot_session:
        bot1 = os.path.basename(self._bot1)
        bot2 = os.path.basename(self._bot2)
        await ctx.send(f'{bot1} vs. {bot2} on stream.')

      if not self._sessions:
        await ctx.send('No active sessions.')
        return

      now = datetime.datetime.now()
      for session_info in self._sessions.values():
        timedelta = format_td(now - session_info.start_time)
        is_stream = session_info.twitch_name == self._streaming_against
        on_off = "on" if is_stream else "off"
        menu_frames = ray.get(session_info.session.num_menu_frames.remote())
        menu_time = format_td(datetime.timedelta(seconds=menu_frames / 60))
        player = session_info.twitch_name
        agent = session_info.agent
        await ctx.send(
            f'Playing against {player} as {agent} {on_off} stream'
            f' for {timedelta} (in menu for {menu_time}).')

  def _stop_sessions(self, infos: list[SessionInfo]):
    """All (non-bot) session stops go through this method."""
    with self.lock:
      ray.wait([info.session.stop.remote() for info in infos])

      logging.info(f'Stopped {len(infos)} human sessions.')

      for info in infos:
        del self._sessions[info.twitch_name]
        if self._streaming_against == info.twitch_name:
          self._streaming_against = None
          self._last_stream_time = time.time()

  def _stop_bot_session(self):
    with self.lock:
      if not self._bot_session:
        return
      logging.info('Stopping bot session.')
      ray.wait([self._bot_session.stop.remote()])
      self._bot_session = None

  @commands.command()
  async def stop_bot_session(self, ctx: commands.Context):
    if not self._bot_session:
      await ctx.send('Bot session not running.')
      return

    self._stop_bot_session()
    await ctx.send('Stopped bot session.')

  @commands.command()
  async def kick(self, ctx: commands.Context):
    if not ctx.author.is_mod:
      await ctx.send('Only mods can kick players.')
      return

    words = ctx.message.content.split(' ')[1:]
    if len(words) != 1:
      await ctx.send('Must specify a player to kick')
      return

    name = words[0]
    if name not in self._sessions:
      await ctx.send(f'"{name}" isn\'t playing right now')
      return

    self._stop_sessions([self._sessions[name]])
    await ctx.send(f'Kicked {name}')

  async def _gc_sessions(self) -> list[SessionInfo]:
    """Stop sessions that have died or been in the menu for too long."""
    with self.lock:
      if self._bot_session is not None:
        status: SessionStatus = ray.get(self._bot_session.status.remote())
        if not status.is_alive:
          logging.info('Bot session died.')
          self._stop_bot_session()

      to_gc: list[SessionInfo] = []
      for info in self._sessions.values():
        status: SessionStatus = ray.get(info.session.status.remote())
        menu_minutes = status.num_menu_frames / (60 * 60)
        if not status.is_alive or menu_minutes > self._menu_timeout:
          to_gc.append(info)

      self._stop_sessions(to_gc)
      if to_gc:
        names = ", ".join([info.twitch_name for info in to_gc])
        logging.info(f'GCed sessions: {names}')
        chan = self.get_channel(self.owner)
        await chan.send(f'Stopped idle session against {names}')
      return to_gc

  @commands.command()
  async def gc(self, ctx: commands.Context):
    infos = await self._gc_sessions()
    names = [info.twitch_name for info in infos]
    names = ", ".join(names)
    await ctx.send(f"GCed ({names})")

  @routines.routine(minutes=1)
  async def _do_chores(self):
    with self.lock:
      await self._gc_sessions()
      await self._maybe_start_bot_session()

  def shutdown(self):
    with self.lock:
      logging.info('Shutting down')
      self._stop_sessions(list(self._sessions.values()))
      self._stop_bot_session()
      self._do_chores.stop()

def main(_):
  eval_lib.disable_gpus()

  agent_kwargs = AGENT.value
  if not agent_kwargs['path']:
    raise ValueError('Must provide agent path.')

  bot = Bot(
      token=ACCESS_TOKEN.value,
      prefix='!',
      channel=CHANNEL.value,
      models_path=MODELS_PATH.value,
      max_sessions=MAX_SESSIONS.value,
      dolphin_config=flag_utils.dataclass_from_dict(
          dolphin_lib.DolphinConfig, DOLPHIN.value),
      agent_kwargs=agent_kwargs,
      stream=STREAM.value,
      bot_session_interval=BOT_SESSION_INTERVAL.value,
      bot=BOT.value,
      bot2=BOT2.value,
      medium_agent=MEDIUM_AGENT.value,
  )

  try:
    bot.run()
  finally:
    bot.shutdown()

if __name__ == '__main__':
    app.run(main)
