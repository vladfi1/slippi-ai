"""Bot that runs on twitch and lets people play against phillip 2."""

import abc
import dataclasses
import datetime
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
    jit_compile=ff.Boolean(True),
    run_on_cpu=ff.Boolean(False),
)
del agent_flags['path']  # not used
AGENT = ff.DEFINE_dict('agent', **agent_flags)


BOT = flags.DEFINE_string('bot', None, 'Path to first bot agent.')
BOT2 = flags.DEFINE_string('bot2', None, 'Path to second bot agent.')

DEFAULT_AGENT = flags.DEFINE_string('default_agent', None, 'Name of default agent.')

BOTMATCH_REPLAY_DIR = flags.DEFINE_string(
    'botmatch_replay_dir', 'Replays/BotMatch',
    'Base directory for user bot-match replays.')
BOTMATCH_MAX_GAMES = flags.DEFINE_integer(
    'botmatch_max_games', 25,
    'Max games per bot-match session before it auto-stops.')

@dataclasses.dataclass
class SessionStatus:
  num_menu_frames: int
  is_alive: bool
  games_completed: int = 0

Agent = Union[eval_lib.Agent, eval_lib.EnsembleAgent]

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
    self.config = flag_utils.dataclass_from_dict(
        train_lib.Config, self.state['config'])
    self._loaded = True

  @property
  def delay(self) -> int:
    self._load()
    return self.config.policy.delay

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
      delay: int = 18,
      agent_kwargs: dict = {},
  ):
    self._character = character
    self._delay = delay
    self.models_path = models_path
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
        character=self.character,
        delay=self.delay,
        models_path=self.models_path,
        port=port,
        opponent_port=opponent_port,
        controller=controller,
        console_delay=console_delay,
        **self.agent_kwargs,
    )

class BotSession:
  """Session between two bots playing locally."""

  def __init__(
      self,
      dolphin_config: dolphin_lib.DolphinConfig,
      agent_configs: dict[int, AgentConfig],
      extra_dolphin_kwargs: dict = {},
      run_on_cpu: bool = False,
      max_games: Optional[int] = None,
  ):
    logging.basicConfig(level=logging.INFO)

    if run_on_cpu:
      eval_lib.disable_gpus()
    else:
      set_memory_growth()

    self.dolphin_config = dolphin_config
    self.stop_requested = threading.Event()
    self._games_completed = 0

    ports = agent_configs.keys()
    dolphin_kwargs = dolphin_config.to_kwargs()
    dolphin_kwargs.update(extra_dolphin_kwargs)

    players = {
        port: dolphin_lib.AI(character=config.character)
        for port, config in agent_configs.items()
    }
    dolphin = dolphin_lib.Dolphin(
        players=players,
        **dolphin_kwargs,
    )

    agents: dict[int, Agent] = {}
    for port, opponent_port in zip(ports, reversed(ports)):
      agents[port] = agent_configs[port].build(
          controller=dolphin.controllers[port],
          port=port,
          opponent_port=opponent_port,
          console_delay=dolphin.console.online_delay,
      )

    def run():
      # Don't block in the menu so that we can stop if asked to.
      gamestates = dolphin.iter_gamestates(skip_menu_frames=False)

      # Main loop
      for agent in agents.values():
        agent.start()
      try:
        was_in_game = False
        while not self.stop_requested.is_set():
          gamestate = next(gamestates)
          in_game = not dolphin_lib.is_menu_state(gamestate)
          if in_game:
            for agent in agents.values():
              agent.step(gamestate)
          # Detect game-end transition: was in game, now in menu.
          if was_in_game and not in_game:
            self._games_completed += 1
            if max_games is not None:
              logging.info(f'Bot session: game {self._games_completed} completed.')
              if self._games_completed >= max_games:
                logging.info(f'Bot session: reached max_games={max_games}, stopping.')
                break
          was_in_game = in_game
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
        games_completed=self._games_completed,
    )

RemoteBotSession = ray.remote(num_gpus=0)(BotSession)
RemoteGpuBotSession = ray.remote(num_gpus=0.1)(BotSession)

def get_ports(gamestate: melee.GameState, display_name: str):
  name_to_port = {
      player.displayName: port for port, player in gamestate.players.items()
  }
  if display_name not in name_to_port:
    raise ValueError(f'Could not find display name {display_name} in {name_to_port}')
  actual_port = name_to_port[display_name]
  ports = list(gamestate.players)
  ports.remove(actual_port)
  opponent_port = ports[0]
  return actual_port, opponent_port


class Session:

  def __init__(
      self,
      dolphin_config: dolphin_lib.DolphinConfig,
      agent_config: AgentConfig,
      extra_dolphin_kwargs: dict = {},
      stages: Optional[list[melee.Stage]] = None,
      run_on_cpu: bool = False,
  ):
    if run_on_cpu:
      eval_lib.disable_gpus()
    else:
      set_memory_growth()

    self.dolphin_config = dolphin_config
    self.agent_config = agent_config
    self.stop_requested = threading.Event()
    stages = stages or [dolphin_config.stage]

    dolphin_config.online_delay = agent_config.target_console_delay
    logging.info(f'Setting console delay to {agent_config.target_console_delay}')

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
        console_delay=agent_config.target_console_delay,
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

RemoteSession = ray.remote(num_gpus=0)(Session)
RemoteGpuSession = ray.remote(num_gpus=0.1)(Session)

HELP_MESSAGE = """
!play <code>: Have the bot connect to you. Connect to the bot with code {bot_code}.
!agents[_full]: List available agents to play against. The auto-* agents will pick the strongest agent based the matchup. The basic-* agents are much weaker, and the medium-* agents are in the middle.
!agent <name>: Select an agent to play against.
!botmatch <agent1> [<agent2>]: Start an off-stream bot-vs-bot match ({botmatch_max_games} games).
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
  session: Session  # actually a RemoteSession or RemoteBotSession
  start_time: datetime.datetime
  twitch_name: str
  connect_code: Optional[str]
  agent: str

def format_td(td: datetime.timedelta) -> str:
  """Chop off microseconds."""
  return str(td).split('.')[0]


auto_prefix = 'auto-'
imitation_prefix = 'basic-'

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

def tokens(message: str) -> list[str]:
  return [x for x in message.split(' ') if x != '']

async def send_list(ctx: commands.Context, items: list[str]):
  max_chars = 500
  chunks = [[]]
  chunk_size = 0
  for name in items:
    chunk = chunks[-1]
    new_chunk_size = chunk_size + len(name) + 1
    if new_chunk_size > max_chars:
      chunk = []
      chunks.append(chunk)
      chunk_size = len(name)
    else:
      chunk_size = new_chunk_size
    chunk.append(name)

  for chunk in chunks:
    message = " ".join(chunk)
    assert len(message) <= max_chars
    await ctx.send(message)


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
      default_agent: Optional[str] = None,
      botmatch_replay_dir: str = 'Replays/BotMatch',
      botmatch_max_games: int = 10,
  ):
    super().__init__(token=token, prefix=prefix, initial_channels=[channel])
    self.owner = channel

    self.dolphin_config = dolphin_config
    self.agent_kwargs = agent_kwargs
    self.run_on_cpu = agent_kwargs['run_on_cpu']

    if self.run_on_cpu:
      eval_lib.disable_gpus()
    else:
      set_memory_growth()

    self._max_sessions = max_sessions
    self._menu_timeout = menu_timeout
    self._stream = stream
    self._bot_session_interval = bot_session_interval

    self._auto_delay = auto_delay
    self._botmatch_replay_dir = botmatch_replay_dir
    self._botmatch_max_games = botmatch_max_games

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
        botmatch_max_games=botmatch_max_games,
    )

    self._models_path = models_path
    self._default_agent = default_agent
    self._reload_models()

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

    # User-specific state.
    self._requested_agent_configs: dict[str, AgentConfig] = {}
    self._play_codes: dict[str, str] = {}
    self._stages: dict[str, list[melee.Stage]] = {}

    self._do_chores.start()

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

  def _auto_agent(self, char: melee.Character) -> AutoAgent:
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
    self._special_agents: list[str] = []
    self._agents: dict[str, AgentConfig] = {}

    # For inspection by the !config command
    self._model_configs: dict[str, dict] = {}
    keys = ['step', 'config', 'rl_config', 'agent_config', 'opponent']

    def add_agent(agent_config: AgentConfig):
      if agent_config.name in self._agents:
        logging.warning(f'Duplicate agents named {agent_config.name}')
      self._agents[agent_config.name] = agent_config

    regular_multiname_agents: list[tuple[str, list[str]]] = []

    # Regular agents
    for model in os.listdir(self._models_path):
      path = os.path.join(self._models_path, model)
      state = saving.load_state_from_disk(path)
      state = {k: state[k] for k in keys if k in state}
      self._model_configs[model] = state

      summary = eval_lib.AgentSummary.from_state(state)
      if len(summary.characters) == 1:
        add_agent(self._single_agent(model=model))
      else:
        char_names = []
        for char in summary.characters:
          char_names.append(char.name.lower())
          agent_config = self._single_agent(
              model=model, char=char,
              name=model + '-' + char.name.lower())
          add_agent(agent_config)

        regular_multiname_agents.append((model, char_names))

    models_by_chars: dict[tuple[str, ...], list[str]] = {}
    for model, char_names in regular_multiname_agents:
      key = tuple(sorted(char_names))
      models_by_chars.setdefault(key, []).append(model)

    def names_to_str(names: tp.Sequence[str]) -> str:
      if len(names) > 1:
        return "[" + ",".join(names) + "]"
      return names[0]

    for char_tuple, models in models_by_chars.items():
      self._special_agents.append(
          names_to_str(models) + '-' + names_to_str(char_tuple))

    # imitation agents
    imitation_models = eval_lib.get_imitation_agents(
        self._models_path, delay=self._auto_delay)

    imitation_names = []
    for char, model in imitation_models.items():
      agent_config = self._single_agent(
          model=model,
          char=char,
          name=imitation_prefix + char.name.lower(),
      )
      add_agent(agent_config)
      imitation_names.append(char.name.lower())

    if len(imitation_names) > 0:
      self._special_agents.append(
          f'{imitation_prefix}[{",".join(imitation_names)}]')

    # auto agents
    self._auto_agents: dict[melee.Character, AutoAgent] = {}
    matchup_table = eval_lib.build_matchup_table(
        self._models_path, delay=self._auto_delay)
    auto_names = []
    for character in matchup_table:
      agent_config = self._auto_agent(character)
      self._auto_agents[character] = agent_config
      add_agent(agent_config)
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
    if agent not in self._model_configs:
      await ctx.send(f'{agent} is not a valid agent')
      models_str = ", ".join(self._model_configs)
      await ctx.send(f'Available agents: {models_str}')
      return

    message = json.dumps(self._model_configs[agent])
    chunk_size = 500

    for i in range(0, len(message), chunk_size):
      await ctx.send(message[i:i+chunk_size])

  @commands.command()
  async def agents(self, ctx: commands.Context):
    await send_list(ctx, sorted(self._special_agents))

  @commands.command()
  async def agents_full(self, ctx: commands.Context):
    await send_list(ctx, sorted(self._agents.keys()))

  @commands.command()
  async def agent(self, ctx: commands.Context):
    words = ctx.message.content.split(' ')

    if len(words) != 2:
      await ctx.send('You must specify a single agent.')
      return

    agent_name: str = words[1]

    agent_config = self._agents.get(agent_name)

    if agent_config is None:
      await ctx.send(f'{agent_name} is not a valid agent')
      return

    name = ctx.author.name
    assert isinstance(name, str)
    self._requested_agent_configs[name] = agent_config
    await ctx.send(f'{name} has selected {agent_name}')

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

      agent_config = self._get_opponent_config(name)
      is_weak_agent = parse_imitation_char(agent_config.name) is not None

      is_stream = self._stream and (self._streaming_against is None)
      if is_weak_agent:
        is_stream = False

      if is_stream:
        self._stop_bot_session()

      on_off = "on" if is_stream else "off"
      message = f"Connecting to {name} ({connect_code}) {on_off} stream."
      logging.info(message)
      await ctx.send(message)

      session = self._start_session(
          connect_code,
          agent_config=agent_config,
          render=is_stream,
          stages=self._stages.get(name, None),
          save_replays=not is_weak_agent,
      )
      self._sessions[name] = SessionInfo(
          session=session,
          start_time=datetime.datetime.now(),
          twitch_name=name,
          connect_code=connect_code,
          agent=agent_config.name,
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
  async def botmatch(self, ctx: commands.Context):
    with self.lock:
      name = ctx.author.name
      assert isinstance(name, str)
      words = ctx.message.content.split(' ')[1:]

      if len(words) == 1:
        agent1_name = agent2_name = words[0]
      elif len(words) == 2:
        agent1_name, agent2_name = words
      else:
        await ctx.send('Usage: !botmatch agent1 [agent2]')
        return

      agent1_config = self._agents.get(agent1_name)
      if agent1_config is None:
        await ctx.send(f'Unknown agent: {agent1_name}')
        return

      agent2_config = self._agents.get(agent2_name)
      if agent2_config is None:
        await ctx.send(f'Unknown agent: {agent2_name}')
        return

      if name in self._sessions:
        await ctx.send(f'{name}, you already have an active session.')
        return

      await self._gc_sessions()

      if len(self._sessions) == self._max_sessions:
        await ctx.send('Sorry, too many sessions already active.')
        return

      agent_configs = {1: agent1_config, 2: agent2_config}
      session = self._start_user_bot_session(
          agent_configs, max_games=self._botmatch_max_games)

      agent_label = f'{agent1_name} vs {agent2_name}'
      self._sessions[name] = SessionInfo(
          session=session,
          start_time=datetime.datetime.now(),
          twitch_name=name,
          connect_code=None,
          agent=agent_label,
      )
      await ctx.send(
          f'Started bot match: {agent_label}'
          f' ({self._botmatch_max_games} games).')

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
      agent_config: AgentConfig,
      stages: Optional[list[melee.Stage]],
      render: bool = False,
      save_replays: bool = True,
  ) -> Session:
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

    session_class = RemoteSession if self.run_on_cpu else RemoteGpuSession

    return session_class.remote(
        config,
        agent_config,
        extra_dolphin_kwargs=extra_dolphin_kwargs,
        stages=stages,
        run_on_cpu=self.run_on_cpu,
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
      config.emulation_speed = 1
      # TODO: don't hardcode this
      extra_dolphin_kwargs['env_vars'] = dict(DISPLAY=":99")

    session_class = RemoteBotSession if self.run_on_cpu else RemoteGpuBotSession

    return session_class.remote(
        config, self._bot_configs,
        extra_dolphin_kwargs=extra_dolphin_kwargs,
        run_on_cpu=self.run_on_cpu,
    )

  def _start_user_bot_session(
      self,
      agent_configs: dict[int, AgentConfig],
      max_games: int,
  ) -> BotSession:
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

    session_class = RemoteBotSession if self.run_on_cpu else RemoteGpuBotSession

    return session_class.remote(
        config, agent_configs,
        run_on_cpu=self.run_on_cpu,
        max_games=max_games,
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
        bot1 = self._bot_configs[1].name
        bot2 = self._bot_configs[2].name
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

    bot_names = {
        1: bot1,
        2: bot2,
    }

    with self.lock:
      bot_configs: dict[int, AgentConfig] = {}
      for port, name in bot_names.items():
        agent_config = self._agents.get(name)
        if agent_config is None:
          await ctx.send(f'Unknown agent: {name}')
          return
        bot_configs[port] = agent_config

      self._bot_configs = bot_configs

      self._stop_bot_session()
      await self._maybe_start_bot_session()

  def _get_opponent_config(self, name: str) -> AgentConfig:
    return self._requested_agent_configs.get(name, self._default_agent_config)

  @commands.command()
  async def status(self, ctx: commands.Context):
    with self.lock:

      agent_config = self._get_opponent_config(ctx.author.name)
      await ctx.send(f'Selected agent: {agent_config.name}')

      if self._bot_session:
        bot1 = self._bot_configs[1].name
        bot2 = self._bot_configs[2].name
        await ctx.send(f'{bot1} vs. {bot2} on stream.')

      if not self._sessions:
        await ctx.send('No active sessions.')
        return

      now = datetime.datetime.now()
      for session_info in self._sessions.values():
        timedelta = format_td(now - session_info.start_time)
        player = session_info.twitch_name
        agent = session_info.agent

        if session_info.connect_code is None:
          # Bot-match session
          status: SessionStatus = ray.get(session_info.session.status.remote())
          await ctx.send(
              f'Bot match: {agent} for {timedelta},'
              f' {status.games_completed} games completed'
              f' (requested by {player}).')
        else:
          is_stream = player == self._streaming_against
          on_off = "on" if is_stream else "off"
          menu_frames = ray.get(session_info.session.num_menu_frames.remote())
          menu_time = format_td(datetime.timedelta(seconds=menu_frames / 60))
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

def set_memory_growth():
  """Set memory growth for all available GPUs."""
  import tensorflow as tf

  gpus = tf.config.experimental.list_physical_devices('GPU')

  for gpu in gpus:
    try:
      tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      logging.warning(f'Could not set memory growth for GPU {gpu}: {e}')

def main(_):
  bot = Bot(
      token=ACCESS_TOKEN.value,
      prefix='!',
      channel=CHANNEL.value,
      models_path=MODELS_PATH.value,
      max_sessions=MAX_SESSIONS.value,
      dolphin_config=flag_utils.dataclass_from_dict(
          dolphin_lib.DolphinConfig, DOLPHIN.value),
      agent_kwargs=AGENT.value,
      stream=STREAM.value,
      bot_session_interval=BOT_SESSION_INTERVAL.value,
      bot=BOT.value,
      bot2=BOT2.value,
      default_agent=DEFAULT_AGENT.value,
      botmatch_replay_dir=BOTMATCH_REPLAY_DIR.value,
      botmatch_max_games=BOTMATCH_MAX_GAMES.value,
  )

  try:
    bot.run()
  finally:
    bot.shutdown()

if __name__ == '__main__':
    app.run(main)
