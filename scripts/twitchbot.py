"""Bot that runs on twitch and lets people play against phillip 2."""

import dataclasses
import datetime
import json
import logging
import os
import time
import threading
from typing import Optional

from absl import app, flags
import fancyflags as ff
from twitchio.ext import commands, routines
import portpicker
import ray

from slippi_ai import train_lib
from slippi_ai import flag_utils, eval_lib
from slippi_ai import dolphin as dolphin_lib

# Twitch settings
default_access_token = os.environ.get('TWITCHBOT_ACCESS_TOKEN')
ACCESS_TOKEN = flags.DEFINE_string(
    'token', default_access_token, 'Access token for the twitch bot.',
    required=default_access_token is None)
CHANNEL = flags.DEFINE_string('channel', 'x_pilot', 'twitch channel')

BOT_SESSION_INTERVAL = flags.DEFINE_float(
    'bot_session_interval', 1,
    'minutes before starting a bot session, negative means disabled')

# Bot settings
_DOLPHIN_CONFIG = dolphin_lib.DolphinConfig(
    infinite_time=False,
    save_replays=True,
    replay_dir='Replays/Twitchbot',
    console_timeout=10,
)
DOLPHIN = ff.DEFINE_dict(
    'dolphin', **flag_utils.get_flags_from_default(_DOLPHIN_CONFIG))

MODELS_PATH = flags.DEFINE_string('models', 'pickled_models', 'Path to models')

# Serves as the default agent people play against, and the "screensaver" agent.
agent_flags = eval_lib.AGENT_FLAGS.copy()
agent_flags.update(
    async_inference=ff.Boolean(True),
    jit_compile=ff.Boolean(False),
)
AGENT = ff.DEFINE_dict('agent', **agent_flags)

BOT = flags.DEFINE_string('bot', None, 'Screensaver agent.')
BOT2 = flags.DEFINE_string('bot2', None, 'Second screensaver agent.')

class BotSession:
  """Session between two bots playing locally."""

  def __init__(
      self,
      dolphin_config: dolphin_lib.DolphinConfig,
      agent_kwargs: dict[int, dict],
      extra_dolphin_kwargs: dict = {},
  ):
    eval_lib.disable_gpus()
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

RemoteBotSession = ray.remote(BotSession)

@dataclasses.dataclass
class SessionStatus:
  num_menu_frames: int
  is_alive: bool

class Session:

  def __init__(
      self,
      dolphin_config: dolphin_lib.DolphinConfig,
      agent_kwargs: dict,
      extra_dolphin_kwargs: dict = {},
  ):
    eval_lib.disable_gpus()
    self.dolphin_config = dolphin_config
    self.agent_kwargs = agent_kwargs
    self.stop_requested = threading.Event()

    agent_path = self.agent_kwargs['path']
    agent_state = eval_lib.load_state(path=agent_path)
    agent_config = flag_utils.dataclass_from_dict(
        train_lib.Config, agent_state['config'])

    # Console delay should be at least two to avoid rollbacks, and
    # local delay doesn't need to be more than three.
    max_local_delay = 3
    agent_delay = agent_config.policy.delay
    min_console_delay = min(2, agent_delay)
    margin = agent_delay - min_console_delay
    local_delay = min(margin, max_local_delay)
    console_delay = agent_delay - local_delay

    dolphin_config.online_delay = console_delay
    logging.info(f'Setting console delay to {console_delay}')

    port = 1
    dolphin_kwargs = dolphin_config.to_kwargs()
    dolphin_kwargs.update(extra_dolphin_kwargs)

    player = dolphin_lib.AI()
    dolphin = dolphin_lib.Dolphin(
        players={port: player},
        **dolphin_kwargs,
    )

    agent = eval_lib.build_agent(
        controller=dolphin.controllers[port],
        opponent_port=None,  # will be set later
        console_delay=console_delay,
        run_on_cpu=True,
        state=agent_state,
        **agent_kwargs,
    )

    eval_lib.update_character(player, agent.config)

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

      # This gets us through the menus and into the first frame of the actual game
      for gamestate in gamestates:
        if self.stop_requested.is_set():
          dolphin.stop()
          return

        if not dolphin_lib.is_menu_state(gamestate):
          self._num_menu_frames = 0
          break

        self._num_menu_frames += 1

      # Now we have access to the display names to set the correct ports.
      name_to_port = {
          player.displayName: port for port, player in gamestate.players.items()
      }
      actual_port = name_to_port[display_name]
      ports = list(gamestate.players)
      ports.remove(actual_port)
      opponent_port = ports[0]
      agent.players = (actual_port, opponent_port)

      # Main loop
      agent.start()
      try:
        while not self.stop_requested.is_set():
          gamestate = next(gamestates)
          if not dolphin_lib.is_menu_state(gamestate):
            agent.step(gamestate)
            self._num_menu_frames = 0
          else:
            self._num_menu_frames += 1
      finally:
        agent.stop()
        dolphin.stop()

    self._thread = threading.Thread(target=run)
    self._thread.start()

  def num_menu_frames(self) -> int:
    return self._num_menu_frames

  def status(self) -> SessionStatus:
    return SessionStatus(
        num_menu_frames=self._num_menu_frames,
        is_alive=self._thread.is_alive(),
    )

  def stop(self):
    self.stop_requested.set()
    self._thread.join()

RemoteSession = ray.remote(Session)

HELP_MESSAGE = """
!status: Displays selected agent and current sessions.
!play <code>: Have the bot connect to you.
!stop: Stop the bot after you are done. Doesn't work if the game is paused.
!agents: List available agents to play against.
!agent <name>: Select an agent to play against.
!about: Some info about the this AI.
To play against the bot, use the !play command with your connect code, and then direct connect to code {bot_code}.
If you disconnect from the bot in the direct connect lobby, you will have to stop and restart it.
At most {max_players} players can be active at once, with one player on stream. If no one is playing, bots may be on stream.
""".strip()

ABOUT_MESSAGE = """
Melee AI trained with a combination of imitation learning from slippi replays and self-play reinforcement learning.
 Replays are mostly from ranked, with some tournaments and personal dumps.
 Code at https://github.com/vladfi1/slippi-ai.
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

class Bot(commands.Bot):

  def __init__(
      self, token: str, prefix: str, channel: str,
      dolphin_config: dolphin_lib.DolphinConfig,
      agent_kwargs: dict,
      models_path: str,
      max_sessions: int = 4,  # Includes stream session.
      menu_timeout: float = 3,  # in minutes
      bot_session_interval: float = 1, # in minutes
      bot: Optional[str] = None,
      bot2: Optional[str] = None,
  ):
    super().__init__(token=token, prefix=prefix, initial_channels=[channel])
    self.owner = channel

    self.dolphin_config = dolphin_config
    self.agent_kwargs = agent_kwargs
    self._max_sessions = max_sessions
    self._menu_timeout = menu_timeout
    self._bot_session_interval = bot_session_interval

    bot1 = bot or agent_kwargs['path']
    bot2 = bot2 or bot1
    self._bot_agent_kwargs = {
        1: dict(agent_kwargs, path=bot1),
        2: dict(agent_kwargs, path=bot2),
    }
    self._bot1 = bot1
    self._bot2 = bot2

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
    self._reload_models()

    self._default_agent_name = os.path.basename(agent_kwargs['path'])
    self._requested_agents = {}
    self._play_codes = {}

    self._do_chores.start()

  @commands.command()
  async def help(self, ctx: commands.Context):
    for line in self.help_message.split('\n'):
      await ctx.send(line)

  @commands.command()
  async def about(self, ctx: commands.Context):
    await ctx.send(ABOUT_MESSAGE)

  def _reload_models(self):
    self._models: dict[str, dict] = {}
    agents = os.listdir(self._models_path)

    for agent in agents:
      path = os.path.join(self._models_path, agent)
      state = eval_lib.load_state(path=path)
      state = {k: state[k] for k in ['step', 'config', 'rl_config'] if k in state}
      self._models[agent] = state

  @commands.command()
  async def reload(self, ctx: commands.Context):
    with self.lock:
      self._reload_models()
    models_str = ", ".join(self._models)
    await ctx.send(f'Available agents: {models_str}')

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
    max_chars = 500
    chunks = [[]]
    chunk_size = 0
    for model in sorted(self._models):
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

    agent = words[1]
    if agent not in self._models:
      await ctx.send(f'{agent} is not a valid agent')
      models_str = ", ".join(self._models)
      await ctx.send(f'Available agents: {models_str}')
      return

    name = ctx.author.name
    self._requested_agents[name] = agent
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
      connect_code = self._play_codes[name]
      assert connect_code

      if name in self._sessions:
        await ctx.send(f'{name}, you are already playing')
        return

      await self._gc_sessions()

      if len(self._sessions) == self._max_sessions:
        await ctx.send('Sorry, too many sessions already active.')
        return

      is_stream = self._streaming_against is None
      if is_stream:
        self._stop_bot_session()

      on_off = "on" if is_stream else "off"
      message = f"Connecting to {name} ({connect_code}) {on_off} stream."
      logging.info(message)
      await ctx.send(message)

      agent = self._get_opponent(name)
      session = self._start_session(
          connect_code,
          render=is_stream,
          agent=agent,
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
      agent: str,
      render: bool = False,
  ) -> Session:
    config = dataclasses.replace(self.dolphin_config)
    config.slippi_port = portpicker.pick_unused_port()
    config.connect_code = connect_code
    config.render = render
    config.headless = not render
    config.replay_dir = os.path.join(config.replay_dir, agent)
    os.makedirs(config.replay_dir, exist_ok=True)
    extra_dolphin_kwargs = {}
    if render:
      # TODO: don't hardcode this
      extra_dolphin_kwargs['env_vars'] = dict(DISPLAY=":99")

    agent_kwargs = self.agent_kwargs.copy()
    agent_kwargs['path'] = os.path.join(self._models_path, agent)

    return RemoteSession.remote(
        config, agent_kwargs,
        extra_dolphin_kwargs=extra_dolphin_kwargs,
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
      if chan:
        # Might be None if we haven't logged in yet
        await chan.send('Started bot session on stream.')
      return True

  @commands.command()
  async def start_bot_session(self, ctx: commands.Context):
    started = await self._maybe_start_bot_session()
    if started:
      await ctx.send('Started bot session on stream.')
    else:
      await ctx.send('Did not start bot session on stream.')

  def _get_opponent(self, name: str) -> str:
    return self._requested_agents.get(name, self._default_agent_name)

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
    """Stop sessions that have been in the menu for too long."""
    with self.lock:
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
      dolphin_config=flag_utils.dataclass_from_dict(
          dolphin_lib.DolphinConfig, DOLPHIN.value),
      agent_kwargs=agent_kwargs,
      bot_session_interval=BOT_SESSION_INTERVAL.value,
      bot=BOT.value,
      bot2=BOT2.value,
  )

  try:
    bot.run()
  finally:
    bot.shutdown()

if __name__ == '__main__':
    app.run(main)
