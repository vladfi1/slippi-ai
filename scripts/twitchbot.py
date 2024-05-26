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
from twitchio.ext import commands
import portpicker
import ray

from slippi_ai import train_lib
from slippi_ai.rl import run_lib
from slippi_ai import flag_utils, eval_lib
from slippi_ai import dolphin as dolphin_lib

# Twitch settings
default_access_token = os.environ.get('TWITCHBOT_ACCESS_TOKEN')
ACCESS_TOKEN = flags.DEFINE_string(
    'token', default_access_token, 'Access token for the twitch bot.',
    required=default_access_token is None)
CHANNEL = flags.DEFINE_string('channel', 'x_pilot', 'twitch channel')

# Bot settings
_DOLPHIN_CONFIG = dolphin_lib.DolphinConfig(
    infinite_time=False,
)
DOLPHIN = ff.DEFINE_dict(
    'dolphin', **flag_utils.get_flags_from_default(_DOLPHIN_CONFIG))

# MODELS_PATH = flags.DEFINE_string('models', 'pickled_models', 'Path to models')

AGENT = ff.DEFINE_dict('agent', **eval_lib.AGENT_FLAGS)
# MODEL = flags.DEFINE_string('model', None, 'Path to pickled model.', required=True)


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

    # For RL, we know the name that was used during training.
    if 'rl_config' in agent_state:
      rl_config = flag_utils.dataclass_from_dict(
          run_lib.Config, agent_state['rl_config'])
      agent_kwargs['name'] = rl_config.agent.name
      logging.info(f'Setting agent name to "{rl_config.agent.name}"')

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

  def stop(self):
    self.stop_requested.set()
    self._thread.join()

RemoteSession = ray.remote(Session)

HELP_MESSAGE = """
!help: Display this message.
!status: Displays current status.
!play <code>: Have the bot connect to you.
!stop: Stop the bot after you are done.
To play against the bot, use the !play command with your connect code, and then direct connect to code {bot_code}.
If you disconnect from the bot in the direct connect lobby, you will have to stop and restart it.
At most {max_players} players can be active at once, with one player on stream.
""".strip()

@dataclasses.dataclass
class SessionInfo:
  session: Session  # actually a RemoteSession
  start_time: datetime.datetime
  twitch_name: str
  connect_code: str

def format_td(td: datetime.timedelta) -> str:
  """Chop off microseconds."""
  return str(td).split('.')[0]

class Bot(commands.Bot):

  def __init__(
      self, token: str, prefix: str, channel: str,
      dolphin_config: dolphin_lib.DolphinConfig,
      agent_kwargs: dict,
      max_sessions: int = 5,
      menu_timeout: float = 5,  # in minutes
  ):
    super().__init__(token=token, prefix=prefix, initial_channels=[channel])
    self.owner = channel

    self.dolphin_config = dolphin_config
    self.agent_kwargs = agent_kwargs
    self._max_sessions = max_sessions
    self._menu_timeout = menu_timeout

    self._sessions: dict[str, SessionInfo] = {}
    self._streaming_against: Optional[str] = None

    self.lock = threading.RLock()

    with open(dolphin_config.user_json_path) as f:
      user_json = json.load(f)

    self.help_message = HELP_MESSAGE.format(
        max_players=max_sessions,
        bot_code=user_json['connectCode'],
    )

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

      session = self._sessions.pop(name).session
      ray.wait([session.stop.remote()])
      await ctx.send(f'Stopped playing against {name}')

      if name == self._streaming_against:
        self._streaming_against = None

  @commands.command()
  async def play(self, ctx: commands.Context):
    with self.lock:
      self._gc_sessions()

      if len(self._sessions) == self._max_sessions:
        await ctx.send('Sorry, too many sessions already active.')
        return

      is_stream = self._streaming_against is None
      name = ctx.author.name
      connect_code = ctx.message.content.split(' ')[1]

      on_off = "on" if is_stream else "off"
      message = f"Connecting to {name} ({connect_code}) {on_off} stream."
      logging.info(message)
      await ctx.send(message)

      session = self._start_session(connect_code, render=is_stream)
      self._sessions[name] = SessionInfo(
          session=session,
          start_time=datetime.datetime.now(),
          twitch_name=name,
          connect_code=connect_code,
      )
      if is_stream:
        self._streaming_against = name

  def _start_session(
      self,
      connect_code: str,
      render: bool = False,
  ) -> Session:
      config = dataclasses.replace(self.dolphin_config)
      config.slippi_port = portpicker.pick_unused_port()
      config.connect_code = connect_code
      config.render = render
      config.headless = not render
      extra_dolphin_kwargs = {}
      if render:
        # TODO: don't hardcode this
        extra_dolphin_kwargs['env_vars'] = dict(DISPLAY=":99")

      return RemoteSession.remote(
          config, self.agent_kwargs,
          extra_dolphin_kwargs=extra_dolphin_kwargs,
      )

  @commands.command()
  async def status(self, ctx: commands.Context):
    with self.lock:
      agent_name = os.path.basename(self.agent_kwargs['path'])
      await ctx.send(f'Bot name: {agent_name}')

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
        await ctx.send(
            f'Playing against {session_info.twitch_name} {on_off} stream.'
            f' Duration {timedelta}, in menu for {menu_time}.')

  @commands.command()
  async def help(self, ctx: commands.Context):
    for line in self.help_message.split('\n'):
      await ctx.send(line)

  def stop_all(self):
    self._stop_sessions(list(self._sessions.values()))

  def _stop_sessions(self, infos: list[SessionInfo]):
    with self.lock:
      ray.wait([info.session.stop.remote() for info in infos])

      for info in infos:
        del self._sessions[info.twitch_name]
        if self._streaming_against == info.twitch_name:
          self._streaming_against = None

  def _gc_sessions(self) -> list[SessionInfo]:
    """Stop sessions that have been in the menu for too long."""
    with self.lock:
      to_gc = []
      for info in self._sessions.values():
        num_menu_frames = ray.get(info.session.num_menu_frames.remote())
        menu_minutes = num_menu_frames / (60 * 60)
        if menu_minutes > self._menu_timeout:
          to_gc.append(info)

      self._stop_sessions(to_gc)
      logging.info(f'GCed {len(to_gc)} sessions.')
      return to_gc

  @commands.command()
  async def gc(self, ctx: commands.Context):
    infos = self._gc_sessions()
    names = [info.twitch_name for info in infos]
    names = ", ".join(names)
    await ctx.send(f"GCed ({names})")

def main(_):
  eval_lib.disable_gpus()

  agent_kwargs = AGENT.value
  if not agent_kwargs['path']:
    raise ValueError('Must provide agent path.')

  bot = Bot(
      token=ACCESS_TOKEN.value,
      prefix='!',
      channel=CHANNEL.value,
      dolphin_config=flag_utils.dataclass_from_dict(
          dolphin_lib.DolphinConfig, DOLPHIN.value),
      agent_kwargs=agent_kwargs,
  )

  try:
    bot.run()
  finally:
    bot.stop_all()

if __name__ == '__main__':
    app.run(main)
