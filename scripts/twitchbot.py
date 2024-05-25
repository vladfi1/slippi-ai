"""Bot that runs on twitch and lets people play against phillip 2."""

import dataclasses
import json
import multiprocessing as mp
from multiprocessing.synchronize import Event
import logging
import os
import threading
from typing import Optional

from absl import app, flags
import fancyflags as ff
from twitchio.ext import commands
import portpicker

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


def run_bot(
    config: dolphin_lib.DolphinConfig,
    agent_kwargs: dict,
    stop: Event,
):
  eval_lib.disable_gpus()

  agent_path = agent_kwargs['path']
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

  config.online_delay = console_delay
  logging.info(f'Setting console delay to {console_delay}')

  # For RL, we know the name that was used during training.
  if 'rl_config' in agent_state:
    rl_config = flag_utils.dataclass_from_dict(
        run_lib.Config, agent_state['rl_config'])
    agent_kwargs['name'] = rl_config.agent.name
    logging.info(f'Setting agent name to "{rl_config.agent.name}"')

  port = 1
  dolphin_kwargs = config.to_kwargs()

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

  # Start game
  gamestate = dolphin.step()

  # Figure out which port we are
  with open(config.user_json_path) as f:
    user_json = json.load(f)
  display_name = user_json['displayName']

  name_to_port = {
      player.displayName: port for port, player in gamestate.players.items()
  }
  actual_port = name_to_port[display_name]
  ports = list(gamestate.players)
  ports.remove(actual_port)
  opponent_port = ports[0]
  agent.players = (actual_port, opponent_port)

  # Main loop
  while not stop.is_set():
    # "step" to the next frame
    gamestate = dolphin.step()
    agent.step(gamestate)

  dolphin.stop()

HELP_MESSAGE = """
!help: Display this message.
!status: Displays current status.
!play <code>: Have the bot connect to you.
!stop: Stop the bot after you are done.
To play against the bot, use the !play command with your connect code, and then direct connect to code {bot_code}.
If you disconnect from the bot in the direct connect lobby, you will have to stop and restart it.
""".strip()

class Bot(commands.Bot):

  def __init__(
      self, token: str, prefix: str, channel: str,
      dolphin_config: dolphin_lib.DolphinConfig,
      agent_kwargs: dict,
  ):
    super().__init__(token=token, prefix=prefix, initial_channels=[channel])
    self.owner = channel

    self.dolphin_config = dolphin_config
    self.agent_kwargs = agent_kwargs

    self.lock = threading.RLock()
    self.process: Optional[mp.Process] = None
    self._stop_flag: Event = mp.Event()
    self.playing_against: Optional[str] = None

    with open(dolphin_config.user_json_path) as f:
      user_json = json.load(f)

    self.help_message = HELP_MESSAGE.format(bot_code=user_json['connectCode'])

  async def event_ready(self):
    # Notify us when everything is ready!
    # We are logged in and ready to chat and use commands...
    print(f'Logged in as | {self.nick}')
    print(f'User id is | {self.user_id}')

  @commands.command()
  async def hello(self, ctx: commands.Context):
    # Here we have a command hello, we can invoke our command with our prefix and command name
    # e.g ?hello
    # We can also give our commands aliases (different names) to invoke with.

    # Send a hello back!
    # Sending a reply back to the channel is easy... Below is an example.
    await ctx.send(f'Hello {ctx.author.name}!')

  def stop_bot(self):
    with self.lock:
      if self.process is None:
        return

      logging.info('Stopping bot process')
      self._stop_flag.set()
      self.process.join(timeout=10)
      if self.process.is_alive():
        logging.warning('Forcibly terminating bot process.')
        self.process.kill()
        self.process.join()
      self.process = None

  @commands.command()
  async def stop(self, ctx: commands.Context):
    with self.lock:
      if self.process is None:
        await ctx.send('Already stopped.')
        return

      if not (ctx.author.is_mod or ctx.author.name == self.playing_against):
        await ctx.send('Insufficient priviledges; only the mods and the current player may stop the bot.')
        return

      self.stop_bot()

  @commands.command()
  async def play(self, ctx: commands.Context):
    with self.lock:
      if self.process is not None:
        # TODO: implement a timeout
        await ctx.send('Sorry, already playing')

      connect_code = ctx.message.content.split(' ')[1]
      logging.info(f'Connecting to {connect_code}')
      await ctx.send(f'Connecting to {connect_code}')

      config = dataclasses.replace(self.dolphin_config)
      config.slippi_port = portpicker.pick_unused_port()
      config.connect_code = connect_code

      self._stop_flag.clear()
      self.process = mp.Process(target=run_bot, kwargs=dict(
          config=config,
          agent_kwargs=self.agent_kwargs,
          stop=self._stop_flag,
      ))
      self.process.start()
      self.playing_against = ctx.author.name

  @commands.command()
  async def status(self, ctx: commands.Context):
    with self.lock:
      agent_name = os.path.basename(self.agent_kwargs['path'])
      await ctx.send(f'Bot name: {agent_name}')

      if self.process is None:
        await ctx.send('Not currently running.')
      else:
        await ctx.send(f'Currently playing against "{self.playing_against}"')

  @commands.command()
  async def help(self, ctx: commands.Context):
    for line in self.help_message.split('\n'):
      await ctx.send(line)

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
    bot.stop_bot()

if __name__ == '__main__':
    app.run(main)
