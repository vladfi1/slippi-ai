"""Twitch chat frontend: lets people play phillip 2 from twitch chat.

Needs the backend (scripts/tbot_backend.py) to be running, and can be started
and stopped without it.
"""

import asyncio
import json
import os
from typing import Optional

from absl import app, flags
import ray
from twitchio.ext import commands

import melee

from slippi_ai.tbot import api
from slippi_ai.tbot import client as client_lib
from slippi_ai.tbot.api import format_td, parse_character
from slippi_ai.tbot.client import ManagerClient

ACCESS_TOKEN = flags.DEFINE_string(
    'token', os.environ.get('TWITCHBOT_ACCESS_TOKEN'),
    'Access token for the twitch bot.')
CHANNEL = flags.DEFINE_string('channel', 'x_pilot', 'twitch channel')
RAY_ADDRESS = flags.DEFINE_string(
    'ray_address', 'auto', 'Address of the Ray cluster the backend runs in.')

HELP_MESSAGE = """
!play <code>: Have the bot connect to you. Connect to the bot with code {bot_code}.
!agents [<char> [<opp>]]: List available agents to play against, e.g. "!agents fox falco" lists fox agents trained vs. falco (use "all" for any). The auto-* agents will pick the strongest agent based the matchup. The basic-* agents are much weaker, and the medium-* agents are in the middle.
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
!auto <char> <opp>: Show which model auto-<char> plays vs. <opp>.
!rating <agent>: Show an agent's rating from our evals.
!stocks <n>: Have the bot SD down to n stocks at the start of each game. Use "!stocks 4" to disable.
!bots <agent1> [<agent2>]: Set one or two "screensaver" agents to play while no one is on stream. Locks the bots for {bots_lock_minutes} minutes or until you !relinquish.
!about: Print some info about the this AI.
"""

ABOUT_MESSAGE = """
Melee AI trained with a combination of imitation learning from slippi replays and self-play reinforcement learning.
 Replays are mostly from ranked, with some tournaments and personal dumps.
 Agents by default have a reaction time of 18 frames (300 ms).
 Agents cannot see Randall, FoD platforms, Nana, and items or projectiles.
 Code: https://github.com/vladfi1/slippi-ai. Discord: https://discord.gg/hfVTXGu.
""".strip()

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

def play_message(user: api.User, result: api.PlayResult) -> str:
  on_off = "on" if result.on_stream else "off"
  return f"Connecting to {user.name} ({result.connect_code}) {on_off} stream."


class Bot(commands.Bot):
  """Twitch chat frontend to the SessionManager."""

  def __init__(
      self, token: str, prefix: str, channel: str,
      manager: ManagerClient, info: api.ManagerInfo,
  ):
    """Run manager.listen(bot.send_to_channel) to pass on announcements."""
    super().__init__(token=token, prefix=prefix, initial_channels=[channel])
    self.owner = channel
    self.manager = manager
    self.botmatch_max_games = info.botmatch_max_games

    self.help_message = HELP_MESSAGE.format(
        max_players=info.max_sessions,
        bot_code=info.bot_connect_code,
        botmatch_max_games=info.botmatch_max_games,
    )
    self.extra_help_message = EXTRA_HELP_MESSAGE.format(
        bots_lock_minutes=info.bots_lock_minutes)

  async def send_to_channel(self, message: str):
    chan = self.get_channel(self.owner)
    # Might be None if we haven't logged in yet
    if chan:
      await chan.send(message)

  async def event_command_error(
      self, context: commands.Context, error: Exception):
    if isinstance(error, api.UserError):
      await context.send(str(error))
      return
    await super().event_command_error(context, error)

  async def event_ready(self):
    # Notify us when everything is ready!
    # We are logged in and ready to chat and use commands...
    print(f'Logged in as | {self.nick}')
    print(f'User id is | {self.user_id}')

  @commands.command()
  async def help(self, ctx: commands.Context):
    for line in self.help_message.split('\n'):
      await ctx.send(line)

  @commands.command()
  async def more(self, ctx: commands.Context):
    for line in self.extra_help_message.split('\n'):
      await ctx.send(line)

  @commands.command()
  async def about(self, ctx: commands.Context):
    await ctx.send(ABOUT_MESSAGE)

  @commands.command()
  async def reload(self, ctx: commands.Context):
    num_loaded = await self.manager.reload()
    await ctx.send(f'Loaded {num_loaded} new or updated models.')
    await self.agents_full(ctx)

  @commands.command()
  async def config(self, ctx: commands.Context):
    words = ctx.message.content.split(' ')
    if len(words) != 2:
      await ctx.send('You must specify an agent.')
      return

    agent = words[1]
    try:
      config = await self.manager.model_config(agent)
    except api.UserError as e:
      await ctx.send(str(e))
      models_str = ", ".join(await self.manager.model_names())
      await ctx.send(f'Available agents: {models_str}')
      return

    message = json.dumps(config)
    chunk_size = 500

    for i in range(0, len(message), chunk_size):
      await ctx.send(message[i:i+chunk_size])

  @commands.command()
  async def agents(self, ctx: commands.Context):
    """Lists agents, optionally filtered by character and opponent.

    !agents: list the special (auto/basic/multi-character) agents.
    !agents <char>: list all agents that play <char>.
    !agents <char> <opp>: list all agents that play <char> vs. <opp>.
    "all" may be used in place of either character.
    """
    words = tokens(ctx.message.content)[1:]

    if len(words) == 0:
      await send_list(ctx, await self.manager.special_agents())
      return

    if len(words) > 2:
      await ctx.send('Usage: !agents [<character> [<opponent>]]')
      return

    filters: list[Optional[melee.Character]] = []
    for word in words:
      if word.lower() == 'all':
        filters.append(None)
        continue
      char = parse_character(word)
      if char is None:
        await ctx.send(f'Unknown character "{word}".')
        return
      filters.append(char)

    char = filters[0]
    opponent = filters[1] if len(filters) > 1 else None

    names = await self.manager.find_agents(char, opponent)
    if len(names) == 0:
      await ctx.send('No matching agents.')
      return

    await send_list(ctx, names)

  @commands.command()
  async def rating(self, ctx: commands.Context):
    words = tokens(ctx.message.content)[1:]
    if len(words) != 1:
      await ctx.send('Usage: !rating <agent>')
      return
    name = words[0]

    auto_char = await self.manager.auto_agent_character(name)
    if auto_char is not None:
      char_name = auto_char.name.lower()
      await ctx.send(
          f'{name} picks a model per opponent; try !auto {char_name} <opp>.')
      return

    model, ratings = await self.manager.ratings(name)
    if not ratings:
      await ctx.send(f'{name} has no rating.')
      return
    if len(ratings) == 1:
      r, = ratings
      char_name = r.character.name.lower()
      await ctx.send(
          f'{model} ({char_name}): rating {r.rating:.0f},'
          f' #{r.rank} of {r.count} {char_name} agents.')
      return

    # One compact entry per character, e.g. "fox 460 (#24/37)".
    items = [f'{model} ratings:']
    for r in ratings:
      items.append(
          f'{r.character.name.lower()} {r.rating:.0f} (#{r.rank}/{r.count}),')
    items[-1] = items[-1].rstrip(',')
    await send_list(ctx, items)

  @commands.command()
  async def auto(self, ctx: commands.Context):
    words = tokens(ctx.message.content)[1:]
    if len(words) != 2:
      await ctx.send('Usage: !auto <character> <opponent>')
      return

    chars: list[melee.Character] = []
    for word in words:
      char = parse_character(word)
      if char is None:
        await ctx.send(f'Unknown character "{word}".')
        return
      chars.append(char)
    char, opponent = chars

    pick = await self.manager.auto_pick(char, opponent)
    opp_name = opponent.name.lower()
    trained = 'trained' if pick.trained else 'not trained'
    message = (
        f'{pick.agent} vs. {opp_name} plays {pick.model}'
        f' ({trained} vs. {opp_name}).')
    if pick.rating is not None:
      message += f' Rating {pick.rating:.0f}.'
    await ctx.send(message)

  @commands.command()
  async def agents_full(self, ctx: commands.Context):
    await send_list(ctx, await self.manager.agent_names())

  @commands.command()
  async def agent(self, ctx: commands.Context):
    words = ctx.message.content.split(' ')

    if len(words) != 2:
      await ctx.send('You must specify a single agent.')
      return

    agent_name: str = words[1]
    user = twitch_user(ctx)
    result = await self.manager.select_agent(user, agent_name)
    await ctx.send(f'{user.name} has selected {agent_name}')

    if result.restarted is not None:
      await ctx.send(f'Restarted session with {user.name}')
      await ctx.send(play_message(user, result.restarted))

  @commands.command()
  async def stages(self, ctx: commands.Context):
    user = twitch_user(ctx)
    name = user.name
    words = ctx.message.content.split(' ')

    stage_names = words[1:]
    if len(stage_names) == 0:
      stages = await self.manager.stages(user)
      if stages is None:
        stages_string = 'all'
      else:
        stages_string = ", ".join([s.name.lower() for s in stages])
      await ctx.send(
          f'{name} has selected {stages_string}.'
          f' Available stages: {", ".join(api.NAME_TO_STAGE)}')
      return

    stages: list[melee.Stage] = []
    for stage_name in stage_names:
      stage_name = stage_name.lower()
      if stage_name not in api.NAME_TO_STAGE:
          await ctx.send(
              f'{name}, {stage_name} is not a valid stage.'
              f' Available stages: {", ".join(api.NAME_TO_STAGE)}')
          return
      stages.append(api.NAME_TO_STAGE[stage_name])

    await self.manager.set_stages(user, stages)
    await ctx.send(f'{name} has set the stages to {", ".join(stage_names)}')

  @commands.command()
  async def stocks(self, ctx: commands.Context):
    user = twitch_user(ctx)
    name = user.name
    words = ctx.message.content.split(' ')

    if len(words) == 1:
      stocks = await self.manager.sd_stocks(user)
      if stocks is None:
        await ctx.send(
            f'{name}, the bot will play with all its stocks.'
            ' Use "!stocks <n>" to have it SD down to n stocks.')
      else:
        await ctx.send(f'{name}, the bot will SD down to {stocks} stocks.')
      return

    try:
      stocks = int(words[1])
    except ValueError:
      await ctx.send(f'{name}, "{words[1]}" is not a number.')
      return

    await self.manager.set_sd_stocks(user, stocks)

    if stocks == api.MAX_STOCKS:
      await ctx.send(
          f'{name}, the bot will play with all its stocks.'
          ' Takes effect on your next game.')
      return

    await ctx.send(
        f'{name}, the bot will SD down to {stocks} stocks at the start of'
        ' each game. Takes effect on your next session (!play or !reset).')

  @commands.command()
  async def stop(self, ctx: commands.Context):
    user = twitch_user(ctx)
    await self.manager.stop(user)
    await ctx.send(f'Stopped playing against {user.name}')

  @commands.command()
  async def play(self, ctx: commands.Context):
    user = twitch_user(ctx)
    words = ctx.message.content.split(' ')

    if len(words) > 1:
      await self.manager.set_connect_code(user, words[1])

    result = await self.manager.play(user)
    await ctx.send(play_message(user, result))

  @commands.command()
  async def botmatch(self, ctx: commands.Context):
    words = ctx.message.content.split(' ')[1:]

    if len(words) == 1:
      agent1_name = agent2_name = words[0]
    elif len(words) == 2:
      agent1_name, agent2_name = words
    else:
      await ctx.send('Usage: !botmatch agent1 [agent2]')
      return

    agent_label = await self.manager.botmatch(
        twitch_user(ctx), agent1_name, agent2_name)
    await ctx.send(
        f'Started bot match: {agent_label}'
        f' ({self.botmatch_max_games} games).')

  @commands.command()
  async def reset(self, ctx: commands.Context):
    user = twitch_user(ctx)
    result = await self.manager.reset(user)
    await ctx.send(play_message(user, result))

  @commands.command()
  async def start_bot_session(self, ctx: commands.Context):
    started = await self.manager.start_bot_session()
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

    await self.manager.set_bots(
        twitch_user(ctx), bot1, bot2, is_mod=ctx.author.is_mod)

  @commands.command()
  async def relinquish(self, ctx: commands.Context):
    await self.manager.relinquish(twitch_user(ctx), is_mod=ctx.author.is_mod)
    await ctx.send('Bots relinquished; anyone can now use !bots.')

  @commands.command()
  async def status(self, ctx: commands.Context):
    user = twitch_user(ctx)
    await ctx.send(f'Selected agent: {await self.manager.selected_agent(user)}')

    status = await self.manager.status()

    bots = status.bots
    if bots is not None:
      message = f'{bots.bot1} vs. {bots.bot2} on stream'
      if bots.set_by is not None and bots.playing_for is not None:
        message += (
            f' (set by {bots.set_by},'
            f' playing for {format_td(bots.playing_for)}')
        if bots.lock_remaining is not None:
          message += f', locked for {format_td(bots.lock_remaining)} more'
        message += ')'
      await ctx.send(message + '.')

    if not status.sessions:
      await ctx.send('No active sessions.')
      return

    for session in status.sessions:
      duration = format_td(session.duration)
      player = session.user.name
      agent = session.agent

      if session.connect_code is None:
        games = session.games_completed or 0
        await ctx.send(
            f'Bot match: {agent} for {duration}, {games} games completed'
            f' (requested by {player}).')
      else:
        on_off = "on" if session.on_stream else "off"
        if session.menu_time is None:
          menu = 'starting up'
        else:
          menu = f'in menu for {format_td(session.menu_time)}'
        await ctx.send(
            f'Playing against {player} as {agent} {on_off} stream'
            f' for {duration} ({menu}).')

  @commands.command()
  async def stop_bot_session(self, ctx: commands.Context):
    if await self.manager.stop_bot_session():
      await ctx.send('Stopped bot session.')
    else:
      await ctx.send('Bot session not running.')

  @commands.command()
  async def kick(self, ctx: commands.Context):
    if not ctx.author.is_mod:
      await ctx.send('Only mods can kick players.')
      return

    words = ctx.message.content.split(' ')[1:]
    if len(words) != 1:
      await ctx.send('Must specify a player to kick')
      return

    user = await self.manager.kick(words[0])
    await ctx.send(f'Kicked {user.name}')

  @commands.command()
  async def gc(self, ctx: commands.Context):
    users = await self.manager.gc()
    names = ", ".join(user.name for user in users)
    await ctx.send(f"GCed ({names})")

def twitch_user(ctx: commands.Context) -> api.User:
  name = ctx.author.name
  assert isinstance(name, str)
  return api.User(frontend='twitch', key=name, name=name)

def main(_):
  if not ACCESS_TOKEN.value:
    raise app.UsageError('Set --token or TWITCHBOT_ACCESS_TOKEN.')

  ray.init(address=RAY_ADDRESS.value, namespace=client_lib.NAMESPACE)
  manager = ManagerClient.connect()
  try:
    asyncio.run(run(manager))
  except KeyboardInterrupt:
    pass

async def run(manager: ManagerClient):
  # Made inside the event loop, which twitchio picks up.
  bot = Bot(
      token=ACCESS_TOKEN.value,
      prefix='!',
      channel=CHANNEL.value,
      manager=manager,
      info=await manager.info(),
  )
  await asyncio.gather(bot.start(), manager.listen(bot.send_to_channel))

if __name__ == '__main__':
  app.run(main)
