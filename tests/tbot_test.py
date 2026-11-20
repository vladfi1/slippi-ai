"""Tests for slippi_ai.tbot and the twitchbot. Run with `python tests/tbot_test.py`.

Sessions are replaced by fakes, so no Dolphin or ray workers are started.
Needs the twitchbot extras (twitchio, ray, aiohttp).
"""

import asyncio
import json
import os
import pickle
import sys
import tempfile
import time
import types
import typing as tp
import unittest

from aiohttp import test_utils
import melee

from slippi_ai import dolphin as dolphin_lib
from slippi_ai.tbot import manager as manager_lib
from slippi_ai.tbot import sessions as sessions_lib
from slippi_ai.tbot import web as web_lib
from slippi_ai.tbot.persistence import LAST_SEEN, TOUCH_INTERVAL, UserPrefs
from slippi_ai.tbot.manager import User, UserError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import twitchbot  # pylint: disable=wrong-import-position


# Model states, as in tests/eval_db_test.py.

def imitation_state(characters: str, opponents: str = 'all', delay: int = 18):
  return dict(
      config=dict(
          dataset=dict(allowed_characters=characters, allowed_opponents=opponents),
          policy=dict(delay=delay),
      ),
      name_map={},
      state={},
  )

def rl_state(character: str, opponent: str, delay: int = 21):
  state = imitation_state(character, 'all', delay)
  state['agent_config'] = dict(name=['Cody', 'Hax'], rating=3000.0)
  state['opponent'] = opponent
  return state

def multichar_rl_state(characters: list[str], delay: int = 21):
  state = imitation_state(','.join(characters), 'all', delay)
  state['rl_config'] = dict(agent=dict(name='Master Player', rating=2500.0))
  return state

MODELS = {
    'fox_imitation': imitation_state('fox'),
    'fox_vs_falco': rl_state('fox', 'falco'),
    'fox_vs_marth': rl_state('fox', 'marth', delay=18),
    'marth_vs_fox': rl_state('marth', 'fox', delay=18),
    'multi': multichar_rl_state(['fox', 'falco']),
}

BOT_CONNECT_CODE = 'PHAI#591'
DAY = 24 * 60 * 60


# Fake ray actors.

class Later(tp.NamedTuple):
  """A result that a FakeRef gets by awaiting fn() once it is awaited."""
  fn: tp.Callable[[], tp.Awaitable]

class FakeRef:
  """An awaitable like a ray ObjectRef."""

  def __init__(self, fn: tp.Callable[[], tp.Any]):
    # Run now, like a remote call that has been submitted.
    try:
      self._result = fn()
      self._error = None
    except Exception as e:
      self._result = None
      self._error = e

  def __await__(self):
    if self._error is not None:
      raise self._error
    result = self._result
    if isinstance(result, Later):
      result = yield from result.fn().__await__()
    else:
      yield from asyncio.sleep(0).__await__()
    return result

class FakeMethod:

  def __init__(self, fn: tp.Callable):
    self._fn = fn

  def remote(self, *args, **kwargs) -> FakeRef:
    return FakeRef(lambda: self._fn(*args, **kwargs))

class FakeSession:
  """Stands in for a Session or BotSession actor."""

  def __init__(self, kind: str, args: tuple, kwargs: dict):
    self.kind = kind
    self.args = args
    self.kwargs = kwargs
    self.alive = True
    self.num_menu_frames = 0
    self.lines = ['session started']
    # Behaviors the tests can turn on.
    self.hang_stop = False  # stop() waits until Dolphin is killed
    self.crashed = False  # the actor has died
    self.starting = False  # still loading agents, so not answering
    self.stop_calls = 0
    self.killed = False

    self.stop = FakeMethod(self._stop)
    self.status = FakeMethod(self._status)
    self.logs = FakeMethod(self._logs)
    self.kill_dolphin = FakeMethod(self._kill_dolphin)

  def _stop(self):
    self.stop_calls += 1
    if self.crashed:
      raise RuntimeError('The actor died.')
    if self.hang_stop and not self.killed:
      return Later(self._wait_for_kill)
    self.alive = False

  async def _wait_for_kill(self):
    while not self.killed:
      await asyncio.sleep(0.01)
    self.alive = False

  def _kill_dolphin(self):
    self.killed = True

  def _status(self):
    if self.crashed:
      raise RuntimeError('The actor died.')
    if self.starting:
      return Later(lambda: asyncio.sleep(3600))
    return sessions_lib.SessionStatus(
        num_menu_frames=self.num_menu_frames, is_alive=self.alive,
        games_completed=2)

  def _logs(self, since: int):
    return len(self.lines), self.lines[since:]

class FakeRemote:
  """Stands in for ray.remote(Session) and friends."""

  def __init__(self, kind: str, created: list[FakeSession]):
    self.kind = kind
    self.created = created

  def remote(self, *args, **kwargs) -> FakeSession:
    session = FakeSession(self.kind, args, kwargs)
    self.created.append(session)
    return session


class ManagerTestCase(unittest.IsolatedAsyncioTestCase):
  """Sets up a SessionManager with fake sessions and synthetic models."""

  # Overridden by tests that need them.
  max_sessions = 3
  bot_session_interval = -1.0  # no bot session on stream
  user_prefs: dict = {}
  prefs_max_days: dict = {}

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    root = self.tmp.name

    models_path = os.path.join(root, 'models')
    os.makedirs(models_path)
    for name, state in MODELS.items():
      with open(os.path.join(models_path, name), 'wb') as f:
        pickle.dump(state, f)

    user_json_path = os.path.join(root, 'user.json')
    with open(user_json_path, 'w') as f:
      json.dump(dict(connectCode=BOT_CONNECT_CODE, displayName='phillip'), f)

    self.state_dir = os.path.join(root, 'state')
    if self.user_prefs:
      os.makedirs(self.state_dir)
      with open(os.path.join(self.state_dir, 'user_prefs.json'), 'w') as f:
        json.dump(self.user_prefs, f)

    self.sessions: list[FakeSession] = []
    self.patches = []
    for name, kind in [
        ('RemoteSession', 'human'), ('RemoteGpuSession', 'human'),
        ('RemoteBotSession', 'bot'), ('RemoteGpuBotSession', 'bot'),
    ]:
      self._patch(sessions_lib, name, FakeRemote(kind, self.sessions))
    self._patch(manager_lib, 'STOP_TIMEOUT', 0.2)
    self._patch(manager_lib, 'KILL_TIMEOUT', 2)

    self.manager = manager_lib.SessionManager(
        dolphin_config=dolphin_lib.DolphinConfig(
            user_json_path=user_json_path,
            replay_dir=os.path.join(root, 'replays'),
        ),
        agent_kwargs=dict(run_on_cpu=True, batch_steps=1),
        models_path=models_path,
        max_sessions=self.max_sessions,
        bot_session_interval=self.bot_session_interval,
        default_agent='fox_vs_falco',
        botmatch_replay_dir=os.path.join(root, 'botmatch'),
        botmatch_max_games=5,
        state_dir=self.state_dir,
        prefs_max_days=self.prefs_max_days,
    )
    self.announcements: list[str] = []

    async def listener(message: str):
      self.announcements.append(message)

    self.manager.add_listener(listener)

  def _patch(self, module, name: str, value):
    self.patches.append((module, name, getattr(module, name)))
    setattr(module, name, value)

  def tearDown(self):
    for module, name, value in reversed(self.patches):
      setattr(module, name, value)
    self.tmp.cleanup()

  def player(self, name: str, connect_code: tp.Optional[str] = None) -> User:
    user = User(frontend='twitch', key=name, name=name)
    if connect_code is not None:
      self.manager.set_connect_code(user, connect_code)
    return user

  def human_sessions(self) -> list[FakeSession]:
    return [s for s in self.sessions if s.kind == 'human']


class AgentsTest(ManagerTestCase):

  def test_agent_names(self):
    names = self.manager.agent_names()
    for name in ['fox_vs_falco', 'marth_vs_fox', 'multi-fox', 'multi-falco',
                 'basic-fox']:
      self.assertIn(name, names)
    self.assertIn('auto-fox', names)

  def test_find_agents(self):
    fox_vs_falco = self.manager.find_agents(
        melee.Character.FOX, melee.Character.FALCO)
    self.assertIn('fox_vs_falco', fox_vs_falco)
    self.assertNotIn('fox_vs_marth', fox_vs_falco)
    self.assertNotIn('auto-fox', fox_vs_falco)
    marth = self.manager.find_agents(melee.Character.MARTH)
    self.assertEqual(marth, ['marth_vs_fox'])

  def test_agent_table(self):
    rows = {row.name: row for row in self.manager.agent_table()}
    self.assertEqual(rows['auto-fox'].kind, 'auto')
    self.assertEqual(rows['basic-fox'].kind, 'basic')
    self.assertEqual(rows['fox_vs_falco'].kind, 'model')
    self.assertEqual(rows['fox_vs_falco'].delay, 21)
    self.assertEqual(rows['fox_vs_falco'].opponents, [melee.Character.FALCO])
    # Imitation models are only listed as basic-<char>.
    self.assertNotIn('fox_imitation', rows)

  def test_unknown_agent(self):
    with self.assertRaises(UserError):
      self.manager.model_config('nope')


class PlayTest(ManagerTestCase):

  async def test_play_needs_connect_code(self):
    with self.assertRaisesRegex(UserError, 'connect code'):
      await self.manager.play(self.player('alice'))

  async def test_invalid_connect_code(self):
    with self.assertRaises(UserError):
      self.player('alice', 'ABC123')

  async def test_play_and_stop(self):
    alice = self.player('alice', 'abc#123')
    result = await self.manager.play(alice)
    self.assertEqual(result.connect_code, 'ABC#123')
    self.assertEqual(result.agent, 'fox_vs_falco')
    self.assertTrue(result.on_stream)

    session, = self.human_sessions()
    config, agent_config = session.args
    self.assertEqual(config.connect_code, 'ABC#123')
    self.assertTrue(config.render)
    self.assertEqual(agent_config.name, 'fox_vs_falco')

    with self.assertRaisesRegex(UserError, 'already playing'):
      await self.manager.play(alice)

    await self.manager.stop(alice)
    self.assertEqual(session.stop_calls, 1)
    self.assertIsNone(self.manager.session_id(alice))
    with self.assertRaisesRegex(UserError, 'not playing'):
      await self.manager.stop(alice)

  async def test_stream_slot(self):
    alice = self.player('alice', 'AAA#1')
    bob = self.player('bob', 'BBB#2')
    self.assertTrue((await self.manager.play(alice)).on_stream)
    self.assertFalse((await self.manager.play(bob)).on_stream)
    await self.manager.stop(alice)
    await self.manager.stop(bob)
    self.assertTrue((await self.manager.play(bob)).on_stream)

  async def test_connect_code_in_one_session(self):
    await self.manager.play(self.player('alice', 'AAA#1'))
    with self.assertRaisesRegex(UserError, 'already playing'):
      await self.manager.play(self.player('bob', 'aaa#1'))

  async def test_max_sessions(self):
    for i in range(self.max_sessions):
      await self.manager.play(self.player(f'p{i}', f'P{i}#1'))
    with self.assertRaisesRegex(UserError, 'too many sessions'):
      await self.manager.play(self.player('late', 'LATE#1'))

  async def test_select_agent_restarts(self):
    alice = self.player('alice', 'AAA#1')
    result = await self.manager.select_agent(alice, 'marth_vs_fox')
    self.assertIsNone(result.restarted)

    await self.manager.play(alice)
    result = await self.manager.select_agent(alice, 'fox_vs_marth')
    self.assertIsNotNone(result.restarted)
    self.assertEqual(result.restarted.agent, 'fox_vs_marth')
    first, second = self.human_sessions()
    self.assertFalse(first.alive)
    self.assertEqual(second.args[1].name, 'fox_vs_marth')

  async def test_stages_and_stocks(self):
    alice = self.player('alice', 'AAA#1')
    self.manager.set_stages(alice, [melee.Stage.BATTLEFIELD])
    self.manager.set_sd_stocks(alice, 2)
    with self.assertRaises(UserError):
      self.manager.set_sd_stocks(alice, 5)
    await self.manager.play(alice)
    session, = self.human_sessions()
    self.assertEqual(session.kwargs['stages'], [melee.Stage.BATTLEFIELD])
    self.assertEqual(session.kwargs['sd_stocks'], 2)

    self.manager.set_sd_stocks(alice, manager_lib.MAX_STOCKS)
    self.assertIsNone(self.manager.sd_stocks(alice))
    self.manager.set_stages(alice, None)
    self.assertIsNone(self.manager.stages(alice))

  async def test_botmatch(self):
    alice = self.player('alice')
    label = await self.manager.botmatch(alice, 'fox_vs_falco', 'marth_vs_fox')
    self.assertEqual(label, 'fox_vs_falco vs marth_vs_fox')
    session, = self.sessions
    self.assertEqual(session.kind, 'bot')
    self.assertEqual(session.kwargs['max_games'], 5)
    with self.assertRaisesRegex(UserError, 'Unknown agent'):
      await self.manager.botmatch(self.player('bob'), 'nope', 'fox_vs_falco')

  async def test_kick(self):
    await self.manager.play(self.player('alice', 'AAA#1'))
    kicked = await self.manager.kick('alice')
    self.assertEqual(kicked.name, 'alice')
    with self.assertRaises(UserError):
      await self.manager.kick('alice')

  async def test_status(self):
    alice = self.player('alice', 'AAA#1')
    await self.manager.play(alice)
    self.human_sessions()[0].num_menu_frames = 120
    status = await self.manager.status()
    info, = status.sessions
    self.assertEqual(info.user, alice)
    self.assertTrue(info.on_stream)
    self.assertEqual(info.menu_time.total_seconds(), 2)

  async def test_status_while_starting(self):
    await self.manager.play(self.player('alice', 'AAA#1'))
    self.human_sessions()[0].starting = True
    info, = (await self.manager.status()).sessions
    self.assertIsNone(info.menu_time)

  async def test_session_logs(self):
    alice = self.player('alice', 'AAA#1')
    self.assertIsNone(await self.manager.session_logs(alice, 0))
    await self.manager.play(alice)
    logs = await self.manager.session_logs(alice, 0)
    self.assertEqual(logs.lines, ['session started'])
    self.assertEqual(logs.session_id, self.manager.session_id(alice))
    self.assertEqual((await self.manager.session_logs(alice, 1)).lines, [])


class StopTest(ManagerTestCase):

  async def test_hung_stop_kills_dolphin(self):
    alice = self.player('alice', 'AAA#1')
    await self.manager.play(alice)
    session, = self.human_sessions()
    session.hang_stop = True

    # Returns after STOP_TIMEOUT rather than waiting for the session.
    await asyncio.wait_for(self.manager.stop(alice), timeout=1)
    self.assertTrue(session.killed)
    self.assertIsNone(self.manager.session_id(alice))
    # The stream slot is free again.
    bob = self.player('bob', 'BBB#2')
    self.assertTrue((await self.manager.play(bob)).on_stream)

    # The session finishes stopping in the background.
    await asyncio.gather(*self.manager._background_stops)
    self.assertFalse(session.alive)

  async def test_failed_stop(self):
    alice = self.player('alice', 'AAA#1')
    await self.manager.play(alice)
    self.human_sessions()[0].crashed = True
    with self.assertLogs(level='ERROR'):
      await self.manager.stop(alice)
    self.assertIsNone(self.manager.session_id(alice))

  async def test_shutdown_kills_hung_sessions(self):
    # shutdown() blocks on ray.wait, which is faked here.
    await self.manager.play(self.player('alice', 'AAA#1'))
    session, = self.human_sessions()
    session.hang_stop = True

    waits = []

    def fake_wait(refs, num_returns, timeout):
      waits.append(len(refs))
      # The first wait times out on the hung session; then it is killed.
      pending = [r for r in refs if not session.killed]
      return [r for r in refs if r not in pending], pending

    self._patch(manager_lib.ray, 'wait', fake_wait)
    self.manager.shutdown()
    self.assertTrue(session.killed)
    self.assertEqual(waits, [1, 1])


class GcTest(ManagerTestCase):

  async def test_gc(self):
    for name in ['idle', 'dead', 'crashed']:
      await self.manager.play(self.player(name, f'{name.upper()}#1'))
    idle_session, dead_session, crashed_session = self.human_sessions()
    idle_session.num_menu_frames = 60 * 60 * 10  # 10 minutes in the menu
    dead_session.alive = False
    crashed_session.crashed = True

    with self.assertLogs(level='ERROR'):
      gced = await self.manager.gc()
    self.assertCountEqual([u.name for u in gced], ['idle', 'dead', 'crashed'])
    self.assertEqual(
        self.announcements,
        ['Stopped idle session against idle, dead, crashed'])

  async def test_gc_skips_starting_sessions(self):
    alice = self.player('alice', 'AAA#1')
    await self.manager.play(alice)
    self.human_sessions()[0].starting = True
    self.assertEqual(await self.manager.gc(), [])
    self.assertIsNotNone(self.manager.session_id(alice))


class BotsTest(ManagerTestCase):

  bot_session_interval = 0.0

  async def test_bot_session(self):
    self.assertTrue(await self.manager.start_bot_session())
    bot, = self.sessions
    self.assertEqual(bot.kind, 'bot')
    self.assertEqual(
        self.announcements, ['Started fox_vs_falco vs. fox_vs_falco on stream.'])

    # A player on stream replaces the bots.
    await self.manager.play(self.player('alice', 'AAA#1'))
    self.assertFalse(bot.alive)
    self.assertFalse(await self.manager.start_bot_session())

  async def test_bots_lock(self):
    alice, bob = self.player('alice'), self.player('bob')
    await self.manager.set_bots(alice, 'marth_vs_fox', 'fox_vs_marth')
    status = await self.manager.status()
    self.assertEqual(status.bots.bot1, 'marth_vs_fox')
    self.assertEqual(status.bots.set_by, 'alice')
    self.assertIsNotNone(status.bots.lock_remaining)

    with self.assertRaisesRegex(UserError, 'has the bots'):
      await self.manager.set_bots(bob, 'fox_vs_falco', 'fox_vs_falco')
    with self.assertRaisesRegex(UserError, 'Only alice'):
      await self.manager.relinquish(bob)
    # Mods can override the lock.
    await self.manager.set_bots(bob, 'fox_vs_falco', 'fox_vs_falco', is_mod=True)

    await self.manager.relinquish(bob)
    with self.assertRaisesRegex(UserError, 'aren\'t locked'):
      await self.manager.relinquish(bob)
    await self.manager.set_bots(alice, 'marth_vs_fox', 'marth_vs_fox')


class PrefsMigrationTest(ManagerTestCase):

  user_prefs = {'alice': {'agent': 'marth_vs_fox', 'connect_code': 'AAA#1'}}

  def test_twitch_prefs_migrated(self):
    alice = User(frontend='twitch', key='alice', name='alice')
    self.assertEqual(self.manager.selected_agent(alice), 'marth_vs_fox')
    self.assertEqual(self.manager.connect_code(alice), 'AAA#1')
    with open(os.path.join(self.state_dir, 'user_prefs.json')) as f:
      self.assertEqual(list(json.load(f)), ['twitch:alice'])


class FakeClock:

  def __init__(self):
    self.now = 1e9

  def __call__(self) -> float:
    return self.now

class UserPrefsTest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.path = os.path.join(self.tmp.name, 'prefs.json')
    self.clock = FakeClock()

  def tearDown(self):
    self.tmp.cleanup()

  def load(self) -> UserPrefs:
    return UserPrefs(self.path, clock=self.clock)

  def saved(self) -> dict:
    with open(self.path) as f:
      return json.load(f)

  def test_set_records_last_seen(self):
    prefs = self.load()
    prefs.set('web:a', 'agent', 'fox')
    self.assertEqual(self.saved(), {'web:a': {'agent': 'fox', LAST_SEEN: 1e9}})
    self.assertEqual(self.load().get('web:a', 'agent'), 'fox')

  def test_touch(self):
    prefs = self.load()
    prefs.touch('web:unknown')  # Users without prefs aren't added.
    prefs.set('web:a', 'agent', 'fox')

    self.clock.now += TOUCH_INTERVAL / 2
    prefs.touch('web:a')
    self.assertEqual(self.saved()['web:a'][LAST_SEEN], 1e9)

    self.clock.now += TOUCH_INTERVAL
    prefs.touch('web:a')
    self.assertEqual(self.saved()['web:a'][LAST_SEEN], self.clock.now)
    self.assertNotIn('web:unknown', self.saved())

  def test_prune(self):
    prefs = self.load()
    prefs.set('web:old', 'agent', 'fox')
    prefs.set('twitch:old', 'agent', 'fox')
    self.clock.now += 10 * DAY
    prefs.set('web:new', 'agent', 'fox')

    is_web = lambda user: user.startswith('web:')
    self.assertEqual(prefs.prune(is_web, max_age=5 * DAY), ['web:old'])
    self.assertCountEqual(self.saved(), ['web:new', 'twitch:old'])
    self.assertEqual(prefs.prune(is_web, max_age=5 * DAY), [])

  def test_unseen_users_count_as_seen_on_load(self):
    with open(self.path, 'w') as f:
      json.dump({'web:a': {'agent': 'fox'}}, f)
    prefs = self.load()
    self.assertEqual(self.saved()['web:a'][LAST_SEEN], 1e9)
    self.clock.now += DAY
    self.assertEqual(prefs.prune(lambda _: True, max_age=2 * DAY), [])

  def test_delete_last_pref_removes_user(self):
    prefs = self.load()
    prefs.set('web:a', 'agent', 'fox')
    prefs.delete('web:a', 'agent')
    self.assertEqual(self.saved(), {})


class PrefsPruningTest(ManagerTestCase):

  user_prefs = {
      'web:' + 'o' * 32: {'connect_code': 'OLD#1', LAST_SEEN: 0},
      'web:' + 'n' * 32: {'connect_code': 'NEW#1'},
      # Seen a while ago, but not long enough to be pruned.
      'web:' + 'r' * 32: {
          'connect_code': 'RECENT#1', LAST_SEEN: time.time() - DAY},
      'twitch:old': {'connect_code': 'TWO#1', LAST_SEEN: 0},
  }
  prefs_max_days = {'web': 30}

  def user(self, frontend: str, key: str) -> User:
    return User(frontend=frontend, key=key, name=key)

  def test_old_web_users_pruned(self):
    self.assertIsNone(self.manager.connect_code(self.user('web', 'o' * 32)))
    self.assertEqual(
        self.manager.connect_code(self.user('web', 'n' * 32)), 'NEW#1')
    # Only web users are pruned.
    self.assertEqual(
        self.manager.connect_code(self.user('twitch', 'old')), 'TWO#1')

  async def test_playing_counts_as_seen(self):
    user = self.user('web', 'r' * 32)
    await self.manager.play(user)
    with open(os.path.join(self.state_dir, 'user_prefs.json')) as f:
      last_seen = json.load(f)[user.id][LAST_SEEN]
    self.assertGreater(last_seen, time.time() - 60)


class FakeContext:
  """Just enough of a twitchio Context for the commands."""

  def __init__(self, name: str, content: str, is_mod: bool = False):
    self.author = types.SimpleNamespace(name=name, is_mod=is_mod)
    self.message = types.SimpleNamespace(content=content)
    self.sent: list[str] = []

  async def send(self, message: str):
    self.sent.append(message)


class TwitchTest(ManagerTestCase):

  async def asyncSetUp(self):
    self.bot = twitchbot.Bot(
        token='test', prefix='!', channel='test', manager=self.manager)

  async def command(
      self, name: str, content: str, is_mod: bool = False) -> list[str]:
    ctx = FakeContext(name, content, is_mod)
    command = self.bot.commands[content.split(' ')[0][1:]]
    try:
      await command._callback(self.bot, ctx)
    except Exception as e:
      await self.bot.event_command_error(ctx, e)
    return ctx.sent

  async def test_play_status_stop(self):
    self.assertEqual(
        await self.command('alice', '!play abc#123'),
        ['Connecting to alice (ABC#123) on stream.'])
    self.assertEqual(
        await self.command('alice', '!play'), ['alice, you are already playing'])
    status = await self.command('bob', '!status')
    self.assertEqual(status[0], 'Selected agent: fox_vs_falco')
    self.assertRegex(status[1], 'Playing against alice as fox_vs_falco on stream')
    self.assertEqual(
        await self.command('alice', '!stop'), ['Stopped playing against alice'])

  async def test_agents(self):
    sent = await self.command('alice', '!agents fox falco')
    self.assertIn('fox_vs_falco', ' '.join(sent))
    self.assertEqual(
        await self.command('alice', '!agents zzz'), ['Unknown character "zzz".'])

  async def test_kick_needs_mod(self):
    await self.command('alice', '!play AAA#1')
    self.assertEqual(
        await self.command('bob', '!kick alice'), ['Only mods can kick players.'])
    self.assertEqual(
        await self.command('bob', '!kick alice', is_mod=True), ['Kicked alice'])


TOKEN = 'a' * 32
OTHER_TOKEN = 'b' * 32

class WebTest(ManagerTestCase):

  async def asyncSetUp(self):
    frontend = web_lib.WebFrontend(self.manager, twitch_channel='test')
    self.client = test_utils.TestClient(test_utils.TestServer(frontend.app))
    await self.client.start_server()

  async def asyncTearDown(self):
    await self.client.close()

  async def call(
      self, method: str, path: str, token: tp.Optional[str] = TOKEN,
      body: tp.Optional[dict] = None,
  ) -> tuple[int, tp.Any]:
    headers = {web_lib.TOKEN_HEADER: token} if token else {}
    response = await self.client.request(method, path, headers=headers, json=body)
    return response.status, await response.json()

  async def test_index_and_info(self):
    response = await self.client.get('/')
    self.assertEqual(response.status, 200)
    self.assertIn('Play phillip', await response.text())
    status, info = await self.call('GET', '/api/info', token=None)
    self.assertEqual(status, 200)
    self.assertEqual(info['bot_connect_code'], BOT_CONNECT_CODE)

  async def test_agents(self):
    status, rows = await self.call('GET', '/api/agents')
    self.assertEqual(status, 200)
    row, = [r for r in rows if r['name'] == 'fox_vs_falco']
    self.assertEqual(row['character'], 'fox')
    self.assertEqual(row['opponents'], ['falco'])

  async def test_token_required(self):
    for token in [None, 'short', 'x' * 100, 'not a token!' * 2]:
      status, data = await self.call('GET', '/api/me', token=token)
      self.assertEqual(status, 400)
      self.assertIn('token', data['error'])

  async def test_prefs(self):
    status, me = await self.call('GET', '/api/me')
    self.assertEqual(status, 200)
    self.assertIsNone(me['connect_code'])
    self.assertTrue(me['name'].startswith('web-'))
    self.assertNotIn(TOKEN, me['name'])

    status, data = await self.call('POST', '/api/me', body=dict(
        connect_code='abc#123', agent='marth_vs_fox', stages=['fd', 'bf'],
        sd_stocks=2))
    self.assertEqual(status, 200)
    self.assertEqual(data['me'], dict(
        name='ABC#123', connect_code='ABC#123', agent='marth_vs_fox',
        stages=['fd', 'bf'], sd_stocks=2))

    for body in [dict(agent='nope'), dict(stages=['xx']), dict(sd_stocks='2'),
                 dict(connect_code='ABC123')]:
      status, data = await self.call('POST', '/api/me', body=body)
      self.assertEqual(status, 400, body)
      self.assertIn('error', data)

    status, data = await self.call('POST', '/api/me', body=dict(stages=[]))
    self.assertIsNone(data['me']['stages'])

  async def test_play_stop_reset(self):
    status, data = await self.call('POST', '/api/play')
    self.assertEqual(status, 400)
    await self.call('POST', '/api/me', body=dict(connect_code='AAA#1'))
    status, data = await self.call('POST', '/api/play')
    self.assertEqual(status, 200)
    self.assertEqual(data['connect_code'], 'AAA#1')
    status, data = await self.call('POST', '/api/reset')
    self.assertEqual(status, 200)
    self.assertEqual(len(self.human_sessions()), 2)
    status, _ = await self.call('POST', '/api/stop')
    self.assertEqual(status, 200)
    status, _ = await self.call('POST', '/api/stop')
    self.assertEqual(status, 400)

  async def test_shares_sessions_with_twitch(self):
    await self.manager.play(self.player('alice', 'AAA#1'))
    await self.call('POST', '/api/me', body=dict(connect_code='AAA#1'))
    status, data = await self.call('POST', '/api/play')
    self.assertEqual(status, 400)
    self.assertIn('already playing', data['error'])

  async def test_events(self):
    await self.call('POST', '/api/me', body=dict(connect_code='AAA#1'))
    await self.call(
        'POST', '/api/me', token=OTHER_TOKEN, body=dict(connect_code='BBB#2'))
    await self.call('POST', '/api/play')
    await self.call('POST', '/api/play', token=OTHER_TOKEN)

    response = await self.client.get(f'/api/events?token={TOKEN}')
    self.assertEqual(response.headers['Content-Type'], 'text/event-stream')
    events: dict[str, list] = {}
    raw = ''
    async def read():
      nonlocal raw
      event = None
      async for line in response.content:
        line = line.decode()
        raw += line
        if line.startswith('event: '):
          event = line.removeprefix('event: ').strip()
        elif line.startswith('data: '):
          events.setdefault(event, []).append(json.loads(line[6:]))
        if 'status' in events and any(e['lines'] for e in events.get('logs', [])):
          return
    await asyncio.wait_for(read(), timeout=5)
    response.close()

    sessions = {s['name']: s for s in events['status'][0]['sessions']}
    self.assertTrue(sessions['AAA#1']['is_me'])
    self.assertFalse(sessions['BBB#2']['is_me'])
    self.assertEqual(events['logs'][0], dict(reset=True, lines=[]))
    self.assertEqual(events['logs'][1]['lines'], ['session started'])
    # Tokens are secret: never sent in events.
    self.assertNotIn(TOKEN, raw)
    self.assertNotIn(OTHER_TOKEN, raw)


if __name__ == '__main__':
  unittest.main()
