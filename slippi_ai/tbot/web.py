"""Web frontend to the SessionManager.

Users have no accounts: the page makes a random token, keeps it in the
browser and sends it with each request. The token is the user's key, so it is
never sent to anyone else; other users see the user's connect code instead.
"""

import asyncio
import dataclasses
import datetime
import hashlib
import json
import logging
import os
import re
import typing as tp
from typing import Optional

from aiohttp import web
import melee

from slippi_ai.tbot import manager as manager_lib
from slippi_ai.tbot.manager import SessionManager, User, UserError

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

TOKEN_HEADER = 'X-Tbot-Token'
TOKEN_RE = re.compile(r'^[A-Za-z0-9_-]{16,64}$')

# How often the status and the session logs are pushed to the page.
POLL_INTERVAL = 1  # seconds
# Comments sent on otherwise idle event streams so proxies keep them open.
KEEPALIVE_INTERVAL = 15  # seconds

def character_name(character: melee.Character) -> str:
  return character.name.lower()

def seconds(td: Optional[datetime.timedelta]) -> Optional[float]:
  return None if td is None else td.total_seconds()

def json_response(data: tp.Any, status: int = 200) -> web.Response:
  return web.json_response(data, status=status)

def error_response(message: str, status: int = 400) -> web.Response:
  return json_response({'error': message}, status=status)

def play_json(result: manager_lib.PlayResult) -> dict:
  return dataclasses.asdict(result)

class WebFrontend:

  def __init__(self, manager: SessionManager, twitch_channel: Optional[str]):
    self.manager = manager
    self.twitch_channel = twitch_channel

    # One poller shares the status among all the open event streams.
    self._status: Optional[manager_lib.ManagerStatus] = None
    self._status_version = 0
    self._status_task: Optional[asyncio.Task] = None
    self._subscribers: set[asyncio.Queue] = set()

    manager.add_listener(self._announce)

    self.app = web.Application(middlewares=[self._errors])
    self.app.add_routes([
        web.get('/', self._index),
        web.static('/static', STATIC_DIR),
        web.get('/api/info', self._info),
        web.get('/api/agents', self._agents),
        web.get('/api/me', self._get_me),
        web.post('/api/me', self._set_me),
        web.post('/api/play', self._play),
        web.post('/api/stop', self._stop),
        web.post('/api/reset', self._reset),
        web.get('/api/events', self._events),
    ])

  @web.middleware
  async def _errors(self, request: web.Request, handler):
    try:
      return await handler(request)
    except UserError as e:
      return error_response(str(e))
    except web.HTTPException:
      raise
    except Exception:
      logging.exception(f'Error handling {request.path}')
      return error_response('Internal error.', status=500)

  # Users

  def _user(self, request: web.Request) -> User:
    token = request.headers.get(TOKEN_HEADER) or request.query.get('token')
    if token is None or not TOKEN_RE.match(token):
      raise UserError('Missing or invalid token; try reloading the page.')

    user = User(frontend='web', key=token, name='')
    connect_code = self.manager.connect_code(user)
    if connect_code is None:
      # A stable name that doesn't reveal the token.
      tag = hashlib.sha256(token.encode()).hexdigest()[:6]
      name = f'web-{tag}'
    else:
      name = connect_code
    return dataclasses.replace(user, name=name)

  def _me_json(self, user: User) -> dict:
    stages = self.manager.stages(user)
    stage_names = {
        stage: name for name, stage in manager_lib.NAME_TO_STAGE.items()}
    return dict(
        name=user.name,
        connect_code=self.manager.connect_code(user),
        agent=self.manager.selected_agent(user),
        stages=None if stages is None else [stage_names[s] for s in stages],
        sd_stocks=self.manager.sd_stocks(user),
    )

  async def _json_body(self, request: web.Request) -> dict:
    try:
      body = await request.json()
    except json.JSONDecodeError:
      raise UserError('Invalid JSON.')
    if not isinstance(body, dict):
      raise UserError('Expected a JSON object.')
    return body

  # Pages and static info

  async def _index(self, request: web.Request) -> web.FileResponse:
    return web.FileResponse(os.path.join(STATIC_DIR, 'index.html'))

  async def _info(self, request: web.Request) -> web.Response:
    return json_response(dict(
        bot_connect_code=self.manager.bot_connect_code,
        max_sessions=self.manager.max_sessions,
        stages=[s for s in manager_lib.NAME_TO_STAGE if s != 'all'],
        max_stocks=manager_lib.MAX_STOCKS,
        twitch_channel=self.twitch_channel,
    ))

  async def _agents(self, request: web.Request) -> web.Response:
    rows = []
    for row in self.manager.agent_table():
      rows.append(dict(
          name=row.name,
          kind=row.kind,
          character=character_name(row.character),
          opponents=[character_name(c) for c in row.opponents],
          delay=row.delay,
          model=row.model,
          rating=row.rating,
      ))
    return json_response(rows)

  # Preferences

  async def _get_me(self, request: web.Request) -> web.Response:
    # Called when the page loads, so it marks the user as seen.
    user = self._user(request)
    self.manager.touch(user)
    return json_response(self._me_json(user))

  async def _set_me(self, request: web.Request) -> web.Response:
    user = self._user(request)
    body = await self._json_body(request)

    if 'connect_code' in body:
      code = body['connect_code']
      if not isinstance(code, str):
        raise UserError('Invalid connect code.')
      self.manager.set_connect_code(user, code.strip())
      user = self._user(request)  # The name may have changed.

    if 'stages' in body:
      names = body['stages']
      if not isinstance(names, list) or not all(
          name in manager_lib.NAME_TO_STAGE for name in names):
        raise UserError('Invalid stages.')
      if names and 'all' not in names:
        self.manager.set_stages(
            user, [manager_lib.NAME_TO_STAGE[name] for name in names])
      else:
        self.manager.set_stages(user, None)  # All stages.

    if 'sd_stocks' in body:
      stocks = body['sd_stocks']
      if not isinstance(stocks, int):
        raise UserError('Invalid number of stocks.')
      self.manager.set_sd_stocks(user, stocks)

    restarted = None
    if 'agent' in body:
      agent = body['agent']
      if not isinstance(agent, str):
        raise UserError('Invalid agent.')
      result = await self.manager.select_agent(user, agent)
      if result.restarted is not None:
        restarted = play_json(result.restarted)

    return json_response(dict(me=self._me_json(user), restarted=restarted))

  # Sessions

  async def _play(self, request: web.Request) -> web.Response:
    result = await self.manager.play(self._user(request))
    return json_response(play_json(result))

  async def _stop(self, request: web.Request) -> web.Response:
    await self.manager.stop(self._user(request))
    return json_response({})

  async def _reset(self, request: web.Request) -> web.Response:
    result = await self.manager.reset(self._user(request))
    return json_response(play_json(result))

  # Live updates

  async def _announce(self, message: str):
    for queue in self._subscribers:
      try:
        queue.put_nowait(message)
      except asyncio.QueueFull:
        pass  # A stalled client misses some announcements.

  async def _poll_status(self):
    """Refreshes the shared status while anyone is listening."""
    while self._subscribers:
      try:
        self._status = await self.manager.status()
        self._status_version += 1
      except Exception:
        logging.exception('Failed to get the status')
      await asyncio.sleep(POLL_INTERVAL)
    self._status_task = None

  def _status_json(self, user: User) -> dict:
    status = self._status
    assert status is not None

    bots = None
    if status.bots is not None:
      bots = dict(
          bot1=status.bots.bot1,
          bot2=status.bots.bot2,
          set_by=status.bots.set_by,
          playing_for=seconds(status.bots.playing_for),
          lock_remaining=seconds(status.bots.lock_remaining),
      )

    sessions = []
    for s in status.sessions:
      sessions.append(dict(
          name=s.user.name,
          is_me=s.user == user,
          agent=s.agent,
          duration=seconds(s.duration),
          botmatch=s.connect_code is None,
          on_stream=s.on_stream,
          menu_time=seconds(s.menu_time),
          games_completed=s.games_completed,
      ))

    return dict(bots=bots, sessions=sessions)

  async def _events(self, request: web.Request) -> web.StreamResponse:
    """Server-sent events: status, announcements and the user's logs."""
    user = self._user(request)

    response = web.StreamResponse(headers={
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',  # Don't let nginx buffer the stream.
    })
    await response.prepare(request)

    async def send(event: str, data: tp.Any):
      await response.write(
          f'event: {event}\ndata: {json.dumps(data)}\n\n'.encode())

    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
    self._subscribers.add(queue)
    if self._status_task is None:
      self._status_task = asyncio.create_task(self._poll_status())

    status_version = None
    log_session: Optional[int] = None
    log_next = 0
    loop = asyncio.get_running_loop()
    last_write = loop.time()

    try:
      while True:
        # Announcements are sent as soon as they happen.
        try:
          message = await asyncio.wait_for(queue.get(), timeout=POLL_INTERVAL)
          await send('announce', message)
          last_write = loop.time()
          continue
        except asyncio.TimeoutError:
          pass

        if self._status is not None and status_version != self._status_version:
          status_version = self._status_version
          await send('status', self._status_json(user))
          last_write = loop.time()

        # The logs of a finished session stay on the page until the next.
        session_id = self.manager.session_id(user)
        if session_id is not None and session_id != log_session:
          log_session = session_id
          log_next = 0
          await send('logs', dict(reset=True, lines=[]))

        if session_id is not None:
          try:
            logs = await asyncio.wait_for(
                self.manager.session_logs(user, log_next),
                timeout=POLL_INTERVAL)
          except asyncio.TimeoutError:
            logs = None  # The session is still starting up.

          if logs is not None and logs.session_id == log_session and logs.lines:
            log_next = logs.next
            await send('logs', dict(reset=False, lines=logs.lines))
            last_write = loop.time()

        if loop.time() - last_write > KEEPALIVE_INTERVAL:
          await response.write(b': keepalive\n\n')
          last_write = loop.time()
    except (ConnectionResetError, asyncio.CancelledError):
      pass
    finally:
      self._subscribers.discard(queue)

    return response

async def start(
    frontend: WebFrontend, host: str, port: int) -> web.AppRunner:
  """Starts serving; call runner.cleanup() to stop."""
  runner = web.AppRunner(frontend.app)
  await runner.setup()
  site = web.TCPSite(runner, host, port)
  await site.start()
  logging.info(f'Web frontend listening on http://{host}:{port}')
  return runner
