"""Tkinter launcher for slippi-ai scripts. See docs/superpowers/specs/2026-08-01-slippi-ai-launcher-gui-design.md."""

import asyncio
import collections
import dataclasses
import json
import os
import pathlib
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Hardcoded to avoid importing melee at startup (slow first-open).
# These MUST match melee.Character enum member names exactly.
# fancyflags (eval_two/run_evaluator) accepts these uppercase names as-is;
# absl.flags.DEFINE_enum_class (netplay's --char) accepts the same names lowercased.
CHARACTERS = [
    'FOX', 'FALCO', 'MARTH', 'SHEIK', 'JIGGLYPUFF', 'CPTFALCON',
    'PEACH', 'POPO', 'NANA', 'PIKACHU', 'SAMUS', 'DOC', 'YOSHI',
    'LUIGI', 'GANONDORF', 'MARIO', 'YLINK', 'LINK', 'DK',
    'GAMEANDWATCH', 'MEWTWO', 'ROY', 'PICHU', 'NESS', 'BOWSER',
    'KIRBY', 'ZELDA',
]
PLAYER_TYPES = ['ai', 'human', 'cpu']
MODELS_DIR = REPO_ROOT / 'models'

_CONNECT_CODE_RE = re.compile(r'^[A-Z0-9]{2,6}#\d{1,6}$')


def validate_connect_code(code: str) -> bool:
  return bool(_CONNECT_CODE_RE.match(code or ''))


def validate_supported_character(char: str, supported: list[str]) -> bool:
  return char in supported


# Matches Windows / macOS / Linux user-home path prefixes, e.g.
# `C:\Users\alice`, `C:/Users/alice`, `/Users/alice`, `/home/alice`.
_HOME_PATH_RE = re.compile(
    r'(?:[A-Za-z]:[\\/])?(?:Users|home)[\\/][^\\/\s"\'`]+', re.IGNORECASE)


def scrub_for_public(text: str) -> str:
  """Redact filesystem paths and Python-traceback frame lines from `text`
  before sending it to a public channel (Discord etc.)."""
  out_lines = []
  for line in text.splitlines(keepends=True):
    stripped = line.strip()
    # Drop `File "..."` frame lines and their caret indicators — they leak
    # both absolute paths and internal structure without adding much value.
    if stripped.startswith('File "') and '", line ' in stripped:
      continue
    if stripped and set(stripped) <= {'^', '~'}:
      continue
    out_lines.append(_HOME_PATH_RE.sub('~', line))
  return ''.join(out_lines)


def list_models() -> list[str]:
  if not MODELS_DIR.is_dir():
    return []
  # Models are pickled state files, not directories.
  return sorted(p.name for p in MODELS_DIR.iterdir() if p.is_file())


def _apply_windows_dark_titlebar(root: tk.Tk) -> None:
  """Ask Windows' DWM to render this window's title bar in dark mode.
  No-op on non-Windows or older Windows versions."""
  if sys.platform != 'win32':
    return
  import ctypes

  def _try_set():
    try:
      # Ensure the window has been realized so its HWND is valid.
      root.update_idletasks()
      # Try both HWND resolutions — depending on how the Tk build wraps
      # the toplevel, either the winfo_id itself or its parent is the
      # HWND that owns the title bar.
      candidates = [root.winfo_id()]
      parent = ctypes.windll.user32.GetParent(root.winfo_id())
      if parent:
        candidates.append(parent)
      value = ctypes.c_int(1)
      dwm = ctypes.windll.dwmapi
      for hwnd in candidates:
        for attr in (20, 19):
          hr = dwm.DwmSetWindowAttribute(
              hwnd, attr, ctypes.byref(value), ctypes.sizeof(value))
          if hr == 0:
            # Force a redraw so the title-bar color updates immediately.
            root.withdraw()
            root.deiconify()
            return
      # If we got here, nothing worked. Print once so it shows in the
      # console output when the user runs launcher.bat.
      print(f'[launcher] dark title bar not applied '
            f'(hwnds={candidates}); Windows version may not support it.',
            file=sys.stderr)
    except Exception as e:
      print(f'[launcher] dark title bar error: {e}', file=sys.stderr)

  # Defer to after the window is fully mapped — some Windows builds
  # ignore the attribute if set before the window is on screen.
  root.after(50, _try_set)


class Config:

  def __init__(self):
    self.global_: dict = {}
    self.tabs: dict[str, dict] = {}
    self.last_tab: str = ''

  @staticmethod
  def path() -> pathlib.Path:
    return pathlib.Path(os.path.expanduser('~')) / '.slippi_ai_launcher.json'

  @classmethod
  def load(cls) -> 'Config':
    cfg = cls()
    try:
      data = json.loads(cls.path().read_text())
    except (OSError, ValueError):
      return cfg
    if isinstance(data, dict):
      cfg.global_ = data.get('global', {}) if isinstance(data.get('global'), dict) else {}
      cfg.tabs = data.get('tabs', {}) if isinstance(data.get('tabs'), dict) else {}
      cfg.last_tab = data.get('last_tab', '') if isinstance(data.get('last_tab'), str) else ''
    return cfg

  def save(self) -> None:
    data = {'global': self.global_, 'tabs': self.tabs, 'last_tab': self.last_tab}
    self.path().write_text(json.dumps(data, indent=2))


@dataclasses.dataclass
class MatchRequest:
  user_id: int
  user_name: str
  channel_id: int
  connect_code: str
  character: str
  started_at: float


class DiscordBotThread:

  def __init__(self, app: 'LauncherApp', status_cb):
    self._app = app
    self._status_cb = status_cb
    self._thread: threading.Thread | None = None
    self._loop: asyncio.AbstractEventLoop | None = None
    self._client = None  # discord.Client, created inside the thread
    self._active: 'MatchRequest | None' = None
    self._timeout_task = None
    self._allowed_channels: list[int] = []
    self._model_path: str = ''
    self._user_json_path: str = ''
    self._character_choices: list[str] = []

  @property
  def is_running(self) -> bool:
    return self._thread is not None and self._thread.is_alive()

  def start(self, token: str, allowed_channels: list[int], model_path: str,
            user_json_path: str, character_choices: list[str]) -> None:
    if self.is_running:
      raise RuntimeError('Discord bot already running.')
    self._allowed_channels = list(allowed_channels)
    self._model_path = model_path
    self._user_json_path = user_json_path
    self._character_choices = list(character_choices)
    self._thread = threading.Thread(target=self._run, args=(token,), daemon=True)
    self._thread.start()

  def stop(self) -> None:
    if self._loop is None or self._client is None:
      return
    client = self._client
    discord_mod = self._discord

    async def _graceful_shutdown():
      # Setting presence to invisible before closing propagates the offline
      # state to Discord clients immediately, instead of waiting ~30-60 s
      # for the server-side heartbeat to time out.
      try:
        await client.change_presence(status=discord_mod.Status.invisible)
      except Exception:
        pass
      await client.close()

    fut = asyncio.run_coroutine_threadsafe(_graceful_shutdown(), self._loop)
    try:
      fut.result(timeout=5)
    except Exception:
      pass

  def _post_status(self, text: str) -> None:
    self._app.root.after(0, self._status_cb, text)

  def _log(self, text: str) -> None:
    """Mirror bot events into the launcher's log panel for visibility."""
    self._app.root.after(0, self._app.log.append, f'[discord] {text}\n', 'meta')

  def _bot_connect_code(self) -> str:
    """Read the bot's own connect code from its user.json for display."""
    try:
      with open(self._user_json_path) as f:
        return json.load(f).get('connectCode', '?')
    except (OSError, ValueError):
      return '?'

  async def _set_ready_presence(self) -> None:
    if self._client is None:
      return
    activity = self._discord.Game(name='Ready — @ me <code> <char>')
    await self._client.change_presence(status=self._discord.Status.online, activity=activity)

  async def _set_busy_presence(self, user_name: str, char: str) -> None:
    if self._client is None:
      return
    activity = self._discord.Game(name=f'vs @{user_name} ({char})')
    await self._client.change_presence(status=self._discord.Status.dnd, activity=activity)

  def _run(self, token: str) -> None:
    import discord
    self._loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self._loop)
    intents = discord.Intents.default()
    intents.message_content = True  # Required for @-mention command parsing.
    self._client = discord.Client(intents=intents)
    self._discord = discord  # so helpers can access discord.Status / discord.Game

    @self._client.event
    async def on_ready():
      self._post_status(f'connected as {self._client.user}')
      await self._set_ready_presence()

    @self._client.event
    async def on_disconnect():
      self._post_status('disconnected — reconnecting…')

    @self._client.event
    async def on_resumed():
      self._post_status(f'connected as {self._client.user}')
      await self._set_ready_presence()

    @self._client.event
    async def on_message(message):
      try:
        await _handle_message(message)
      except Exception as e:
        self._log(f'on_message error: {type(e).__name__}: {e}')

    def _bot_is_addressed(message):
      # Direct user mention.
      if self._client.user in message.mentions:
        return True
      # Role mention where the role belongs to the bot (e.g. an auto-created
      # bot role with the same display name — Discord's autocomplete offers
      # both the user and the role, and users often pick the role).
      if message.guild is None:
        return False
      bot_member = message.guild.get_member(self._client.user.id)
      if bot_member is None:
        return False
      return any(role in bot_member.roles for role in message.role_mentions)

    async def _handle_message(message):
      if message.author == self._client.user:
        return
      mentioned = _bot_is_addressed(message)
      self._log(
          f'msg from {message.author} in channel={message.channel.id} '
          f'mentioned={mentioned} content={message.content!r}')
      if not mentioned:
        return
      if message.channel.id not in self._allowed_channels:
        self._log(
            f'channel {message.channel.id} not in allowlist '
            f'{self._allowed_channels} — ignoring')
        return
      # Strip mention tokens (raw form is <@NNN>, <@!NNN>, or <@&NNN> for roles)
      # and split remainder.
      tokens = message.content.split()
      args = [t for t in tokens if not (t.startswith('<@') and t.endswith('>'))]
      self._log(f'parsed args={args}')
      if len(args) != 2:
        bot_name = self._client.user.display_name
        chars = ', '.join(self._character_choices) or '(none configured)'
        await message.channel.send(
            f'Syntax: `@{bot_name} <connect_code> <character>` '
            f'— e.g. `@{bot_name} TNBN#217 fox`\n'
            f'Available characters: {chars}')
        return
      code, char = args
      await self._handle_play(message.channel, message.author, code, char.lower())

    self._log('starting client')
    try:
      self._loop.run_until_complete(self._client.start(token))
    except discord.LoginFailure:
      self._post_status('bad token')
    except Exception as e:
      self._post_status(f'error: {e}')
    finally:
      self._log('client closing (bot going offline)')
      try:
        self._loop.run_until_complete(self._client.close())
      except Exception:
        pass
      self._loop.close()
      self._loop = None
      self._client = None
      self._post_status('stopped')

  async def _handle_play(self, channel, user, code: str, char: str) -> None:
    # Channel allowlist — silent drop if not allowed.
    if channel.id not in self._allowed_channels:
      return
    # Busy guard.
    if self._active is not None:
      await channel.send(
          f'Bot busy — @{self._active.user_name} is playing vs `{self._active.connect_code}`.')
      return
    # Validate.
    if not validate_connect_code(code):
      await channel.send('Invalid code — expected `ABCD#123`.')
      return
    if not validate_supported_character(char, self._character_choices):
      await channel.send(
          f'Model does not support `{char}`. Options: {", ".join(self._character_choices)}')
      return
    # Claim slot.
    request = MatchRequest(
        user_id=user.id,
        user_name=user.display_name,
        channel_id=channel.id,
        connect_code=code,
        character=char,
        started_at=time.monotonic(),
    )
    self._active = request
    self._app.root.after(0, self._set_slot_label,
                        f'match slot: running (@{request.user_name} vs {code})')
    await self._set_busy_presence(user.display_name, char)
    bot_code = self._bot_connect_code()
    await channel.send(
        f'**@{user.display_name}** vs `{code}` as `{char}` — starting…\n'
        f'Enter my code in your Slippi game (Direct Connect): **`{bot_code}`**')
    # Build argv and spawn (from Tk main thread — ProcessRunner uses root.after internally).
    tab_values = {
        'model_path': self._model_path,
        'char': char,
        'costume': '',
        'connect_code': code,
        'runtime': '',
        'user_json_path': self._user_json_path,
    }
    global_paths = self._app.global_paths.values()
    argv = build_netplay_argv(global_paths, tab_values)

    def on_exit_tk(exit_code: int) -> None:
      # Runs on Tk main thread; bounce to discord loop.
      recent = self._app.log.recent_lines()
      if self._loop is not None:
        asyncio.run_coroutine_threadsafe(
            self._on_match_ended(channel, exit_code, recent), self._loop)

    def start_runner():
      # Install the marker watcher before spawn.
      self._app.log.on_line = self._watch_stdout
      try:
        self._app.runner.start(argv, cwd=REPO_ROOT, on_exit=on_exit_tk)
      except RuntimeError as e:
        # ProcessRunner refused (another script running). Report + clear slot.
        if self._loop is not None:
          asyncio.run_coroutine_threadsafe(
              self._on_spawn_failed(channel, str(e)), self._loop)
    self._app.root.after(0, start_runner)
    # Timeout for "opponent never joined".
    try:
      timeout_s = int(self._app.config.tabs.get('discord', {}).get('connect_timeout_s', '600'))
    except ValueError:
      timeout_s = 600
    self._timeout_task = self._loop.create_task(self._connect_timeout(channel, timeout_s))

  def _watch_stdout(self, line: str, kind: str) -> None:
    # Called on Tk main thread by LogPanel.append.
    if '[NETPLAY_MATCH_STARTED]' in line and self._loop is not None:
      asyncio.run_coroutine_threadsafe(self._on_match_started(), self._loop)

  async def _on_match_started(self) -> None:
    if self._timeout_task is not None and not self._timeout_task.done():
      self._timeout_task.cancel()
      self._timeout_task = None
    if self._active is None:
      return
    channel = self._client.get_channel(self._active.channel_id)
    if channel is not None:
      await channel.send('**match live**')

  async def _connect_timeout(self, channel, timeout_s: int) -> None:
    try:
      await asyncio.sleep(timeout_s)
    except asyncio.CancelledError:
      return
    # Timeout fired without a match-started marker — kill it.
    self._app.root.after(0, self._app.runner.stop)
    try:
      await channel.send('Opponent never joined — timing out.')
    except Exception:
      pass

  async def _on_match_ended(self, channel, exit_code: int, recent_lines: list[str]) -> None:
    # Cancel connect timeout if still active.
    if self._timeout_task is not None and not self._timeout_task.done():
      self._timeout_task.cancel()
    self._timeout_task = None
    # Clear slot + UI first so a busy reply after this is honest.
    self._active = None
    self._app.root.after(0, self._set_slot_label, 'match slot: idle')
    self._app.root.after(0, self._clear_log_watcher)
    await self._set_ready_presence()
    if exit_code == 0:
      await channel.send(f'Match ended (exit code 0).')
    else:
      tail = scrub_for_public(''.join(recent_lines[-10:]))
      msg = f'Match failed (exit code {exit_code}).'
      if tail.strip():
        msg += f'\n```\n{tail[-1500:]}\n```'
      await channel.send(msg)

  async def _on_spawn_failed(self, channel, err: str) -> None:
    # Cancel connect timeout if still active.
    if self._timeout_task is not None and not self._timeout_task.done():
      self._timeout_task.cancel()
    self._timeout_task = None
    self._active = None
    self._app.root.after(0, self._set_slot_label, 'match slot: idle')
    self._app.root.after(0, self._clear_log_watcher)
    await self._set_ready_presence()
    await channel.send(f'Failed to spawn netplay: {scrub_for_public(err)}')

  def _set_slot_label(self, text: str) -> None:
    for tab in self._app._discord_tabs():
      tab.slot_label.configure(text=text)

  def _clear_log_watcher(self) -> None:
    if self._app.log.on_line is self._watch_stdout:
      self._app.log.on_line = None


class LogPanel(ttk.Frame):

  _RECENT_MAX = 20

  def __init__(self, parent):
    super().__init__(parent)
    self.text = tk.Text(self, height=15, wrap='none', state='disabled', background='#1e1e1e', foreground='#d4d4d4', insertbackground='#d4d4d4')
    yscroll = ttk.Scrollbar(self, orient='vertical', command=self.text.yview)
    self.text.configure(yscrollcommand=yscroll.set)
    self.text.grid(row=0, column=0, sticky='nsew')
    yscroll.grid(row=0, column=1, sticky='ns')
    self.grid_rowconfigure(0, weight=1)
    self.grid_columnconfigure(0, weight=1)
    self.text.tag_config('stderr', foreground='#f48771')
    self.text.tag_config('meta', foreground='#888888')
    self.on_line: 'Callable[[str, str], None] | None' = None
    self._recent: collections.deque = collections.deque(maxlen=self._RECENT_MAX)

  def append(self, text: str, kind: str = 'stdout') -> None:
    self.text.configure(state='normal')
    self.text.insert('end', text, kind if kind != 'stdout' else ())
    self.text.see('end')
    self.text.configure(state='disabled')
    self._recent.append(text)
    if self.on_line is not None:
      try:
        self.on_line(text, kind)
      except Exception as e:
        # Observer failure must not break logging.
        print(f'LogPanel.on_line raised: {e}', file=sys.stderr)

  def recent_lines(self) -> list[str]:
    return list(self._recent)


class ProcessRunner:

  _POLL_MS = 50
  _STOP_GRACE_S = 3.0
  _TERMINATE_GRACE_S = 2.0

  def __init__(self, root: tk.Tk, log: LogPanel):
    self._root = root
    self._log = log
    self._proc: subprocess.Popen | None = None
    self._queue: queue.Queue = queue.Queue()
    self._on_exit = None
    self._draining = False

  @property
  def is_running(self) -> bool:
    return self._proc is not None and self._proc.poll() is None

  def start(self, argv: list[str], cwd: pathlib.Path, on_exit) -> None:
    if self.is_running:
      raise RuntimeError('A process is already running.')
    self._on_exit = on_exit
    self._log.append('Running: ' + ' '.join(argv) + '\n', kind='meta')
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
    try:
      self._proc = subprocess.Popen(
          argv,
          cwd=str(cwd),
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          bufsize=1,
          text=True,
          creationflags=creationflags,
      )
    except OSError as e:
      self._log.append(f'Failed to start: {e}\n', kind='stderr')
      if self._on_exit:
        self._on_exit(-1)
      return
    threading.Thread(target=self._reader, daemon=True).start()
    if not self._draining:
      self._draining = True
      self._root.after(self._POLL_MS, self._drain)

  def _reader(self) -> None:
    assert self._proc is not None and self._proc.stdout is not None
    for line in self._proc.stdout:
      self._queue.put(('stdout', line))
    self._proc.wait()
    self._queue.put(('__exit__', self._proc.returncode))

  def _drain(self) -> None:
    while True:
      try:
        kind, payload = self._queue.get_nowait()
      except queue.Empty:
        break
      if kind == '__exit__':
        self._log.append(f'Exit code: {payload}\n', kind='meta')
        cb = self._on_exit
        self._on_exit = None
        if cb:
          cb(payload)
      else:
        self._log.append(payload, kind=kind)
    if self.is_running or not self._queue.empty():
      self._root.after(self._POLL_MS, self._drain)
    else:
      self._draining = False

  def stop(self) -> None:
    if not self.is_running:
      return
    assert self._proc is not None
    try:
      if sys.platform == 'win32':
        self._proc.send_signal(signal.CTRL_BREAK_EVENT)
      else:
        self._proc.terminate()
    except OSError:
      pass
    self._root.after(int(self._STOP_GRACE_S * 1000), self._escalate_terminate)

  def _escalate_terminate(self) -> None:
    if not self.is_running:
      return
    assert self._proc is not None
    try:
      self._proc.terminate()
    except OSError:
      pass
    self._root.after(int(self._TERMINATE_GRACE_S * 1000), self._escalate_kill)

  def _escalate_kill(self) -> None:
    if not self.is_running:
      return
    assert self._proc is not None
    try:
      self._proc.kill()
    except OSError:
      pass


class AdvancedSection(ttk.Frame):
  """A checkbutton-toggled container. Access `.body` to pack widgets into."""

  def __init__(self, parent, text: str = 'Advanced'):
    super().__init__(parent)
    self._shown = tk.BooleanVar(value=False)
    self._toggle = ttk.Checkbutton(self, text=text, variable=self._shown, command=self._refresh)
    self._toggle.pack(anchor='w')
    self.body = ttk.Frame(self)
    # Not packed initially; _refresh handles it.

  def _refresh(self):
    if self._shown.get():
      self.body.pack(fill='x', padx=16, pady=(2, 4))
    else:
      self.body.pack_forget()


class GlobalPathsFrame(ttk.LabelFrame):

  def __init__(self, parent, initial: dict):
    super().__init__(parent, text='Global paths')
    self.dolphin_var = tk.StringVar(value=initial.get('dolphin_path', ''))
    self.iso_var = tk.StringVar(value=initial.get('iso_path', ''))
    self._row('Dolphin:', self.dolphin_var, 0, self._pick_dolphin)
    self._row('ISO:', self.iso_var, 1, self._pick_iso)
    self.grid_columnconfigure(1, weight=1)

  def _row(self, label, var, row, browse_cmd):
    ttk.Label(self, text=label).grid(row=row, column=0, sticky='w', padx=4, pady=2)
    ttk.Entry(self, textvariable=var).grid(row=row, column=1, sticky='ew', padx=4, pady=2)
    ttk.Button(self, text='Browse…', command=browse_cmd).grid(row=row, column=2, padx=4, pady=2)

  def _pick_dolphin(self):
    p = filedialog.askopenfilename(title='Select Slippi Dolphin.exe', filetypes=[('Executable', '*.exe'), ('All files', '*.*')])
    if p:
      self.dolphin_var.set(p)

  def _pick_iso(self):
    p = filedialog.askopenfilename(title='Select SSBM ISO', filetypes=[('ISO', '*.iso'), ('All files', '*.*')])
    if p:
      self.iso_var.set(p)

  def values(self) -> dict:
    return {'dolphin_path': self.dolphin_var.get(), 'iso_path': self.iso_var.get()}


class PlayerFrame(ttk.LabelFrame):

  def __init__(self, parent, label: str, initial: dict):
    super().__init__(parent, text=label)
    self.type_var = tk.StringVar(value=initial.get('type', 'ai'))
    self.char_var = tk.StringVar(value=initial.get('character', 'FOX'))
    self.model_var = tk.StringVar(value=initial.get('model_path', ''))
    self.level_var = tk.StringVar(value=initial.get('cpu_level', '9'))

    ttk.Label(self, text='Type:').grid(row=0, column=0, sticky='w', padx=4, pady=2)
    ttk.Combobox(self, textvariable=self.type_var, values=PLAYER_TYPES, state='readonly', width=8).grid(row=0, column=1, sticky='w', padx=4, pady=2)

    self.char_label = ttk.Label(self, text='Character:')
    self.char_combo = ttk.Combobox(self, textvariable=self.char_var, values=CHARACTERS, state='readonly', width=18)

    self.model_label = ttk.Label(self, text='Model:')
    self.model_combo = ttk.Combobox(self, textvariable=self.model_var, values=list_models(), width=30)
    self.model_browse = ttk.Button(self, text='Browse…', command=self._pick_model)

    self.level_label = ttk.Label(self, text='CPU Level:')
    self.level_spin = ttk.Spinbox(self, from_=1, to=9, textvariable=self.level_var, width=4)

    self._layout()
    self.type_var.trace_add('write', lambda *_: self._layout())

  def _pick_model(self):
    p = filedialog.askopenfilename(title='Select model file', initialdir=str(MODELS_DIR) if MODELS_DIR.is_dir() else None)
    if p:
      self.model_var.set(p)

  def _layout(self):
    for w in (self.char_label, self.char_combo, self.model_label, self.model_combo, self.model_browse, self.level_label, self.level_spin):
      w.grid_forget()
    t = self.type_var.get()
    row = 1
    if t in ('ai', 'cpu'):
      self.char_label.grid(row=row, column=0, sticky='w', padx=4, pady=2)
      self.char_combo.grid(row=row, column=1, sticky='w', padx=4, pady=2)
      row += 1
    if t == 'ai':
      self.model_label.grid(row=row, column=0, sticky='w', padx=4, pady=2)
      self.model_combo.grid(row=row, column=1, sticky='ew', padx=4, pady=2)
      self.model_browse.grid(row=row, column=2, padx=4, pady=2)
      row += 1
    if t == 'cpu':
      self.level_label.grid(row=row, column=0, sticky='w', padx=4, pady=2)
      self.level_spin.grid(row=row, column=1, sticky='w', padx=4, pady=2)

  def values(self) -> dict:
    return {
        'type': self.type_var.get(),
        'character': self.char_var.get(),
        'model_path': self.model_var.get(),
        'cpu_level': self.level_var.get(),
    }


class ScriptTab(ttk.Frame):
  """Abstract base for one script's tab."""

  TAB_KEY: str = ''  # subclass sets, e.g. 'eval_two'
  SCRIPT: str = ''   # e.g. 'scripts/eval_two.py'
  LABEL: str = ''    # notebook tab label

  def __init__(self, parent, app):
    super().__init__(parent)
    self.app = app
    self._error_label = ttk.Label(self, foreground='#c00000', wraplength=800, justify='left')
    self._build_widgets()
    self._build_controls()

  # Subclasses override:
  def _build_widgets(self) -> None: raise NotImplementedError
  def _values(self) -> dict: raise NotImplementedError

  @staticmethod
  def build_argv(global_paths: dict, tab_values: dict) -> list[str]:
    raise NotImplementedError

  @staticmethod
  def validate(global_paths: dict, tab_values: dict) -> list[str]:
    raise NotImplementedError

  def _build_controls(self):
    controls = ttk.Frame(self)
    self.run_btn = ttk.Button(controls, text='Run', command=self._on_run)
    self.stop_btn = ttk.Button(controls, text='Stop', command=self._on_stop, state='disabled')
    self.status = ttk.Label(controls, text='idle')
    self.run_btn.pack(side='left', padx=4)
    self.stop_btn.pack(side='left', padx=4)
    self.status.pack(side='left', padx=12)
    controls.pack(fill='x', pady=(8, 4))
    self._error_label.pack(fill='x', pady=(0, 4))

  def _on_run(self):
    global_paths = self.app.global_paths.values()
    tab_values = self._values()
    errors = self.validate(global_paths, tab_values)
    if errors:
      self._error_label.configure(text='\n'.join(errors))
      return
    self._error_label.configure(text='')
    # Persist config snapshot.
    self.app.config.global_ = global_paths
    self.app.config.tabs[self.TAB_KEY] = tab_values
    self.app.config.last_tab = self.TAB_KEY
    self.app.config.save()
    argv = self.build_argv(global_paths, tab_values)
    self.run_btn.configure(state='disabled')
    self.stop_btn.configure(state='normal')
    self.status.configure(text='running')
    self.app.runner.start(argv, cwd=REPO_ROOT, on_exit=self._on_exit)

  def _on_stop(self):
    self.app.runner.stop()

  def _on_exit(self, code: int):
    self.run_btn.configure(state='normal')
    self.stop_btn.configure(state='disabled')
    self.status.configure(text=f'exited (code {code})')


def _add_player_flags(args: list[str], prefix: str, p: dict, advanced: dict | None = None) -> None:
  """Append fancyflags for one player. `advanced` applies only when type=ai."""
  t = p.get('type', 'ai')
  args.append(f'--{prefix}.type={t}')
  if t in ('ai', 'cpu'):
    args.append(f'--{prefix}.character={p.get("character", "FOX")}')
  if t == 'ai' and p.get('model_path'):
    args.append(f'--{prefix}.ai.path={p["model_path"]}')
  if t == 'cpu':
    args.append(f'--{prefix}.level={p.get("cpu_level", "9")}')
  if t == 'ai' and advanced:
    st = str(advanced.get('sample_temperature', '')).strip()
    if st:
      args.append(f'--{prefix}.ai.sample_temperature={st}')
    args.append(f'--{prefix}.ai.async_inference={"true" if advanced.get("async_inference") else "false"}')
    name = str(advanced.get('name', '')).strip()
    if name:
      args.append(f'--{prefix}.ai.name={name}')
    args.append(f'--{prefix}.ai.mirror={"true" if advanced.get("mirror") else "false"}')


class EvalTwoTab(ScriptTab):

  TAB_KEY = 'eval_two'
  SCRIPT = 'scripts/eval_two.py'
  LABEL = 'eval_two'

  def _build_widgets(self):
    initial = self.app.config.tabs.get(self.TAB_KEY, {})
    row = ttk.Frame(self)
    self.p1 = PlayerFrame(row, 'Player 1', initial.get('p1', {'type': 'human'}))
    self.p2 = PlayerFrame(row, 'Player 2', initial.get('p2', {'type': 'ai', 'character': 'FALCO'}))
    self.p1.pack(side='left', fill='both', expand=True, padx=4, pady=4)
    self.p2.pack(side='left', fill='both', expand=True, padx=4, pady=4)
    row.pack(fill='x')

    ngf = ttk.Frame(self)
    ttk.Label(ngf, text='Num games:').pack(side='left', padx=4)
    self.num_games_var = tk.StringVar(value=initial.get('num_games', ''))
    ttk.Entry(ngf, textvariable=self.num_games_var, width=8).pack(side='left', padx=4)
    ttk.Label(ngf, text='(blank = infinite)', foreground='#666666').pack(side='left', padx=4)
    ngf.pack(fill='x', pady=4)

    # Advanced (collapsible) — applies to whichever players are AI.
    adv = initial.get('advanced', {})
    self.sample_temperature = tk.StringVar(value=adv.get('sample_temperature', '1.0'))
    self.async_inference = tk.BooleanVar(value=bool(adv.get('async_inference', True)))
    self.agent_name = tk.StringVar(value=adv.get('name', ''))
    self.mirror = tk.BooleanVar(value=bool(adv.get('mirror', False)))
    adv_sec = AdvancedSection(self)
    body = adv_sec.body
    def _adv_entry(text, var, width=10):
      f = ttk.Frame(body); ttk.Label(f, text=text).pack(side='left', padx=4)
      ttk.Entry(f, textvariable=var, width=width).pack(side='left', padx=4); f.pack(anchor='w')
    _adv_entry('sample_temperature:', self.sample_temperature)
    _adv_entry('name (blank = default):', self.agent_name, width=20)
    ttk.Checkbutton(body, text='async_inference', variable=self.async_inference).pack(anchor='w')
    ttk.Checkbutton(body, text='mirror (flip x axis)', variable=self.mirror).pack(anchor='w')
    adv_sec.pack(fill='x', pady=4)

  def _values(self) -> dict:
    return {
        'p1': self.p1.values(),
        'p2': self.p2.values(),
        'num_games': self.num_games_var.get().strip(),
        'advanced': {
            'sample_temperature': self.sample_temperature.get().strip(),
            'async_inference': self.async_inference.get(),
            'name': self.agent_name.get().strip(),
            'mirror': self.mirror.get(),
        },
    }

  @staticmethod
  def build_argv(global_paths: dict, tab_values: dict) -> list[str]:
    argv = [sys.executable, str(REPO_ROOT / 'scripts' / 'eval_two.py')]
    adv = tab_values.get('advanced', {})
    _add_player_flags(argv, 'p1', tab_values['p1'], adv)
    _add_player_flags(argv, 'p2', tab_values['p2'], adv)
    argv.append(f'--dolphin.path={global_paths.get("dolphin_path", "")}')
    argv.append(f'--dolphin.iso={global_paths.get("iso_path", "")}')
    ng = tab_values.get('num_games', '').strip()
    if ng:
      argv.append(f'--num_games={ng}')
    return argv

  @staticmethod
  def validate(global_paths: dict, tab_values: dict) -> list[str]:
    errors = []
    if not global_paths.get('dolphin_path'):
      errors.append('Dolphin path is required.')
    elif not pathlib.Path(global_paths['dolphin_path']).is_file():
      errors.append(f'Dolphin path does not exist: {global_paths["dolphin_path"]}')
    if not global_paths.get('iso_path'):
      errors.append('ISO path is required.')
    elif not pathlib.Path(global_paths['iso_path']).is_file():
      errors.append(f'ISO path does not exist: {global_paths["iso_path"]}')
    if tab_values['p1']['type'] == 'human' and tab_values['p2']['type'] == 'human':
      errors.append('At least one player must be AI or CPU (both cannot be human).')
    for name, p in (('Player 1', tab_values['p1']), ('Player 2', tab_values['p2'])):
      if p['type'] == 'ai':
        if not p.get('model_path'):
          errors.append(f'{name}: model path is required when type=ai.')
        elif not pathlib.Path(p['model_path']).exists():
          errors.append(f'{name}: model path does not exist: {p["model_path"]}')
    return errors


class RunDolphinTab(ScriptTab):

  TAB_KEY = 'run_dolphin'
  SCRIPT = 'scripts/run_dolphin.py'
  LABEL = 'run_dolphin'

  def _build_widgets(self):
    initial = self.app.config.tabs.get(self.TAB_KEY, {})
    self.n_var = tk.StringVar(value=initial.get('N', '1'))
    self.frames_var = tk.StringVar(value=initial.get('frames', '3600'))
    self.render_var = tk.BooleanVar(value=bool(initial.get('render', False)))

    grid = ttk.Frame(self)
    ttk.Label(grid, text='N (instances):').grid(row=0, column=0, sticky='w', padx=4, pady=2)
    ttk.Spinbox(grid, from_=1, to=32, textvariable=self.n_var, width=6).grid(row=0, column=1, sticky='w', padx=4, pady=2)
    ttk.Label(grid, text='Frames:').grid(row=1, column=0, sticky='w', padx=4, pady=2)
    ttk.Entry(grid, textvariable=self.frames_var, width=10).grid(row=1, column=1, sticky='w', padx=4, pady=2)
    ttk.Checkbutton(grid, text='Render graphics', variable=self.render_var).grid(row=2, column=0, columnspan=2, sticky='w', padx=4, pady=2)
    grid.pack(fill='x')

  def _values(self) -> dict:
    return {'N': self.n_var.get(), 'frames': self.frames_var.get(), 'render': self.render_var.get()}

  @staticmethod
  def build_argv(global_paths: dict, tab_values: dict) -> list[str]:
    argv = [sys.executable, str(REPO_ROOT / 'scripts' / 'run_dolphin.py')]
    argv.append(f'--N={tab_values.get("N", "1")}')
    argv.append(f'--frames={tab_values.get("frames", "3600")}')
    argv.append(f'--render={"true" if tab_values.get("render") else "false"}')
    argv.append(f'--dolphin.path={global_paths.get("dolphin_path", "")}')
    argv.append(f'--dolphin.iso={global_paths.get("iso_path", "")}')
    return argv

  @staticmethod
  def validate(global_paths: dict, tab_values: dict) -> list[str]:
    errors = []
    if not global_paths.get('dolphin_path'):
      errors.append('Dolphin path is required.')
    elif not pathlib.Path(global_paths['dolphin_path']).is_file():
      errors.append(f'Dolphin path does not exist: {global_paths["dolphin_path"]}')
    if not global_paths.get('iso_path'):
      errors.append('ISO path is required.')
    elif not pathlib.Path(global_paths['iso_path']).is_file():
      errors.append(f'ISO path does not exist: {global_paths["iso_path"]}')
    return errors


class RunEvaluatorTab(ScriptTab):

  TAB_KEY = 'run_evaluator'
  SCRIPT = 'scripts/run_evaluator.py'
  LABEL = 'run_evaluator'

  def _build_widgets(self):
    initial = self.app.config.tabs.get(self.TAB_KEY, {})
    self.player_model = tk.StringVar(value=initial.get('player', {}).get('model_path', ''))
    self.player_char = tk.StringVar(value=initial.get('player', {}).get('character', 'FOX'))
    self.opp_model = tk.StringVar(value=initial.get('opponent', {}).get('model_path', ''))
    self.opp_char = tk.StringVar(value=initial.get('opponent', {}).get('character', 'FALCO'))
    self.self_play = tk.BooleanVar(value=bool(initial.get('self_play', False)))
    self.num_envs = tk.StringVar(value=initial.get('num_envs', '4'))
    self.rollout_length = tk.StringVar(value=initial.get('rollout_length', '3600'))
    self.num_games = tk.StringVar(value=initial.get('num_games', ''))
    # Advanced:
    self.use_gpu = tk.BooleanVar(value=bool(initial.get('use_gpu', True)))
    self.async_envs = tk.BooleanVar(value=bool(initial.get('async_envs', False)))
    self.sim_envs = tk.BooleanVar(value=bool(initial.get('sim_envs', False)))
    self.fake_envs = tk.BooleanVar(value=bool(initial.get('fake_envs', False)))
    self.swap_ports = tk.BooleanVar(value=bool(initial.get('swap_ports', True)))
    self.quiet = tk.BooleanVar(value=bool(initial.get('quiet', False)))
    self.burnin = tk.BooleanVar(value=bool(initial.get('burnin', False)))
    self.num_env_steps = tk.StringVar(value=initial.get('num_env_steps', '0'))
    self.inner_batch_size = tk.StringVar(value=initial.get('inner_batch_size', '1'))
    self.num_agent_steps = tk.StringVar(value=initial.get('num_agent_steps', '0'))

    row = 0
    grid = ttk.Frame(self)
    def label_entry(text, var, width=30, r=None):
      nonlocal row
      r = row if r is None else r
      ttk.Label(grid, text=text).grid(row=r, column=0, sticky='w', padx=4, pady=2)
      ttk.Entry(grid, textvariable=var, width=width).grid(row=r, column=1, sticky='ew', padx=4, pady=2)
      row = r + 1
    def label_combo(text, var, values, width=18):
      nonlocal row
      ttk.Label(grid, text=text).grid(row=row, column=0, sticky='w', padx=4, pady=2)
      ttk.Combobox(grid, textvariable=var, values=values, state='readonly', width=width).grid(row=row, column=1, sticky='w', padx=4, pady=2)
      row += 1

    label_entry('Player model path:', self.player_model, width=40)
    label_combo('Player character:', self.player_char, CHARACTERS)
    ttk.Checkbutton(grid, text='Self play (use player for both)', variable=self.self_play).grid(row=row, column=0, columnspan=2, sticky='w', padx=4, pady=2); row += 1
    label_entry('Opponent model path:', self.opp_model, width=40)
    label_combo('Opponent character:', self.opp_char, CHARACTERS)
    label_entry('Num envs:', self.num_envs, width=8)
    label_entry('Rollout length:', self.rollout_length, width=8)
    label_entry('Num games (blank = infinite):', self.num_games, width=8)
    grid.pack(fill='x')

    adv_sec = AdvancedSection(self)
    body = adv_sec.body
    for var, text in [
        (self.use_gpu, 'use_gpu'),
        (self.async_envs, 'async_envs'),
        (self.sim_envs, 'sim_envs'),
        (self.fake_envs, 'fake_envs'),
        (self.swap_ports, 'swap_ports'),
        (self.quiet, 'quiet'),
        (self.burnin, 'burnin'),
    ]:
      ttk.Checkbutton(body, text=text, variable=var).pack(anchor='w', padx=4)
    for var, text in [
        (self.num_env_steps, 'num_env_steps'),
        (self.inner_batch_size, 'inner_batch_size'),
        (self.num_agent_steps, 'num_agent_steps'),
    ]:
      f = ttk.Frame(body)
      ttk.Label(f, text=text + ':').pack(side='left', padx=4)
      ttk.Entry(f, textvariable=var, width=8).pack(side='left', padx=4)
      f.pack(anchor='w')
    adv_sec.pack(fill='x', pady=6)

  def _values(self) -> dict:
    return {
        'player': {'model_path': self.player_model.get(), 'character': self.player_char.get()},
        'opponent': {'model_path': self.opp_model.get(), 'character': self.opp_char.get()},
        'self_play': self.self_play.get(),
        'num_envs': self.num_envs.get(),
        'rollout_length': self.rollout_length.get(),
        'num_games': self.num_games.get().strip(),
        'use_gpu': self.use_gpu.get(),
        'async_envs': self.async_envs.get(),
        'sim_envs': self.sim_envs.get(),
        'fake_envs': self.fake_envs.get(),
        'swap_ports': self.swap_ports.get(),
        'quiet': self.quiet.get(),
        'burnin': self.burnin.get(),
        'num_env_steps': self.num_env_steps.get(),
        'inner_batch_size': self.inner_batch_size.get(),
        'num_agent_steps': self.num_agent_steps.get(),
    }

  @staticmethod
  def build_argv(global_paths: dict, tab_values: dict) -> list[str]:
    argv = [sys.executable, str(REPO_ROOT / 'scripts' / 'run_evaluator.py')]
    p = tab_values['player']
    argv.append(f'--player.ai.path={p["model_path"]}')
    argv.append(f'--player.character={p["character"]}')
    self_play = bool(tab_values.get('self_play'))
    argv.append(f'--self_play={"true" if self_play else "false"}')
    if not self_play:
      o = tab_values['opponent']
      argv.append(f'--opponent.ai.path={o["model_path"]}')
      argv.append(f'--opponent.character={o["character"]}')
    argv.append(f'--num_envs={tab_values.get("num_envs", "1")}')
    argv.append(f'--rollout_length={tab_values.get("rollout_length", "3600")}')
    ng = str(tab_values.get('num_games', '')).strip()
    if ng:
      argv.append(f'--num_games={ng}')
    for key in ('use_gpu', 'async_envs', 'sim_envs', 'fake_envs', 'swap_ports', 'quiet', 'burnin'):
      argv.append(f'--{key}={"true" if tab_values.get(key) else "false"}')
    for key in ('num_env_steps', 'inner_batch_size', 'num_agent_steps'):
      argv.append(f'--{key}={tab_values.get(key, "0")}')
    argv.append(f'--dolphin.path={global_paths.get("dolphin_path", "")}')
    argv.append(f'--dolphin.iso={global_paths.get("iso_path", "")}')
    return argv

  @staticmethod
  def validate(global_paths: dict, tab_values: dict) -> list[str]:
    errors = []
    if not global_paths.get('dolphin_path'):
      errors.append('Dolphin path is required.')
    if not global_paths.get('iso_path'):
      errors.append('ISO path is required.')
    if not tab_values['player'].get('model_path'):
      errors.append('Player: model path is required.')
    elif not pathlib.Path(tab_values['player']['model_path']).exists():
      errors.append(f'Player: model path does not exist: {tab_values["player"]["model_path"]}')
    if not tab_values.get('self_play'):
      if not tab_values['opponent'].get('model_path'):
        errors.append('Opponent: model path is required (or enable self-play).')
      elif not pathlib.Path(tab_values['opponent']['model_path']).exists():
        errors.append(f'Opponent: model path does not exist: {tab_values["opponent"]["model_path"]}')
    return errors


def build_netplay_argv(global_paths: dict, tab_values: dict) -> list[str]:
  argv = [sys.executable, str(REPO_ROOT / 'scripts' / 'netplay.py')]
  argv.append(f'--agent.path={tab_values.get("model_path", "")}')
  argv.append(f'--char={tab_values.get("char", "FOX").lower()}')
  if tab_values.get('costume'):
    argv.append(f'--costume={tab_values["costume"]}')
  argv.append(f'--dolphin.path={global_paths.get("dolphin_path", "")}')
  argv.append(f'--dolphin.iso={global_paths.get("iso_path", "")}')
  argv.append(f'--dolphin.connect_code={tab_values.get("connect_code", "")}')
  if tab_values.get('user_json_path'):
    argv.append(f'--dolphin.user_json_path={tab_values["user_json_path"]}')
  if tab_values.get('runtime'):
    argv.append(f'--runtime={tab_values["runtime"]}')
  return argv


class NetplayTab(ScriptTab):

  TAB_KEY = 'netplay'
  SCRIPT = 'scripts/netplay.py'
  LABEL = 'netplay'

  def _build_widgets(self):
    initial = self.app.config.tabs.get(self.TAB_KEY, {})
    self.model_var = tk.StringVar(value=initial.get('model_path', ''))
    self.char_var = tk.StringVar(value=initial.get('char', 'FOX'))
    self.costume_var = tk.StringVar(value=initial.get('costume', ''))
    self.connect_var = tk.StringVar(value=initial.get('connect_code', ''))
    self.runtime_var = tk.StringVar(value=initial.get('runtime', ''))
    self.user_json_var = tk.StringVar(value=initial.get('user_json_path', ''))

    grid = ttk.Frame(self)
    def row(label, widget, r):
      ttk.Label(grid, text=label).grid(row=r, column=0, sticky='w', padx=4, pady=2)
      widget.grid(row=r, column=1, sticky='ew', padx=4, pady=2)
    row('Model path:', ttk.Entry(grid, textvariable=self.model_var, width=40), 0)
    ttk.Button(grid, text='Browse…', command=self._pick_model).grid(row=0, column=2, padx=4)
    row('Character:', ttk.Combobox(grid, textvariable=self.char_var, values=CHARACTERS, state='readonly', width=18), 1)
    row('Costume (blank = default):', ttk.Entry(grid, textvariable=self.costume_var, width=6), 2)
    row('Connect code:', ttk.Entry(grid, textvariable=self.connect_var, width=16), 3)
    row('Runtime seconds (blank = forever):', ttk.Entry(grid, textvariable=self.runtime_var, width=8), 4)
    row('Slippi user.json:', ttk.Entry(grid, textvariable=self.user_json_var, width=40), 5)
    ttk.Button(grid, text='Browse…', command=self._pick_user_json).grid(row=5, column=2, padx=4)
    grid.pack(fill='x')

  def _pick_model(self):
    p = filedialog.askopenfilename(title='Select model file', initialdir=str(MODELS_DIR) if MODELS_DIR.is_dir() else None)
    if p:
      self.model_var.set(p)

  def _pick_user_json(self):
    p = filedialog.askopenfilename(title='Select Slippi user.json', filetypes=[('JSON', '*.json'), ('All files', '*.*')])
    if p:
      self.user_json_var.set(p)

  def _values(self) -> dict:
    return {
        'model_path': self.model_var.get(),
        'char': self.char_var.get(),
        'costume': self.costume_var.get().strip(),
        'connect_code': self.connect_var.get().strip(),
        'runtime': self.runtime_var.get().strip(),
        'user_json_path': self.user_json_var.get().strip(),
    }

  @staticmethod
  def build_argv(global_paths: dict, tab_values: dict) -> list[str]:
    return build_netplay_argv(global_paths, tab_values)

  @staticmethod
  def validate(global_paths: dict, tab_values: dict) -> list[str]:
    errors = []
    if not global_paths.get('dolphin_path'):
      errors.append('Dolphin path is required.')
    if not global_paths.get('iso_path'):
      errors.append('ISO path is required.')
    if not tab_values.get('model_path'):
      errors.append('Model path is required.')
    elif not pathlib.Path(tab_values['model_path']).exists():
      errors.append(f'Model path does not exist: {tab_values["model_path"]}')
    if not tab_values.get('connect_code'):
      errors.append('Connect code is required for netplay.')
    if not tab_values.get('user_json_path'):
      errors.append('Slippi user.json is required for netplay.')
    elif not pathlib.Path(tab_values['user_json_path']).is_file():
      errors.append(f'user.json does not exist: {tab_values["user_json_path"]}')
    return errors


class DiscordTab(ttk.Frame):

  TAB_KEY = 'discord'
  LABEL = 'discord'

  def __init__(self, parent, app):
    super().__init__(parent)
    self.app = app
    self._bot: DiscordBotThread | None = None
    initial = self.app.config.tabs.get(self.TAB_KEY, {})
    self._error_label = ttk.Label(self, foreground='#c00000', wraplength=800, justify='left')

    self.token_var = tk.StringVar(value=initial.get('token', ''))
    self.channels_var = tk.StringVar(value=initial.get('allowed_channels', ''))
    self.model_var = tk.StringVar(value=initial.get('model_path', ''))
    self.user_json_var = tk.StringVar(value=initial.get('user_json_path', ''))
    self.timeout_var = tk.StringVar(value=initial.get('connect_timeout_s', '600'))
    initial_chars = set(initial.get('supported_characters', []))
    self.char_vars: dict[str, tk.BooleanVar] = {
        c: tk.BooleanVar(value=(c.lower() in initial_chars))
        for c in CHARACTERS
    }

    grid = ttk.Frame(self)
    def row(label, widget, r, browse=None):
      ttk.Label(grid, text=label).grid(row=r, column=0, sticky='w', padx=4, pady=2)
      widget.grid(row=r, column=1, sticky='ew', padx=4, pady=2)
      if browse is not None:
        ttk.Button(grid, text='Browse…', command=browse).grid(row=r, column=2, padx=4)

    row('Bot token:', ttk.Entry(grid, textvariable=self.token_var, show='*', width=50), 0)
    row('Allowed channel IDs:', ttk.Entry(grid, textvariable=self.channels_var, width=50), 1)
    row('Model path:', ttk.Entry(grid, textvariable=self.model_var, width=50), 2, browse=self._pick_model)
    row('Slippi user.json:', ttk.Entry(grid, textvariable=self.user_json_var, width=50), 3, browse=self._pick_user_json)
    row('Connect timeout (s):', ttk.Entry(grid, textvariable=self.timeout_var, width=8), 4)
    grid.grid_columnconfigure(1, weight=1)
    grid.pack(fill='x')

    chars = ttk.LabelFrame(self, text='Supported characters (which your model can play)')
    for i, c in enumerate(CHARACTERS):
      ttk.Checkbutton(chars, text=c.lower(), variable=self.char_vars[c]).grid(
          row=i // 6, column=i % 6, sticky='w', padx=6, pady=2)
    chars.pack(fill='x', pady=4)

    controls = ttk.Frame(self)
    self.start_btn = ttk.Button(controls, text='Start bot', command=self._on_start)
    self.stop_btn = ttk.Button(controls, text='Stop bot', command=self._on_stop, state='disabled')
    self.status = ttk.Label(controls, text='stopped')
    self.slot_label = ttk.Label(controls, text='match slot: idle', foreground='#666666')
    self.start_btn.pack(side='left', padx=4)
    self.stop_btn.pack(side='left', padx=4)
    self.status.pack(side='left', padx=12)
    self.slot_label.pack(side='left', padx=12)
    controls.pack(fill='x', pady=(8, 4))
    self._error_label.pack(fill='x', pady=(0, 4))

  def _pick_model(self):
    p = filedialog.askopenfilename(title='Select model file', initialdir=str(MODELS_DIR) if MODELS_DIR.is_dir() else None)
    if p:
      self.model_var.set(p)

  def _pick_user_json(self):
    p = filedialog.askopenfilename(title='Select Slippi user.json', filetypes=[('JSON', '*.json'), ('All files', '*.*')])
    if p:
      self.user_json_var.set(p)

  def _values(self) -> dict:
    checked = sorted(c.lower() for c, v in self.char_vars.items() if v.get())
    return {
        'token': self.token_var.get(),
        'allowed_channels': self.channels_var.get().strip(),
        'model_path': self.model_var.get(),
        'user_json_path': self.user_json_var.get(),
        'connect_timeout_s': self.timeout_var.get().strip() or '600',
        'supported_characters': checked,
    }

  def _validate(self, values: dict) -> list[str]:
    errors = []
    if not values['token']:
      errors.append('Bot token is required.')
    channel_ids = self._parse_channel_ids(values['allowed_channels'])
    if channel_ids is None:
      errors.append('Allowed channel IDs must be comma-separated integers.')
    elif not channel_ids:
      errors.append('At least one allowed channel ID is required.')
    if not values['model_path']:
      errors.append('Model path is required.')
    elif not pathlib.Path(values['model_path']).exists():
      errors.append(f'Model path does not exist: {values["model_path"]}')
    if not values['user_json_path']:
      errors.append('Slippi user.json is required.')
    elif not pathlib.Path(values['user_json_path']).is_file():
      errors.append(f'user.json does not exist: {values["user_json_path"]}')
    if not values['supported_characters']:
      errors.append('At least one supported character must be checked.')
    try:
      t = int(values['connect_timeout_s'])
      if t <= 0:
        raise ValueError
    except ValueError:
      errors.append('Connect timeout must be a positive integer (seconds).')
    return errors

  @staticmethod
  def _parse_channel_ids(text: str) -> 'list[int] | None':
    if not text.strip():
      return []
    try:
      return [int(x.strip()) for x in text.split(',') if x.strip()]
    except ValueError:
      return None

  def _on_start(self):
    values = self._values()
    errors = self._validate(values)
    if errors:
      self._error_label.configure(text='\n'.join(errors))
      return
    self._error_label.configure(text='')
    self.app.config.tabs[self.TAB_KEY] = values
    self.app.config.save()
    self._bot = DiscordBotThread(self.app, self._update_status)
    self._bot.start(
        token=values['token'],
        allowed_channels=self._parse_channel_ids(values['allowed_channels']) or [],
        model_path=values['model_path'],
        user_json_path=values['user_json_path'],
        character_choices=values['supported_characters'],
    )
    self.start_btn.configure(state='disabled')
    self.stop_btn.configure(state='normal')
    self._update_status('starting…')

  def _on_stop(self):
    if self._bot is not None:
      self._bot.stop()
    # Poll for actual shutdown (bot thread posts 'stopped' via status callback).
    self.stop_btn.configure(state='disabled')

  def _update_status(self, text: str) -> None:
    self.status.configure(text=text)
    if text == 'stopped':
      self.start_btn.configure(state='normal')
      self.stop_btn.configure(state='disabled')


class LauncherApp:

  def __init__(self):
    self.root = tk.Tk()
    self.root.title('Slippi-AI Launcher')
    self.root.geometry('900x700')
    # Modern flat theme (Windows 11 Sun Valley style).
    try:
      import sv_ttk
      sv_ttk.set_theme('dark')
    except ImportError:
      pass  # Falls back to default ttk theme if sv-ttk isn't installed.
    _apply_windows_dark_titlebar(self.root)
    self.config = Config.load()
    self.global_paths = GlobalPathsFrame(self.root, initial=self.config.global_)
    self.global_paths.pack(fill='x', padx=8, pady=(8, 4))
    self.notebook = ttk.Notebook(self.root)
    self.notebook.pack(fill='both', expand=True, padx=8, pady=8)
    for tab_cls in (EvalTwoTab, RunDolphinTab, RunEvaluatorTab, NetplayTab, DiscordTab):
      tab = tab_cls(self.notebook, self)
      self.notebook.add(tab, text=tab_cls.LABEL)
    if self.config.last_tab:
      for i, tab in enumerate(self.notebook.tabs()):
        if self.notebook.tab(tab, 'text') == self.config.last_tab:
          self.notebook.select(i)
          break
    self.log = LogPanel(self.root)
    self.log.pack(fill='both', expand=False, padx=8, pady=(0, 8))
    self.runner = ProcessRunner(self.root, self.log)
    self._quit_requested = False
    self.root.protocol('WM_DELETE_WINDOW', self._on_close)

  def _on_close(self):
    if self.runner.is_running:
      if not self._quit_requested:
        from tkinter import messagebox
        if not messagebox.askyesno('Script running', 'A script is still running. Stop and quit?'):
          return
        self._quit_requested = True
        self.runner.stop()
      # Re-poll every 200 ms until the process has exited.
      self.root.after(200, self._on_close)
      return
    # Stop the Discord bot cleanly if any tab has one running.
    for tab in self._discord_tabs():
      if tab._bot is not None and tab._bot.is_running:
        tab._bot.stop()
    self.config.global_ = self.global_paths.values()
    self.config.save()
    self.root.destroy()

  def _discord_tabs(self):
    return [t for t in (self.notebook.nametowidget(name) for name in self.notebook.tabs())
            if isinstance(t, DiscordTab)]

  def run(self):
    self.root.mainloop()


if __name__ == '__main__':
  LauncherApp().run()
