# Slippi-AI Discord Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Discord tab to the existing launcher that exposes a `/play` slash command so friends can queue a netplay match against Sean's bot with `/play code:TNBN#217 char:falco`.

**Architecture:** New `DiscordTab` in `scripts/launcher.py` owns a `DiscordBotThread` running `discord.py` in a background asyncio loop. The bot reuses the launcher's existing `ProcessRunner` + `LogPanel` to spawn `netplay.py`. One-match-at-a-time slot guard, no queue. A distinctive marker line in `netplay.py` tells the bot when the match is actually live so it can cancel the "opponent never joined" timeout.

**Tech Stack:** Python 3 stdlib + `tkinter` (already used) + one new dep: `discord.py>=2.3`.

## Global Constraints

- **Two-space indentation** (project convention from `.claude/CLAUDE.md`).
- **One new dependency only:** `discord.py>=2.3`. No others.
- **Tests use `unittest.TestCase`** (matches repo convention, added to the existing `tests/test_launcher_command.py`). Run with `venv\Scripts\python.exe -m unittest tests.test_launcher_command -v`.
- **Cross-thread Tk calls forbidden.** The bot thread must call `self.app.root.after(0, callable, *args)` to touch any Tk widget or state. Tk main thread → bot must use `client.loop.call_soon_threadsafe(callable, *args)`.
- **Argv construction is shared.** Both `NetplayTab` and the discord bot call the module-level function `build_netplay_argv(global_paths, tab_values) -> list[str]`. Do not duplicate the logic.
- **Marker string:** exactly `[NETPLAY_MATCH_STARTED]` (with square brackets, no trailing punctuation). `flush=True` on the print — the buffered subprocess pipe will not deliver it otherwise.
- **Connect-code shape:** `^[A-Z0-9]{2,6}#\d{1,6}$`.
- **Slash-command choices are snapshotted at bot-startup.** Changing the character checkboxes while the bot is running requires Stop + Start to update the autocomplete list. `_on_play` re-validates against the current live checkbox state to catch stale choices.
- **Refer to** `docs/superpowers/specs/2026-08-01-discord-bot-design.md` for behavior details not repeated here.

---

## File structure

- **Modify** `scripts/launcher.py` — extract `build_netplay_argv` to module level, add `LogPanel.on_line` callback + `recent_lines` ring buffer, add `MatchRequest`, `DiscordBotThread`, `DiscordTab`. Register the new tab.
- **Modify** `scripts/netplay.py` — one added line to print the match-started marker.
- **Modify** `tests/test_launcher_command.py` — validator tests, shared `build_netplay_argv` tests, LogPanel ring buffer + callback tests.
- **Modify** `requirements.txt` — add `discord.py>=2.3`.
- **Modify** `.gitignore` — add `.slippi_ai_launcher.json` (defense in depth; file lives in `%USERPROFILE%` but a future refactor might move it).

---

### Task 1: Extract `build_netplay_argv` helper

**Files:**
- Modify: `scripts/launcher.py` (the `NetplayTab.build_argv` static method)
- Modify: `tests/test_launcher_command.py` (add tests calling the helper directly)

**Interfaces:**
- Consumes: nothing new.
- Produces: module-level function `build_netplay_argv(global_paths: dict, tab_values: dict) -> list[str]` with the exact same behavior as the current `NetplayTab.build_argv`. `NetplayTab.build_argv` becomes a thin static wrapper that calls the helper.

- [ ] **Step 1: Add a failing test that calls the helper directly**

Append to `NetplayTest` in `tests/test_launcher_command.py`:

```python
  def test_build_netplay_argv_helper_matches_tab(self):
    tv = {
        'model_path': r'C:\M', 'char': 'fox', 'costume': '',
        'connect_code': 'ABCD#123', 'runtime': '',
        'user_json_path': r'C:\U\user.json',
    }
    gp = {'dolphin_path': 'D', 'iso_path': 'I'}
    self.assertEqual(
        launcher.build_netplay_argv(gp, tv),
        launcher.NetplayTab.build_argv(gp, tv),
    )
```

- [ ] **Step 2: Run tests, verify failure**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command.NetplayTest -v`
Expected: FAIL — `AttributeError: module 'launcher' has no attribute 'build_netplay_argv'`.

- [ ] **Step 3: Extract the helper in `scripts/launcher.py`**

Add a module-level function just above `class NetplayTab`:

```python
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
```

Then replace `NetplayTab.build_argv`'s body with:

```python
  @staticmethod
  def build_argv(global_paths: dict, tab_values: dict) -> list[str]:
    return build_netplay_argv(global_paths, tab_values)
```

- [ ] **Step 4: Run all tests to verify no regression**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command -v`
Expected: OK — 18 tests (17 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/launcher.py tests/test_launcher_command.py
git commit -m "[launcher] Extract build_netplay_argv as module-level helper."
```

---

### Task 2: Netplay marker + LogPanel callback + recent-lines ring buffer

**Files:**
- Modify: `scripts/netplay.py` (one added print line)
- Modify: `scripts/launcher.py` (extend `LogPanel`)
- Modify: `tests/test_launcher_command.py` (add `LogPanelTest`)

**Interfaces:**
- Produces on `LogPanel`:
  - `on_line: Callable[[str, str], None] | None` — public attribute, default `None`. If set, `append` calls it with `(line_text, kind)` after writing to the widget. Kind is one of `'stdout'`, `'stderr'`, `'meta'`.
  - `recent_lines() -> list[str]` — returns up to the last 20 raw lines (any kind), oldest first.
- Produces in `netplay.py`:
  - After the first successful `agent.step(gamestate)` call inside `main()` (right after `num_frames = 1`), a single line: `print('[NETPLAY_MATCH_STARTED]', flush=True)`.

- [ ] **Step 1: Write failing tests for LogPanel callback + ring buffer**

Insert this class in `tests/test_launcher_command.py` before `if __name__ == '__main__':`:

```python
class LogPanelTest(unittest.TestCase):

  def setUp(self):
    # Use a hidden root; if display is unavailable, skip.
    try:
      self.root = launcher.tk.Tk()
    except launcher.tk.TclError as e:
      self.skipTest(f'No display: {e}')
    self.root.withdraw()
    self.panel = launcher.LogPanel(self.root)

  def tearDown(self):
    self.root.destroy()

  def test_on_line_callback_fires_with_line_and_kind(self):
    seen = []
    self.panel.on_line = lambda line, kind: seen.append((line, kind))
    self.panel.append('hello\n', kind='stdout')
    self.panel.append('boom\n', kind='stderr')
    self.assertEqual(seen, [('hello\n', 'stdout'), ('boom\n', 'stderr')])

  def test_on_line_none_is_safe(self):
    self.panel.on_line = None
    self.panel.append('quiet\n')  # must not raise

  def test_recent_lines_holds_last_20(self):
    for i in range(25):
      self.panel.append(f'line {i}\n')
    lines = self.panel.recent_lines()
    self.assertEqual(len(lines), 20)
    self.assertEqual(lines[0], 'line 5\n')
    self.assertEqual(lines[-1], 'line 24\n')
```

- [ ] **Step 2: Run tests, verify failure**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command.LogPanelTest -v`
Expected: FAIL — attribute errors on `on_line` / `recent_lines` (or the test may skip if no display; if it skips, still counts as unimplemented).

- [ ] **Step 3: Extend `LogPanel` in `scripts/launcher.py`**

Modify `LogPanel` (below its current definition, replacing it wholesale):

```python
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
```

Add `import collections` at the top of the file if not already imported.

- [ ] **Step 4: Add the marker to `scripts/netplay.py`**

Change the block starting at "Main loop":

```python
    # Main loop
    agent.start()
    agent.step(gamestate)

    num_frames = 1
```

to:

```python
    # Main loop
    agent.start()
    agent.step(gamestate)

    num_frames = 1
    # Marker line consumed by the launcher's Discord bot to cancel its
    # "opponent never joined" timeout once the match is actually live.
    print('[NETPLAY_MATCH_STARTED]', flush=True)
```

- [ ] **Step 5: Run all tests to verify no regression + new tests pass**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command -v`
Expected: OK — 21 tests (18 existing + 3 new). If `LogPanelTest` skips due to no display, that's acceptable — this task should be run on a machine with a display.

- [ ] **Step 6: Commit**

```bash
git add scripts/launcher.py scripts/netplay.py tests/test_launcher_command.py
git commit -m "[launcher] Add LogPanel on_line + recent_lines; add netplay match marker."
```

---

### Task 3: Add discord.py dependency + gitignore

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`

**Interfaces:** none.

- [ ] **Step 1: Add discord.py to `requirements.txt`**

Append the following line (keep whatever line ending the file already uses):

```
discord.py>=2.3
```

Do not remove or reorder existing entries.

- [ ] **Step 2: Install into the venv**

Run: `venv\Scripts\python.exe -m pip install "discord.py>=2.3"`
Expected: success message.

- [ ] **Step 3: Verify the import works**

Run: `venv\Scripts\python.exe -c "import discord; import discord.app_commands; print(discord.__version__)"`
Expected: prints a version >= 2.3.0.

- [ ] **Step 4: Add config file to `.gitignore`**

Read the current `.gitignore`, then append (only if not already present):

```
# Launcher config (stores the Discord bot token in plaintext).
.slippi_ai_launcher.json
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "[deps] Add discord.py; ignore launcher config with bot token."
```

---

### Task 4: Validators

**Files:**
- Modify: `scripts/launcher.py` (add validators near other utility functions)
- Modify: `tests/test_launcher_command.py` (add `ValidatorTest` class)

**Interfaces:**
- Produces:
  - `validate_connect_code(code: str) -> bool` — True iff `code` matches `^[A-Z0-9]{2,6}#\d{1,6}$`.
  - `validate_supported_character(char: str, supported: list[str]) -> bool` — True iff `char` (case-sensitive; lowercase expected) is in `supported`.

- [ ] **Step 1: Write failing tests**

Insert this class in `tests/test_launcher_command.py` before `if __name__ == '__main__':`:

```python
class ValidatorTest(unittest.TestCase):

  def test_connect_code_valid(self):
    for code in ['ABCD#123', 'TNBN#217', 'A1#1', 'ABCDEF#999999']:
      with self.subTest(code=code):
        self.assertTrue(launcher.validate_connect_code(code))

  def test_connect_code_invalid(self):
    for code in ['', 'abcd#123', 'ABCD-123', 'ABCD#', '#123',
                 'A#1', 'ABCDEFG#1', 'ABCD#1234567', 'ABCD#12a']:
      with self.subTest(code=code):
        self.assertFalse(launcher.validate_connect_code(code))

  def test_character_supported(self):
    self.assertTrue(launcher.validate_supported_character('fox', ['fox', 'falco']))

  def test_character_unsupported(self):
    self.assertFalse(launcher.validate_supported_character('marth', ['fox', 'falco']))
    self.assertFalse(launcher.validate_supported_character('FOX', ['fox', 'falco']))  # case-sensitive
    self.assertFalse(launcher.validate_supported_character('', ['fox']))
```

- [ ] **Step 2: Run tests, verify failure**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command.ValidatorTest -v`
Expected: FAIL — `AttributeError: module 'launcher' has no attribute 'validate_connect_code'`.

- [ ] **Step 3: Add validators in `scripts/launcher.py`**

Add near the top of the file (after `CHARACTERS` / `PLAYER_TYPES` constants):

```python
import re

_CONNECT_CODE_RE = re.compile(r'^[A-Z0-9]{2,6}#\d{1,6}$')


def validate_connect_code(code: str) -> bool:
  return bool(_CONNECT_CODE_RE.match(code or ''))


def validate_supported_character(char: str, supported: list[str]) -> bool:
  return char in supported
```

- [ ] **Step 4: Run tests to verify passing**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command -v`
Expected: OK — 26 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/launcher.py tests/test_launcher_command.py
git commit -m "[launcher] Add validate_connect_code and validate_supported_character."
```

---

### Task 5: MatchRequest dataclass

**Files:**
- Modify: `scripts/launcher.py`
- Modify: `tests/test_launcher_command.py`

**Interfaces:**
- Produces:
  ```python
  @dataclasses.dataclass
  class MatchRequest:
    user_id: int
    user_name: str
    channel_id: int
    connect_code: str
    character: str
    started_at: float
  ```

- [ ] **Step 1: Write failing test**

Insert in `tests/test_launcher_command.py` before `if __name__ == '__main__':`:

```python
class MatchRequestTest(unittest.TestCase):

  def test_construction_and_fields(self):
    r = launcher.MatchRequest(
        user_id=42, user_name='sean', channel_id=100,
        connect_code='ABCD#123', character='fox', started_at=1.5,
    )
    self.assertEqual(r.user_id, 42)
    self.assertEqual(r.character, 'fox')
    self.assertEqual(r.started_at, 1.5)
```

- [ ] **Step 2: Run test, verify failure**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command.MatchRequestTest -v`
Expected: FAIL — `AttributeError: module 'launcher' has no attribute 'MatchRequest'`.

- [ ] **Step 3: Add MatchRequest in `scripts/launcher.py`**

Add near the other module-level types (after `Config`):

```python
import dataclasses


@dataclasses.dataclass
class MatchRequest:
  user_id: int
  user_name: str
  channel_id: int
  connect_code: str
  character: str
  started_at: float
```

- [ ] **Step 4: Run tests, verify passing**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command -v`
Expected: OK — 27 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/launcher.py tests/test_launcher_command.py
git commit -m "[launcher] Add MatchRequest dataclass for the Discord bot."
```

---

### Task 6: DiscordBotThread skeleton (connect / disconnect only)

**Files:**
- Modify: `scripts/launcher.py`

**Interfaces:**
- Produces:
  - `class DiscordBotThread`:
    - `__init__(self, app: 'LauncherApp', status_cb: 'Callable[[str], None]')` — `status_cb` will be invoked with human-readable status strings; the class is responsible for scheduling it onto the Tk main thread via `app.root.after(0, status_cb, text)`.
    - `start(self, token: str, allowed_channels: list[int], model_path: str, user_json_path: str, character_choices: list[str]) -> None` — spins the daemon thread. Idempotent-ish: raises `RuntimeError` if already running.
    - `stop(self) -> None` — signals the loop to close the client; safe from Tk main thread; returns immediately. Poll `is_running` to know when it's actually stopped.
    - `is_running: bool` (property).
  - The class stores its constructor args and `character_choices` for later use in Task 8; no `/play` handler is registered in this task.

- [ ] **Step 1: Add `DiscordBotThread` (skeleton) to `scripts/launcher.py`**

Add these imports at the top of the file (only those not already present):

```python
import asyncio
```

Add the class after `MatchRequest`:

```python
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
    fut = asyncio.run_coroutine_threadsafe(self._client.close(), self._loop)
    try:
      fut.result(timeout=5)
    except Exception:
      pass

  def _post_status(self, text: str) -> None:
    self._app.root.after(0, self._status_cb, text)

  def _run(self, token: str) -> None:
    import discord
    self._loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self._loop)
    intents = discord.Intents.default()
    self._client = discord.Client(intents=intents)

    @self._client.event
    async def on_ready():
      self._post_status(f'connected as {self._client.user}')

    @self._client.event
    async def on_disconnect():
      self._post_status('disconnected — reconnecting…')

    @self._client.event
    async def on_resumed():
      self._post_status(f'connected as {self._client.user}')

    try:
      self._loop.run_until_complete(self._client.start(token))
    except discord.LoginFailure:
      self._post_status('bad token')
    except Exception as e:
      self._post_status(f'error: {e}')
    finally:
      try:
        self._loop.run_until_complete(self._client.close())
      except Exception:
        pass
      self._loop.close()
      self._loop = None
      self._client = None
      self._post_status('stopped')
```

- [ ] **Step 2: Syntax check**

Run: `venv\Scripts\python.exe -c "import ast; ast.parse(open('scripts/launcher.py').read())"`
Expected: no output (success).

- [ ] **Step 3: Import-time check**

Run: `venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'scripts'); import launcher; print(launcher.DiscordBotThread)"`
Expected: prints the class object; no ImportError from missing discord/asyncio.

- [ ] **Step 4: Ensure no test regression**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command -v`
Expected: OK — 27 tests (unchanged).

- [ ] **Step 5: Commit**

```bash
git add scripts/launcher.py
git commit -m "[launcher] Add DiscordBotThread skeleton (connect/disconnect only)."
```

---

### Task 7: DiscordTab (UI + config + wire Start/Stop bot)

**Files:**
- Modify: `scripts/launcher.py`

**Interfaces:**
- Consumes:
  - `ScriptTab` (base class; DiscordTab does NOT inherit — see below)
  - `DiscordBotThread` from Task 6
  - `CHARACTERS`, `MODELS_DIR`, `Config`
- Produces:
  - `class DiscordTab(ttk.Frame)` — new tab; does NOT subclass `ScriptTab` because it has no `build_argv` / `Run` semantics. Interfaces the LauncherApp expects:
    - `TAB_KEY = 'discord'`
    - `LABEL = 'discord'`
    - Constructor `__init__(self, parent, app)` (matches other tabs)
  - Registered in `LauncherApp.__init__`'s tab-registration tuple.

**Fields:**
- Bot token (masked entry — `show='*'`)
- Allowed channel IDs (single entry, comma-separated integers)
- Model path (entry + Browse; separate from NetplayTab's field)
- Slippi user.json (entry + Browse)
- Supported characters (grid of checkboxes for all `CHARACTERS`; user ticks which the model plays)
- Connect timeout seconds (entry, default `600`)
- Start bot / Stop bot buttons
- Status label
- Match slot label

- [ ] **Step 1: Add `DiscordTab` to `scripts/launcher.py`**

Add after `NetplayTab`, before `LauncherApp`:

```python
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
```

- [ ] **Step 2: Register the tab in `LauncherApp.__init__`**

Change the tab-registration tuple to include `DiscordTab`:

```python
    for tab_cls in (EvalTwoTab, RunDolphinTab, RunEvaluatorTab, NetplayTab, DiscordTab):
      tab = tab_cls(self.notebook, self)
      self.notebook.add(tab, text=tab_cls.LABEL)
```

- [ ] **Step 3: Extend `_on_close` to stop the bot before destroying**

Find the existing `LauncherApp._on_close` method and inside its "quit path" (after the messagebox but before `runner.stop()`), add:

```python
    # Stop the Discord bot cleanly if any tab has one running.
    for tab in self._discord_tabs():
      if tab._bot is not None and tab._bot.is_running:
        tab._bot.stop()
```

And add this helper to `LauncherApp`:

```python
  def _discord_tabs(self):
    return [t for t in (self.notebook.nametowidget(name) for name in self.notebook.tabs())
            if isinstance(t, DiscordTab)]
```

- [ ] **Step 4: Syntax + import + no test regression**

Run in order:

```
venv\Scripts\python.exe -c "import ast; ast.parse(open('scripts/launcher.py').read())"
venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'scripts'); import launcher; print(launcher.DiscordTab)"
venv\Scripts\python.exe -m unittest tests.test_launcher_command -v
```

Expected: (1) no output; (2) prints the class; (3) OK — 27 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/launcher.py
git commit -m "[launcher] Add DiscordTab with token / channels / model / user.json / character checkboxes."
```

---

### Task 8: `/play` slash command (slot guard, timeout, spawn, status posts)

**Files:**
- Modify: `scripts/launcher.py` (add slash command inside `DiscordBotThread._run`)

**Interfaces:**
- Consumes: `build_netplay_argv` (Task 1), `MatchRequest` (Task 5), `validate_connect_code` / `validate_supported_character` (Task 4), `LogPanel.on_line` + `recent_lines()` (Task 2), the netplay marker (Task 2), `ProcessRunner` (existing).
- Produces: fully-functional `/play` in the bot; slot guard; per-match connect timeout; status/end/failure posts; live end-of-match feedback via the log-panel callback.

- [ ] **Step 1: Extend `DiscordBotThread` to register `/play` and wire the pipeline**

Replace the `_run` method and add the new helpers (adjust imports as needed):

```python
import time  # add near other imports if not already present

# In DiscordBotThread:

  def _run(self, token: str) -> None:
    import discord
    from discord import app_commands
    self._loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self._loop)
    intents = discord.Intents.default()
    self._client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(self._client)

    choices = [app_commands.Choice(name=c, value=c) for c in self._character_choices]

    @tree.command(name='play', description='Challenge the Slippi-AI bot.')
    @app_commands.describe(code='Your Slippi connect code, e.g. TNBN#217',
                           char='Character for the AI to play')
    @app_commands.choices(char=choices)
    async def play_cmd(interaction: 'discord.Interaction',
                       code: str,
                       char: app_commands.Choice[str]) -> None:
      await self._on_play(interaction, code, char.value)

    @self._client.event
    async def on_ready():
      # Sync commands to every guild the bot is in.
      for guild in self._client.guilds:
        try:
          await tree.sync(guild=guild)
        except Exception as e:
          self._post_status(f'sync failed on {guild}: {e}')
      self._post_status(f'connected as {self._client.user}')

    @self._client.event
    async def on_disconnect():
      self._post_status('disconnected — reconnecting…')

    @self._client.event
    async def on_resumed():
      self._post_status(f'connected as {self._client.user}')

    try:
      self._loop.run_until_complete(self._client.start(token))
    except discord.LoginFailure:
      self._post_status('bad token')
    except Exception as e:
      self._post_status(f'error: {e}')
    finally:
      try:
        self._loop.run_until_complete(self._client.close())
      except Exception:
        pass
      self._loop.close()
      self._loop = None
      self._client = None
      self._post_status('stopped')

  async def _on_play(self, interaction, code: str, char: str) -> None:
    # Channel allowlist — silent drop if not allowed.
    if interaction.channel_id not in self._allowed_channels:
      await interaction.response.send_message('Not authorized here.', ephemeral=True)
      return
    # Busy guard.
    if self._active is not None:
      await interaction.response.send_message(
          f'Bot busy — @{self._active.user_name} is playing vs `{self._active.connect_code}`.')
      return
    # Validate.
    if not validate_connect_code(code):
      await interaction.response.send_message(
          'Invalid code — expected `ABCD#123`.', ephemeral=True)
      return
    if not validate_supported_character(char, self._character_choices):
      await interaction.response.send_message(
          f'Model does not support `{char}`. Options: {", ".join(self._character_choices)}',
          ephemeral=True)
      return
    # Claim slot.
    request = MatchRequest(
        user_id=interaction.user.id,
        user_name=interaction.user.display_name,
        channel_id=interaction.channel_id,
        connect_code=code,
        character=char,
        started_at=time.monotonic(),
    )
    self._active = request
    self._app.root.after(0, self._set_slot_label,
                        f'match slot: running (@{request.user_name} vs {code})')
    await interaction.response.send_message(
        f'**@{interaction.user.display_name}** vs `{code}` as `{char}` — starting…')
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
    channel = interaction.channel

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
    # Clear slot + UI first so a busy reply after this is honest.
    self._active = None
    self._app.root.after(0, self._set_slot_label, 'match slot: idle')
    self._app.root.after(0, self._clear_log_watcher)
    if exit_code == 0:
      await channel.send(f'Match ended (exit code 0).')
    else:
      tail = ''.join(recent_lines[-10:])
      msg = f'Match failed (exit code {exit_code}).'
      if tail.strip():
        msg += f'\n```\n{tail[-1500:]}\n```'
      await channel.send(msg)

  async def _on_spawn_failed(self, channel, err: str) -> None:
    self._active = None
    self._app.root.after(0, self._set_slot_label, 'match slot: idle')
    self._app.root.after(0, self._clear_log_watcher)
    await channel.send(f'Failed to spawn netplay: {err}')

  def _set_slot_label(self, text: str) -> None:
    for tab in self._app._discord_tabs():
      tab.slot_label.configure(text=text)

  def _clear_log_watcher(self) -> None:
    if self._app.log.on_line is self._watch_stdout:
      self._app.log.on_line = None
```

- [ ] **Step 2: Syntax + import check**

Run:

```
venv\Scripts\python.exe -c "import ast; ast.parse(open('scripts/launcher.py').read())"
venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'scripts'); import launcher; print(launcher.DiscordBotThread)"
```

Expected: (1) no output; (2) prints the class.

- [ ] **Step 3: Run tests (must still pass unchanged)**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command -v`
Expected: OK — 27 tests. This task adds no unit tests (discord.py needs a real gateway; covered by manual smoke test).

- [ ] **Step 4: Commit**

```bash
git add scripts/launcher.py
git commit -m "[launcher] Implement /play slash command with slot guard, connect timeout, and status posts."
```

- [ ] **Step 5: End-to-end manual smoke test (user runs this — subagent notes as pending)**

Instructions for Sean (write these into the report):

1. Create a Discord application at https://discord.com/developers/applications; add a **Bot** user; copy its token; enable "Application.Commands" scope when generating the invite URL.
2. Invite the bot to a test server; copy the target channel's ID (Discord Developer Mode → right-click channel → Copy ID).
3. In the launcher, open the **discord** tab. Fill token, channel ID, model file, user.json, tick `fox` and `falco` in the character grid. Click **Start bot**. Status → "connected as <BotName#1234>".
4. In the allowed Discord channel, run `/play code:<your own Slippi code> char:fox`. Bot should reply "starting…". Netplay window should appear.
5. Complete or exit the match. Bot should reply "match ended (exit code 0)".
6. Try `/play` twice rapidly: second should be rejected with "Bot busy".
7. Try `/play` with a made-up code (`ZZZZ#999`); wait 10 minutes → "opponent never joined — timing out".
8. Click **Stop bot**. Status → "stopped". Close launcher — no dangling processes.

---

## Self-review notes

- **Spec coverage:**
  - Discord tab UI (spec §UI layout) → Task 7.
  - `DiscordTab` / `DiscordBotThread` / `MatchRequest` (spec §Components) → Tasks 5, 6, 7, 8.
  - Extracted `build_netplay_argv` (spec §Argv construction) → Task 1.
  - Netplay marker (spec §Marker) → Task 2.
  - Slash command details incl. snapshotted choices (spec §Slash command) → Task 8 (chars captured in `character_choices` at `DiscordBotThread.start` in Task 6; used in Task 8's `on_ready`).
  - Data flow (spec §Data flow) → Task 8.
  - Threading model (spec §Threading) → Task 6 (structure) + Task 8 (cross-thread hops).
  - Error handling table (spec §Error handling) → covered across Tasks 6–8.
  - Security (spec §Security) → Task 3 (.gitignore) + Task 8 (validators applied).
  - Testing (spec §Testing) → Task 1 (argv), Task 2 (LogPanel), Task 4 (validators), Task 5 (MatchRequest). Live Discord = manual only, per spec.
- **Placeholder scan:** no `TBD` / `TODO` / "similar to Task N" in the plan. Every code block is complete.
- **Type consistency:** `DiscordBotThread.start`'s kw arg `character_choices: list[str]` (Task 6) matches Task 8's use (`self._character_choices`). `on_line` callback signature `(line: str, kind: str)` matches Task 2 and Task 8's `_watch_stdout(line, kind)`. `MatchRequest` fields (Task 5) match Task 8's construction. `build_netplay_argv` signature stable across Tasks 1 and 8.
