# Bot Retry + Progress Updates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Discord bot so pre-match failures (wrong menu, opponent no-show, hung menu-helper) retry automatically up to N times, and post concise progress updates in Discord — including retry announcements and match duration on success.

**Architecture:** Introduce a small state machine on `DiscordBotThread` that wraps each netplay spawn in an attempt loop. Any pre-match outcome (timeout or process exit without the `[NETPLAY_MATCH_STARTED]` marker) triggers a retry until `max_attempts` is exhausted. Once the marker fires, subsequent exits are treated as normal match endings. A new `_end_attempt` method is the single decision point; `_handle_play` becomes thin.

**Tech Stack:** Python 3 stdlib + `discord.py` (already installed) + `tkinter` (already used). No new dependencies.

## Global Constraints

- **Two-space indentation** (project convention from `.claude/CLAUDE.md`).
- **Zero new Python dependencies.**
- **Tests use `unittest.TestCase`** — added to the existing `tests/test_launcher_command.py`. Run with `venv\Scripts\python.exe -m unittest tests.test_launcher_command -v`.
- **Cross-thread rules from the parent spec** (`2026-08-01-discord-bot-design.md`) still hold: bot loop → Tk via `self._app.root.after(0, ...)`; Tk → bot loop via `client.loop.call_soon_threadsafe(...)` or `asyncio.run_coroutine_threadsafe(...)`.
- **Marker string:** exactly `[NETPLAY_MATCH_STARTED]` (unchanged from parent spec).
- **Default `max_attempts`:** 2. Range: 1–5. Value of 1 disables retries.
- **Inter-retry pause:** `await asyncio.sleep(2)`.
- **Refer to** `docs/superpowers/specs/2026-08-02-bot-retry-and-progress-design.md` for behavior and message-catalog details not repeated here.

---

## File structure

- **Modify** `scripts/launcher.py`:
  - `MatchRequest` (dataclass) — add 3 new fields.
  - Module-level `_format_duration(seconds: float) -> str` — new helper.
  - Module-level `parse_max_attempts(text: str) -> int | None` — new helper.
  - `DiscordBotThread.start(...)` — extend signature with `max_attempts: int`.
  - `DiscordBotThread._start_attempt(...)` — new.
  - `DiscordBotThread._end_attempt(...)` — new.
  - `DiscordBotThread._handle_play(...)` — refactored to delegate spawn+timeout to `_start_attempt`.
  - `DiscordBotThread._on_match_started(...)` — sets `_active.match_started = True`; sets `_active.started_at`; existing behavior otherwise.
  - `DiscordBotThread._connect_timeout(...)` — no longer posts directly; delegates to `_end_attempt`.
  - `DiscordBotThread._on_match_ended(...)` — deleted (folded into `_end_attempt`).
  - `DiscordTab` — add `max_attempts_var`, spinbox, config persistence, validation.
- **Modify** `tests/test_launcher_command.py`:
  - Update `MatchRequestTest.test_construction_and_fields` to cover new default fields.
  - Add `FormatDurationTest`.
  - Add `MaxAttemptsParseTest`.

No new files. No new dependencies.

---

### Task 1: Extend `MatchRequest` dataclass

**Files:**
- Modify: `scripts/launcher.py` (the existing `MatchRequest` class near the top of the file)
- Modify: `tests/test_launcher_command.py` (update `MatchRequestTest.test_construction_and_fields`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `MatchRequest` grows three optional fields:
  - `attempt: int = 1`
  - `max_attempts: int = 2`
  - `match_started: bool = False`
  All existing call sites (currently only `_handle_play`) work unchanged because the new fields have defaults.

- [ ] **Step 1: Update the existing `MatchRequestTest.test_construction_and_fields` test**

Open `tests/test_launcher_command.py`, find `MatchRequestTest`, and REPLACE the test method body with:

```python
  def test_construction_and_fields(self):
    r = launcher.MatchRequest(
        user_id=42, user_name='sean', channel_id=100,
        connect_code='ABCD#123', character='fox', started_at=1.5,
    )
    self.assertEqual(r.user_id, 42)
    self.assertEqual(r.character, 'fox')
    self.assertEqual(r.started_at, 1.5)
    # New defaults.
    self.assertEqual(r.attempt, 1)
    self.assertEqual(r.max_attempts, 2)
    self.assertFalse(r.match_started)

  def test_construction_with_new_fields(self):
    r = launcher.MatchRequest(
        user_id=1, user_name='x', channel_id=1,
        connect_code='X#1', character='fox', started_at=0.0,
        attempt=2, max_attempts=3, match_started=True,
    )
    self.assertEqual(r.attempt, 2)
    self.assertEqual(r.max_attempts, 3)
    self.assertTrue(r.match_started)
```

- [ ] **Step 2: Run tests to verify the new default assertions fail**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command.MatchRequestTest -v`
Expected: FAIL — `AttributeError: 'MatchRequest' object has no attribute 'attempt'` (or similar).

- [ ] **Step 3: Extend `MatchRequest` in `scripts/launcher.py`**

Find the existing dataclass:

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

Replace with:

```python
@dataclasses.dataclass
class MatchRequest:
  user_id: int
  user_name: str
  channel_id: int
  connect_code: str
  character: str
  started_at: float
  attempt: int = 1
  max_attempts: int = 2
  match_started: bool = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command -v`
Expected: OK — 31 tests (30 previous + 1 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/launcher.py tests/test_launcher_command.py
git commit -m "[bot-retry] Extend MatchRequest with attempt/max_attempts/match_started."
```

---

### Task 2: Add `_format_duration` helper

**Files:**
- Modify: `scripts/launcher.py` (add module-level helper near other utilities)
- Modify: `tests/test_launcher_command.py` (add `FormatDurationTest`)

**Interfaces:**
- Consumes: nothing.
- Produces: `_format_duration(seconds: float) -> str` — returns `"{m}m {s}s"` where `m` and `s` are integer minutes and remainder seconds. Negative or zero → `"0m 0s"`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_launcher_command.py` before `if __name__ == '__main__':`:

```python
class FormatDurationTest(unittest.TestCase):

  def test_typical_match(self):
    self.assertEqual(launcher._format_duration(4 * 60 + 12), '4m 12s')

  def test_under_a_minute(self):
    self.assertEqual(launcher._format_duration(45), '0m 45s')

  def test_over_an_hour(self):
    self.assertEqual(launcher._format_duration(65 * 60), '65m 0s')

  def test_fractional_seconds_floor(self):
    self.assertEqual(launcher._format_duration(90.9), '1m 30s')

  def test_zero_and_negative(self):
    self.assertEqual(launcher._format_duration(0), '0m 0s')
    self.assertEqual(launcher._format_duration(-5), '0m 0s')
```

- [ ] **Step 2: Run tests, verify failure**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command.FormatDurationTest -v`
Expected: FAIL — `AttributeError: module 'launcher' has no attribute '_format_duration'`.

- [ ] **Step 3: Add the helper to `scripts/launcher.py`**

Add near the other module-level utility functions (near `validate_connect_code` / `scrub_for_public`):

```python
def _format_duration(seconds: float) -> str:
  """Format an elapsed-seconds value as '{m}m {s}s'. Clamps at 0."""
  if seconds <= 0:
    return '0m 0s'
  total = int(seconds)
  return f'{total // 60}m {total % 60}s'
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command -v`
Expected: OK — 36 tests (31 previous + 5 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/launcher.py tests/test_launcher_command.py
git commit -m "[bot-retry] Add _format_duration helper."
```

---

### Task 3: Add `parse_max_attempts` helper

**Files:**
- Modify: `scripts/launcher.py`
- Modify: `tests/test_launcher_command.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `parse_max_attempts(text: str) -> int | None` — returns an int in `[1, 5]` when `text` parses as an integer in that range; returns `None` otherwise. Blank strings return `1` (retries disabled).

- [ ] **Step 1: Write failing tests**

Append to `tests/test_launcher_command.py` before `if __name__ == '__main__':`:

```python
class ParseMaxAttemptsTest(unittest.TestCase):

  def test_valid_values(self):
    for text, expected in [('1', 1), ('2', 2), ('3', 3), ('5', 5)]:
      with self.subTest(text=text):
        self.assertEqual(launcher.parse_max_attempts(text), expected)

  def test_blank_returns_one(self):
    self.assertEqual(launcher.parse_max_attempts(''), 1)
    self.assertEqual(launcher.parse_max_attempts('   '), 1)

  def test_out_of_range(self):
    self.assertIsNone(launcher.parse_max_attempts('0'))
    self.assertIsNone(launcher.parse_max_attempts('6'))
    self.assertIsNone(launcher.parse_max_attempts('-1'))

  def test_non_integer(self):
    self.assertIsNone(launcher.parse_max_attempts('two'))
    self.assertIsNone(launcher.parse_max_attempts('2.5'))
    self.assertIsNone(launcher.parse_max_attempts('1a'))
```

- [ ] **Step 2: Run tests, verify failure**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command.ParseMaxAttemptsTest -v`
Expected: FAIL — `AttributeError: module 'launcher' has no attribute 'parse_max_attempts'`.

- [ ] **Step 3: Add the helper to `scripts/launcher.py`**

Add near `_format_duration` / other utility helpers:

```python
def parse_max_attempts(text: str) -> 'int | None':
  """Parse the max-attempts field. Blank -> 1 (retries disabled).
  Non-integer or out-of-range -> None."""
  s = text.strip()
  if not s:
    return 1
  try:
    n = int(s)
  except ValueError:
    return None
  if 1 <= n <= 5:
    return n
  return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m unittest tests.test_launcher_command -v`
Expected: OK — 40 tests (36 previous + 4 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/launcher.py tests/test_launcher_command.py
git commit -m "[bot-retry] Add parse_max_attempts helper."
```

---

### Task 4: Extend `DiscordBotThread.start` with `max_attempts`

**Files:**
- Modify: `scripts/launcher.py` (the `DiscordBotThread.__init__` and `.start(...)` method)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - New instance attribute `self._max_attempts: int` (defaults to 2 in `__init__`).
  - `DiscordBotThread.start(...)` gains a new positional-or-keyword parameter `max_attempts: int = 2` at the end. Callers must eventually pass it (Task 6 wires the UI).

- [ ] **Step 1: Update `__init__` to add the attribute**

Find `DiscordBotThread.__init__` and add `self._max_attempts: int = 2` next to the other instance attributes (near `self._character_choices`).

- [ ] **Step 2: Update `start()` signature**

Find the existing:

```python
  def start(self, token: str, allowed_channels: list[int], model_path: str,
            user_json_path: str, character_choices: list[str]) -> None:
```

Change to:

```python
  def start(self, token: str, allowed_channels: list[int], model_path: str,
            user_json_path: str, character_choices: list[str],
            max_attempts: int = 2) -> None:
```

And inside the method body, alongside the other `self._X = X` assignments, add:

```python
    self._max_attempts = max_attempts
```

- [ ] **Step 3: Syntax + no test regression**

Run:

```
venv\Scripts\python.exe -c "import ast; ast.parse(open('scripts/launcher.py').read())"
venv\Scripts\python.exe -m unittest tests.test_launcher_command -v
```

Expected: (1) no output, (2) OK — 40 tests.

- [ ] **Step 4: Commit**

```bash
git add scripts/launcher.py
git commit -m "[bot-retry] Thread max_attempts through DiscordBotThread.start()."
```

---

### Task 5: Refactor to `_start_attempt` + `_end_attempt` (the retry state machine)

**Files:**
- Modify: `scripts/launcher.py` (`DiscordBotThread._handle_play`, `_on_match_started`, `_connect_timeout`, `_on_match_ended`, plus new `_start_attempt` and `_end_attempt` methods)

**Interfaces:**
- Consumes: `MatchRequest` (with new fields, from Task 1), `_format_duration` (from Task 2), `self._max_attempts` (from Task 4), existing `build_netplay_argv`, `ProcessRunner`, `LogPanel`, `scrub_for_public`.
- Produces:
  - New `_start_attempt(request, channel, first: bool) -> None` (async method). Contains the spawn logic (log-watcher install, `runner.start` via `root.after`, timeout task creation). If `first=False`, first posts the retry message. Reads timeout from `self._app.config.tabs['discord']['connect_timeout_s']` (same as current code).
  - New `_end_attempt(request, channel, reason: str, exit_code: 'int | None' = None, recent_lines: 'list[str] | None' = None) -> None` (async method). Central state machine — see spec §Components for the branch structure.
  - `_handle_play` — trimmed: still does validation, slot claim, "starting…" post, then calls `_start_attempt(request, channel, first=True)`. No longer builds argv or handles exit/timeout directly.
  - `_on_match_started` — sets `self._active.match_started = True`, sets `self._active.started_at = time.monotonic()`, cancels timeout task, posts `**match live**`.
  - `_connect_timeout` — no longer posts anything or kills directly. On the sleep completing (not cancelled), calls `_end_attempt(request, channel, reason='timeout')` via `run_coroutine_threadsafe` — wait, `_connect_timeout` already runs on the bot loop, so it can `await self._end_attempt(...)` directly.
  - `_on_match_ended` — DELETED (all its logic folded into `_end_attempt` with `reason='exit'`).

- [ ] **Step 1: Add `_start_attempt` method to `DiscordBotThread`**

Add this new method inside the class (place near the other match-lifecycle methods):

```python
  async def _start_attempt(self, request: 'MatchRequest', channel, first: bool) -> None:
    """Spawn one netplay attempt. If `first` is False, first posts a retry
    message. Installs the marker watcher, spawns netplay, and starts the
    connect timeout."""
    if not first:
      await channel.send(
          f'Opponent didn\'t join, retrying ({request.attempt}/{request.max_attempts})…')

    # Build argv (same as before).
    tab_values = {
        'model_path': self._model_path,
        'char': request.character,
        'costume': '',
        'connect_code': request.connect_code,
        'runtime': '',
        'user_json_path': self._user_json_path,
    }
    global_paths = self._app.global_paths.values()
    argv = build_netplay_argv(global_paths, tab_values)

    def on_exit_tk(exit_code: int) -> None:
      # Runs on Tk main thread; bounce to bot loop.
      recent = self._app.log.recent_lines()
      if self._loop is not None:
        asyncio.run_coroutine_threadsafe(
            self._end_attempt(request, channel, reason='exit',
                              exit_code=exit_code, recent_lines=recent),
            self._loop)

    def start_runner():
      # Install marker watcher before spawn.
      self._app.log.on_line = self._watch_stdout
      try:
        self._app.runner.start(argv, cwd=REPO_ROOT, on_exit=on_exit_tk)
      except RuntimeError as e:
        if self._loop is not None:
          asyncio.run_coroutine_threadsafe(
              self._on_spawn_failed(channel, str(e)), self._loop)
    self._app.root.after(0, start_runner)

    # Connect timeout for this attempt.
    try:
      timeout_s = int(self._app.config.tabs.get('discord', {}).get('connect_timeout_s', '600'))
    except ValueError:
      timeout_s = 600
    self._timeout_task = self._loop.create_task(self._connect_timeout(request, channel, timeout_s))
```

- [ ] **Step 2: Replace `_connect_timeout` body**

Find the existing `_connect_timeout` and replace ENTIRELY with:

```python
  async def _connect_timeout(self, request: 'MatchRequest', channel, timeout_s: int) -> None:
    try:
      await asyncio.sleep(timeout_s)
    except asyncio.CancelledError:
      return
    await self._end_attempt(request, channel, reason='timeout')
```

- [ ] **Step 3: Add `_end_attempt` method**

Add near `_start_attempt`:

```python
  async def _end_attempt(self, request: 'MatchRequest', channel, reason: str,
                         exit_code: 'int | None' = None,
                         recent_lines: 'list[str] | None' = None) -> None:
    """Central decision point for what to do when an attempt finishes
    (either from timeout or from process exit). `reason` is 'timeout' or
    'exit'."""
    # Idempotence: if the slot has already been cleared, another handler
    # already dealt with this. E.g. timeout fires, kills the process, then
    # the process's real exit fires this again.
    if self._active is not request:
      return

    # Cancel any lingering timeout task.
    if self._timeout_task is not None and not self._timeout_task.done():
      self._timeout_task.cancel()
    self._timeout_task = None

    # On timeout, we need to kill the still-running netplay process.
    # (On exit reason, the process already died.)
    if reason == 'timeout':
      self._app.root.after(0, self._app.runner.stop)

    if request.match_started:
      # Real match end (or mid-match crash). No retry.
      self._active = None
      self._app.root.after(0, self._set_slot_label, 'match slot: idle')
      self._app.root.after(0, self._clear_log_watcher)
      await self._set_ready_presence()
      if reason == 'exit' and exit_code == 0:
        duration = _format_duration(time.monotonic() - request.started_at)
        await channel.send(f'Match ended after {duration}.')
      else:
        tail = scrub_for_public(''.join((recent_lines or [])[-10:]))
        msg = f'Match failed (exit code {exit_code}).'
        if tail.strip():
          msg += f'\n```\n{tail[-1500:]}\n```'
        await channel.send(msg)
      return

    # Pre-match failure (timeout, or process exit without marker).
    if request.attempt < request.max_attempts:
      request.attempt += 1
      await asyncio.sleep(2)
      # Re-check the slot in case the user cancelled or the bot stopped
      # during the pause.
      if self._active is not request:
        return
      await self._start_attempt(request, channel, first=False)
      return

    # All attempts exhausted.
    self._active = None
    self._app.root.after(0, self._set_slot_label, 'match slot: idle')
    self._app.root.after(0, self._clear_log_watcher)
    await self._set_ready_presence()
    await channel.send(
        f'Opponent didn\'t join after {request.max_attempts} attempts.')
```

- [ ] **Step 4: Delete the old `_on_match_ended` method**

Find and DELETE the entire `async def _on_match_ended(...)` method (roughly 15 lines including the exit-code branching and stderr tail). Its logic is now inside `_end_attempt`.

- [ ] **Step 5: Update `_on_match_started` to flip the flag and set started_at**

Find the existing method:

```python
  async def _on_match_started(self) -> None:
    if self._timeout_task is not None and not self._timeout_task.done():
      self._timeout_task.cancel()
      self._timeout_task = None
    if self._active is None:
      return
    channel = self._client.get_channel(self._active.channel_id)
    if channel is not None:
      await channel.send('**match live**')
```

Replace with:

```python
  async def _on_match_started(self) -> None:
    if self._timeout_task is not None and not self._timeout_task.done():
      self._timeout_task.cancel()
      self._timeout_task = None
    if self._active is None:
      return
    self._active.match_started = True
    self._active.started_at = time.monotonic()
    channel = self._client.get_channel(self._active.channel_id)
    if channel is not None:
      await channel.send('**match live**')
```

- [ ] **Step 6: Refactor `_handle_play` to delegate to `_start_attempt`**

Find the existing `_handle_play` and REPLACE its entire body with:

```python
  async def _handle_play(self, channel, user, code: str, char: str) -> None:
    # Channel allowlist — silent drop.
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
        max_attempts=self._max_attempts,
    )
    self._active = request
    self._app.root.after(0, self._set_slot_label,
                        f'match slot: running (@{request.user_name} vs {code})')
    await self._set_busy_presence(user.display_name, char)
    bot_code = self._bot_connect_code()
    await channel.send(
        f'**@{user.display_name}** vs `{code}` as `{char}` — starting…\n'
        f'Enter my code in your Slippi game (Direct Connect): **`{bot_code}`**')
    await self._start_attempt(request, channel, first=True)
```

- [ ] **Step 7: Syntax + no test regression**

Run:

```
venv\Scripts\python.exe -c "import ast; ast.parse(open('scripts/launcher.py').read())"
venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'scripts'); import launcher; print(launcher.DiscordBotThread)"
venv\Scripts\python.exe -m unittest tests.test_launcher_command -v
```

Expected: (1) no output, (2) prints the class, (3) OK — 40 tests.

- [ ] **Step 8: Commit**

```bash
git add scripts/launcher.py
git commit -m "[bot-retry] Route spawn/exit/timeout through _start_attempt + _end_attempt."
```

---

### Task 6: Add Max-attempts field to `DiscordTab` UI

**Files:**
- Modify: `scripts/launcher.py` (`DiscordTab._build_widgets`, `._values`, `._validate`, `._on_start`)

**Interfaces:**
- Consumes: `parse_max_attempts` (Task 3), `DiscordBotThread.start(..., max_attempts=...)` (Task 4).
- Produces: a persisted config key `'max_attempts'` under `config.tabs['discord']`.

- [ ] **Step 1: Add the widget in `_build_widgets`**

Inside `DiscordTab._build_widgets`, find the block that creates `self.timeout_var`:

```python
    self.timeout_var = tk.StringVar(value=initial.get('connect_timeout_s', '600'))
```

Add right after it:

```python
    self.max_attempts_var = tk.StringVar(value=initial.get('max_attempts', '2'))
```

Then find the `row('Connect timeout (s):', ...)` line and add right after it:

```python
    row('Max attempts per request:',
        ttk.Spinbox(grid, from_=1, to=5, textvariable=self.max_attempts_var, width=6), 5)
```

(The next row number after `Connect timeout` is currently 5 for the character grid — bump that grid's row index by 1 if using explicit indices, or if it's on a separate frame `chars = ttk.LabelFrame(...)` that packs independently, no change needed. Check the existing code and preserve packing order.)

- [ ] **Step 2: Update `_values()` to include the new field**

Find `_values()` in `DiscordTab` and add `'max_attempts': self.max_attempts_var.get()` to the returned dict.

- [ ] **Step 3: Update `_validate()` to check the new field**

Find `_validate()` in `DiscordTab`. Add this block near the existing timeout validation:

```python
    if launcher.parse_max_attempts(values.get('max_attempts', '')) is None:
      errors.append('Max attempts must be an integer between 1 and 5.')
```

Wait — `_validate` is inside `DiscordTab` which itself is inside the `launcher` module. Don't prefix with `launcher.`. Use the bare name:

```python
    if parse_max_attempts(values.get('max_attempts', '')) is None:
      errors.append('Max attempts must be an integer between 1 and 5.')
```

- [ ] **Step 4: Pass to `DiscordBotThread.start` in `_on_start`**

Find `_on_start` in `DiscordTab`. It currently calls:

```python
    self._bot.start(
        token=values['token'],
        allowed_channels=self._parse_channel_ids(values['allowed_channels']) or [],
        model_path=values['model_path'],
        user_json_path=values['user_json_path'],
        character_choices=values['supported_characters'],
    )
```

Change to:

```python
    self._bot.start(
        token=values['token'],
        allowed_channels=self._parse_channel_ids(values['allowed_channels']) or [],
        model_path=values['model_path'],
        user_json_path=values['user_json_path'],
        character_choices=values['supported_characters'],
        max_attempts=parse_max_attempts(values['max_attempts']) or 2,
    )
```

- [ ] **Step 5: Syntax + import + tests**

Run:

```
venv\Scripts\python.exe -c "import ast; ast.parse(open('scripts/launcher.py').read())"
venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'scripts'); import launcher; print(launcher.DiscordTab)"
venv\Scripts\python.exe -m unittest tests.test_launcher_command -v
```

Expected: (1) no output, (2) prints the class, (3) OK — 40 tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/launcher.py
git commit -m "[bot-retry] Add Max-attempts field to DiscordTab; wire to start()."
```

- [ ] **Step 7: Manual smoke test — instructions for the user, not the subagent**

The subagent cannot run the GUI or connect to real Discord. Include this in the report as a note for the user to do manually:

1. Restart the launcher, open the Discord tab, verify the new "Max attempts per request" spinbox appears next to Connect timeout, default 2.
2. Start the bot. From Discord, mention it with a valid character but a *made-up* connect code (e.g. `@bot ZZZZ#999 fox`). Wait through one timeout — should see the retry message. Wait through the second — should see "Opponent didn't join after 2 attempts."
3. Trigger with a real code, complete a real match, verify `Match ended after Xm Ys.` appears (not `exit code 0`).

---

## Self-review notes

- **Spec coverage:**
  - Extended `MatchRequest` fields → Task 1.
  - `_format_duration` helper → Task 2.
  - `parse_max_attempts` helper → Task 3.
  - `DiscordBotThread.start(..., max_attempts=...)` signature → Task 4.
  - `_start_attempt` / `_end_attempt` state machine + retry pause + idempotence + timeout-kill routing → Task 5.
  - `_handle_play` refactor → Task 5.
  - `_on_match_started` sets `match_started` + resets `started_at` → Task 5.
  - `_connect_timeout` delegates to `_end_attempt` → Task 5.
  - `_on_match_ended` deletion (folded into `_end_attempt`) → Task 5.
  - Discord message catalog (retry, final failure, match-ended-with-duration) → Task 5.
  - DiscordTab spinbox + config persistence + validation → Task 6.
  - Manual smoke test → Task 6.
- **Placeholder scan:** no `TBD` / `TODO` / "similar to Task N" / "add appropriate error handling" anywhere. Every code block is complete and directly usable.
- **Type consistency:**
  - `MatchRequest.attempt`, `max_attempts`, `match_started` — declared in Task 1, used in Task 5 (`request.attempt`, `request.max_attempts`, `request.match_started`).
  - `_format_duration(seconds: float) -> str` — declared in Task 2, called with `time.monotonic() - request.started_at` in Task 5.
  - `parse_max_attempts(text: str) -> int | None` — declared in Task 3, called in Task 6.
  - `DiscordBotThread.start(..., max_attempts: int = 2)` — extended in Task 4, called with keyword arg in Task 6.
  - `_start_attempt(request, channel, first: bool)` — declared in Task 5, called by `_handle_play` (Task 5) and recursively from `_end_attempt` (Task 5).
  - `_end_attempt(request, channel, reason: str, exit_code=None, recent_lines=None)` — declared in Task 5, called from `_connect_timeout` and `on_exit_tk` (Task 5).
