# Slippi-AI Launcher GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native Tkinter launcher (`scripts/launcher.py`) plus a one-click `launcher.bat` so the user can run `eval_two.py`, `run_dolphin.py`, `run_evaluator.py`, and `netplay.py` from a GUI on Windows.

**Architecture:** Single-file Tk app organized around a `ttk.Notebook` with one tab per script. Each tab subclasses a common `ScriptTab` base that owns its fields, a pure `build_argv()` method (unit-tested), and shares one `LogPanel` and `ProcessRunner` at the bottom of the window. Config auto-persists to JSON in the user profile.

**Tech Stack:** Python 3 stdlib only — `tkinter`, `tkinter.ttk`, `subprocess`, `queue`, `threading`, `json`, `pathlib`, `sys`. No new dependencies.

## Global Constraints

- **Two-space indentation** (project convention from `.claude/CLAUDE.md`).
- **Zero new Python dependencies** — stdlib only.
- **Windows-first** — the `.bat` and `CTRL_BREAK_EVENT`-based Stop are Windows-specific.
- **Config path:** `%USERPROFILE%\.slippi_ai_launcher.json`.
- **Command formatting:** always `--key=value` (single argv element). Nested fancyflags use dots (`--p1.ai.path=...`, `--dolphin.iso=...`). Booleans use `--foo=true` / `--foo=false`. Blank optional fields → omit the flag.
- **Python interpreter for subprocesses:** `sys.executable` (matches the launcher's own interpreter regardless of activation state).
- **Character enum values** must match `melee.Character` names exactly (hardcoded list in `launcher.py`).
- Refer to the spec at `docs/superpowers/specs/2026-08-01-slippi-ai-launcher-gui-design.md` for behavior details not repeated here.

---

## File structure

- **Create** `scripts/launcher.py` — single-file Tk app. Sections in order: imports, constants (`CHARACTERS`, `PLAYER_TYPES`, `REPO_ROOT`), `Config`, `LogPanel`, `ProcessRunner`, `AdvancedSection`, `GlobalPathsFrame`, `PlayerFrame`, `ScriptTab` (abstract), the four tab subclasses, `LauncherApp` (owns root + notebook), and `if __name__ == "__main__": LauncherApp().run()`.
- **Create** `launcher.bat` at repo root — activates venv and runs the launcher.
- **Create** `tests/test_launcher_command.py` — unit tests for each tab's `build_argv()`.

---

### Task 1: Scaffolding + launcher.bat + smoke-testable main window

**Files:**
- Create: `scripts/launcher.py`
- Create: `launcher.bat` (repo root)

**Interfaces:**
- Consumes: nothing.
- Produces: `LauncherApp` class with a `run()` method that opens a Tk window containing a `ttk.Notebook` (empty for now) and a title. Later tasks add tabs.

- [ ] **Step 1: Create `launcher.bat` in repo root**

```bat
@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python scripts\launcher.py
```

- [ ] **Step 2: Create `scripts/launcher.py` with the minimum viable app**

```python
"""Tkinter launcher for slippi-ai scripts. See docs/superpowers/specs/2026-08-01-slippi-ai-launcher-gui-design.md."""

import pathlib
import sys
import tkinter as tk
from tkinter import ttk

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class LauncherApp:

  def __init__(self):
    self.root = tk.Tk()
    self.root.title('Slippi-AI Launcher')
    self.root.geometry('900x700')
    self.notebook = ttk.Notebook(self.root)
    self.notebook.pack(fill='both', expand=True, padx=8, pady=8)

  def run(self):
    self.root.mainloop()


if __name__ == '__main__':
  LauncherApp().run()
```

- [ ] **Step 3: Smoke-test manually**

Run: double-click `launcher.bat` (or `venv\Scripts\python.exe scripts\launcher.py` from an activated shell).
Expected: an empty 900x700 window titled "Slippi-AI Launcher" opens. Closing it exits cleanly.

- [ ] **Step 4: Commit**

```bash
git add scripts/launcher.py launcher.bat
git commit -m "[launcher] Add empty Tkinter shell and launcher.bat."
```

---

### Task 2: Config load/save

**Files:**
- Modify: `scripts/launcher.py` (add `Config` class near the top, below constants)
- Create: `tests/test_launcher_command.py` (first test lives here)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Config.path() -> pathlib.Path` — returns `%USERPROFILE%\.slippi_ai_launcher.json`.
  - `Config.load() -> Config` — reads JSON or returns a fresh default. Never raises on malformed/missing.
  - `Config.save(self) -> None` — writes JSON atomically-ish (`Path.write_text`).
  - `Config` has attributes: `global_: dict[str, str]`, `tabs: dict[str, dict]`, `last_tab: str`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_launcher_command.py`:

```python
"""Unit tests for scripts/launcher.py."""

import json
import pathlib
import sys

# Allow importing `scripts/launcher.py` as a module.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

import launcher  # noqa: E402


def test_config_load_missing_file_returns_defaults(tmp_path, monkeypatch):
  monkeypatch.setattr(launcher.Config, 'path', staticmethod(lambda: tmp_path / 'missing.json'))
  cfg = launcher.Config.load()
  assert cfg.global_ == {}
  assert cfg.tabs == {}
  assert cfg.last_tab == ''


def test_config_load_malformed_returns_defaults(tmp_path, monkeypatch):
  p = tmp_path / 'cfg.json'
  p.write_text('not json {{{')
  monkeypatch.setattr(launcher.Config, 'path', staticmethod(lambda: p))
  cfg = launcher.Config.load()
  assert cfg.global_ == {}


def test_config_save_then_load_roundtrips(tmp_path, monkeypatch):
  p = tmp_path / 'cfg.json'
  monkeypatch.setattr(launcher.Config, 'path', staticmethod(lambda: p))
  cfg = launcher.Config()
  cfg.global_ = {'dolphin_path': 'X', 'iso_path': 'Y'}
  cfg.tabs = {'eval_two': {'p1_type': 'human'}}
  cfg.last_tab = 'eval_two'
  cfg.save()
  loaded = launcher.Config.load()
  assert loaded.global_ == cfg.global_
  assert loaded.tabs == cfg.tabs
  assert loaded.last_tab == 'eval_two'
  # Sanity-check on-disk shape.
  data = json.loads(p.read_text())
  assert data['global'] == cfg.global_
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_launcher_command.py -v`
Expected: FAIL — `AttributeError: module 'launcher' has no attribute 'Config'` (or import error).

- [ ] **Step 3: Add `Config` to `scripts/launcher.py`**

Add after the `REPO_ROOT = ...` line:

```python
import json
import os


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_launcher_command.py -v`
Expected: 3 passed.

- [ ] **Step 5: Wire Config into LauncherApp**

Modify `LauncherApp.__init__` to add:

```python
    self.config = Config.load()
    self.root.protocol('WM_DELETE_WINDOW', self._on_close)

  def _on_close(self):
    self.config.save()
    self.root.destroy()
```

- [ ] **Step 6: Commit**

```bash
git add scripts/launcher.py tests/test_launcher_command.py
git commit -m "[launcher] Add Config with JSON persistence and tests."
```

---

### Task 3: LogPanel + ProcessRunner

**Files:**
- Modify: `scripts/launcher.py`

**Interfaces:**
- Consumes: `LauncherApp.root` (the Tk root, for `after()` scheduling).
- Produces:
  - `LogPanel(parent)` — packable frame with `append(text: str, kind: str = 'stdout')`. `kind` in `{'stdout', 'stderr', 'meta'}`. Auto-scrolls to bottom.
  - `ProcessRunner(root: tk.Tk, log: LogPanel)`:
    - `start(argv: list[str], cwd: pathlib.Path, on_exit: Callable[[int], None]) -> None`
    - `stop() -> None` — sends `CTRL_BREAK_EVENT`, then `terminate()` after 3s, then `kill()` after 2s more.
    - `is_running: bool` property.
    - Raises `RuntimeError` if `start()` called while already running.

- [ ] **Step 1: Add `LogPanel` class to `scripts/launcher.py`**

```python
class LogPanel(ttk.Frame):

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

  def append(self, text: str, kind: str = 'stdout') -> None:
    self.text.configure(state='normal')
    self.text.insert('end', text, kind if kind != 'stdout' else ())
    self.text.see('end')
    self.text.configure(state='disabled')
```

- [ ] **Step 2: Add `ProcessRunner` class to `scripts/launcher.py`**

```python
import queue
import subprocess
import threading


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
    import time
    assert self._proc is not None
    try:
      if sys.platform == 'win32':
        self._proc.send_signal(signal.CTRL_BREAK_EVENT)
      else:
        self._proc.terminate()
    except OSError:
      pass
    deadline = time.monotonic() + self._STOP_GRACE_S
    while time.monotonic() < deadline and self._proc.poll() is None:
      time.sleep(0.1)
    if self._proc.poll() is None:
      self._proc.terminate()
      deadline = time.monotonic() + self._TERMINATE_GRACE_S
      while time.monotonic() < deadline and self._proc.poll() is None:
        time.sleep(0.1)
    if self._proc.poll() is None:
      self._proc.kill()
```

Also add `import signal` at the top of the file.

- [ ] **Step 2.5: Add `AdvancedSection` collapsible container**

Add below `ProcessRunner`:

```python
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
```

- [ ] **Step 3: Add LogPanel + ProcessRunner to LauncherApp**

Modify `LauncherApp.__init__`:

```python
    # After notebook.pack:
    self.log = LogPanel(self.root)
    self.log.pack(fill='both', expand=False, padx=8, pady=(0, 8))
    self.runner = ProcessRunner(self.root, self.log)
```

Modify `_on_close`:

```python
  def _on_close(self):
    if self.runner.is_running:
      from tkinter import messagebox
      if not messagebox.askyesno('Script running', 'A script is still running. Stop and quit?'):
        return
      self.runner.stop()
    self.config.save()
    self.root.destroy()
```

- [ ] **Step 4: Manual smoke test**

Run: `venv\Scripts\python.exe scripts\launcher.py`
Expected: window opens with an empty notebook on top and a dark log area on the bottom. Close it — no errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/launcher.py
git commit -m "[launcher] Add LogPanel, ProcessRunner, and AdvancedSection."
```

---

### Task 4: GlobalPathsFrame + PlayerFrame

**Files:**
- Modify: `scripts/launcher.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `CHARACTERS: list[str]` — hardcoded module-level constant.
  - `PLAYER_TYPES = ['ai', 'human', 'cpu']`.
  - `GlobalPathsFrame(parent, initial: dict)`:
    - `.values() -> dict[str, str]` returning `{'dolphin_path': ..., 'iso_path': ...}`.
  - `PlayerFrame(parent, label: str, initial: dict)`:
    - `.values() -> dict[str, str]` returning `{'type', 'character', 'model_path', 'cpu_level'}`.
    - Reactive show/hide of character/model/cpu_level based on type.

- [ ] **Step 1: Add constants near the top of `launcher.py` (after `REPO_ROOT`)**

```python
# Hardcoded to avoid importing melee at startup (slow first-open).
# Source: melee.Character enum in libmelee.
CHARACTERS = [
    'FOX', 'FALCO', 'MARTH', 'SHEIK', 'JIGGLYPUFF', 'CAPTAIN_FALCON',
    'PEACH', 'ICE_CLIMBERS', 'PIKACHU', 'SAMUS', 'DR_MARIO', 'YOSHI',
    'LUIGI', 'GANONDORF', 'MARIO', 'YOUNG_LINK', 'LINK', 'DONKEY_KONG',
    'GAME_AND_WATCH', 'MEWTWO', 'ROY', 'PICHU', 'NESS', 'BOWSER',
    'KIRBY', 'ZELDA',
]
PLAYER_TYPES = ['ai', 'human', 'cpu']
MODELS_DIR = REPO_ROOT / 'models'


def list_models() -> list[str]:
  if not MODELS_DIR.is_dir():
    return []
  return sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir())
```

- [ ] **Step 2: Add `GlobalPathsFrame`**

```python
from tkinter import filedialog


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
```

- [ ] **Step 3: Add `PlayerFrame`**

```python
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
    p = filedialog.askdirectory(title='Select model directory', initialdir=str(MODELS_DIR) if MODELS_DIR.is_dir() else None)
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
```

- [ ] **Step 4: Wire `GlobalPathsFrame` into `LauncherApp`**

Modify `LauncherApp.__init__`:

```python
    # Insert BEFORE the notebook.pack call:
    self.global_paths = GlobalPathsFrame(self.root, initial=self.config.global_)
    self.global_paths.pack(fill='x', padx=8, pady=(8, 4))
```

Modify `_on_close` to snapshot global paths before saving:

```python
    self.config.global_ = self.global_paths.values()
```

- [ ] **Step 5: Manual smoke test**

Run: `venv\Scripts\python.exe scripts\launcher.py`
Expected: window shows Global Paths frame with entries + Browse buttons. Log at bottom. Close, reopen — paths persist.

- [ ] **Step 6: Commit**

```bash
git add scripts/launcher.py
git commit -m "[launcher] Add GlobalPathsFrame and reactive PlayerFrame."
```

---

### Task 5: EvalTwoTab with unit-tested build_argv

**Files:**
- Modify: `scripts/launcher.py`
- Modify: `tests/test_launcher_command.py`

**Interfaces:**
- Consumes: `PlayerFrame.values()`, `GlobalPathsFrame.values()`, `Config`, `ProcessRunner`, `LogPanel`.
- Produces:
  - `ScriptTab` abstract base with `build_argv(global_paths: dict, tab_values: dict) -> list[str]` and `validate(global_paths, tab_values) -> list[str]`.
  - `EvalTwoTab(parent, app: LauncherApp)` — concrete tab; `build_argv` and `validate` are `@staticmethod`s callable from tests without instantiating Tk.

- [ ] **Step 1: Write failing test for eval_two build_argv**

Add to `tests/test_launcher_command.py`:

```python
_DEFAULT_ADV = {'sample_temperature': '1.0', 'async_inference': True, 'name': '', 'mirror': False}


def test_eval_two_build_argv_human_vs_ai():
  argv = launcher.EvalTwoTab.build_argv(
      global_paths={
          'dolphin_path': r'C:\Dolphin\Slippi Dolphin.exe',
          'iso_path': r'C:\ISO\SSBM.iso',
      },
      tab_values={
          'p1': {'type': 'human', 'character': 'FOX', 'model_path': '', 'cpu_level': '9'},
          'p2': {'type': 'ai', 'character': 'FALCO', 'model_path': r'C:\models\medium-v2', 'cpu_level': '9'},
          'num_games': '',
          'advanced': _DEFAULT_ADV,
      },
  )
  assert argv[0] == sys.executable
  assert argv[1] == str(launcher.REPO_ROOT / 'scripts' / 'eval_two.py')
  rest = argv[2:]
  assert '--p1.type=human' in rest
  assert '--p2.type=ai' in rest
  assert '--p2.character=FALCO' in rest
  assert r'--p2.ai.path=C:\models\medium-v2' in rest
  assert r'--dolphin.path=C:\Dolphin\Slippi Dolphin.exe' in rest
  assert r'--dolphin.iso=C:\ISO\SSBM.iso' in rest
  # Advanced flags apply only to the AI player.
  assert '--p2.ai.sample_temperature=1.0' in rest
  assert '--p2.ai.async_inference=true' in rest
  assert '--p2.ai.mirror=false' in rest
  assert not any(a.startswith('--p1.ai.') for a in rest)  # p1 is human
  assert not any(a.startswith('--p2.ai.name') for a in rest)  # blank name omitted
  # No num_games flag when blank.
  assert not any(a.startswith('--num_games') for a in rest)


def test_eval_two_build_argv_includes_num_games_when_set():
  argv = launcher.EvalTwoTab.build_argv(
      global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
      tab_values={
          'p1': {'type': 'ai', 'character': 'FOX', 'model_path': 'M1', 'cpu_level': '9'},
          'p2': {'type': 'cpu', 'character': 'FALCO', 'model_path': '', 'cpu_level': '7'},
          'num_games': '3',
          'advanced': _DEFAULT_ADV,
      },
  )
  rest = argv[2:]
  assert '--num_games=3' in rest
  assert '--p2.level=7' in rest


def test_eval_two_build_argv_advanced_overrides():
  argv = launcher.EvalTwoTab.build_argv(
      global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
      tab_values={
          'p1': {'type': 'human', 'character': 'FOX', 'model_path': '', 'cpu_level': '9'},
          'p2': {'type': 'ai', 'character': 'FALCO', 'model_path': 'M', 'cpu_level': '9'},
          'num_games': '',
          'advanced': {'sample_temperature': '0.8', 'async_inference': False, 'name': 'FalcoBot', 'mirror': True},
      },
  )
  rest = argv[2:]
  assert '--p2.ai.sample_temperature=0.8' in rest
  assert '--p2.ai.async_inference=false' in rest
  assert '--p2.ai.name=FalcoBot' in rest
  assert '--p2.ai.mirror=true' in rest


def test_eval_two_validate_flags_missing_dolphin():
  errors = launcher.EvalTwoTab.validate(
      global_paths={'dolphin_path': '', 'iso_path': ''},
      tab_values={
          'p1': {'type': 'human', 'character': 'FOX', 'model_path': '', 'cpu_level': '9'},
          'p2': {'type': 'ai', 'character': 'FALCO', 'model_path': '', 'cpu_level': '9'},
          'num_games': '',
          'advanced': _DEFAULT_ADV,
      },
  )
  assert any('Dolphin' in e for e in errors)
  assert any('ISO' in e for e in errors)
  assert any('model' in e.lower() for e in errors)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_launcher_command.py -v`
Expected: FAIL — `AttributeError: module 'launcher' has no attribute 'EvalTwoTab'`.

- [ ] **Step 3: Add `ScriptTab` base + `EvalTwoTab` to `scripts/launcher.py`**

```python
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
    for name, p in (('Player 1', tab_values['p1']), ('Player 2', tab_values['p2'])):
      if p['type'] == 'ai':
        if not p.get('model_path'):
          errors.append(f'{name}: model path is required when type=ai.')
        elif not pathlib.Path(p['model_path']).exists():
          errors.append(f'{name}: model path does not exist: {p["model_path"]}')
    return errors
```

- [ ] **Step 4: Register the tab in `LauncherApp.__init__`**

```python
    # After creating self.notebook but before self.log:
    for tab_cls in (EvalTwoTab,):
      tab = tab_cls(self.notebook, self)
      self.notebook.add(tab, text=tab_cls.LABEL)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_launcher_command.py -v`
Expected: 7 passed.

- [ ] **Step 6: Manual smoke test**

Run: `venv\Scripts\python.exe scripts\launcher.py`
Set Dolphin path, ISO path, P1=human, P2=ai with `models/medium-v2` and character FALCO, click Run.
Expected: log shows `Running: ...python.exe .../eval_two.py --p1.type=human --p2.type=ai --p2.character=FALCO --p2.ai.path=... --dolphin.path=... --dolphin.iso=...`, and the eval_two subprocess starts (Dolphin window appears).

- [ ] **Step 7: Commit**

```bash
git add scripts/launcher.py tests/test_launcher_command.py
git commit -m "[launcher] Add EvalTwoTab with build_argv unit tests."
```

---

### Task 6: RunDolphinTab

**Files:**
- Modify: `scripts/launcher.py`
- Modify: `tests/test_launcher_command.py`

**Interfaces:**
- Produces: `RunDolphinTab` — fields for `N` (int), `frames` (int), `render` (bool). Uses only `--dolphin.*` from globals; no player configuration (the script hardcodes `AI` + `CPU`).

- [ ] **Step 1: Write failing test**

Add to `tests/test_launcher_command.py`:

```python
def test_run_dolphin_build_argv_defaults():
  argv = launcher.RunDolphinTab.build_argv(
      global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
      tab_values={'N': '1', 'frames': '3600', 'render': False},
  )
  rest = argv[2:]
  assert argv[1] == str(launcher.REPO_ROOT / 'scripts' / 'run_dolphin.py')
  assert '--N=1' in rest
  assert '--frames=3600' in rest
  assert '--render=false' in rest
  assert '--dolphin.path=D' in rest
  assert '--dolphin.iso=I' in rest


def test_run_dolphin_validate_requires_paths():
  errors = launcher.RunDolphinTab.validate(
      global_paths={'dolphin_path': '', 'iso_path': ''},
      tab_values={'N': '1', 'frames': '3600', 'render': False},
  )
  assert any('Dolphin' in e for e in errors)
  assert any('ISO' in e for e in errors)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_launcher_command.py::test_run_dolphin_build_argv_defaults -v`
Expected: FAIL.

- [ ] **Step 3: Add `RunDolphinTab` to `scripts/launcher.py`**

```python
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
```

- [ ] **Step 4: Register tab**

Update the `for tab_cls in (...)` tuple in `LauncherApp.__init__` to include `RunDolphinTab`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_launcher_command.py -v`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/launcher.py tests/test_launcher_command.py
git commit -m "[launcher] Add RunDolphinTab."
```

---

### Task 7: RunEvaluatorTab

**Files:**
- Modify: `scripts/launcher.py`
- Modify: `tests/test_launcher_command.py`

**Interfaces:**
- Produces: `RunEvaluatorTab` — basics: `player.ai.path`, `player.character`, `self_play` (bool), `opponent.ai.path`, `opponent.character` (shown when `self_play=false`), `num_envs`, `rollout_length`, `num_games`. Advanced (all optional bools/ints): `use_gpu`, `async_envs`, `sim_envs`, `fake_envs`, `swap_ports`, `quiet`, `burnin`, `num_env_steps`, `inner_batch_size`, `num_agent_steps`.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_launcher_command.py`:

```python
def _eval_defaults():
  return {
      'player': {'model_path': r'C:\M1', 'character': 'FOX'},
      'opponent': {'model_path': r'C:\M2', 'character': 'FALCO'},
      'self_play': False,
      'num_envs': '4',
      'rollout_length': '3600',
      'num_games': '',
      # Advanced (defaults):
      'use_gpu': True,
      'async_envs': False,
      'sim_envs': False,
      'fake_envs': False,
      'swap_ports': True,
      'quiet': False,
      'burnin': False,
      'num_env_steps': '0',
      'inner_batch_size': '1',
      'num_agent_steps': '0',
  }


def test_run_evaluator_build_argv_no_self_play():
  argv = launcher.RunEvaluatorTab.build_argv(
      global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
      tab_values=_eval_defaults(),
  )
  rest = argv[2:]
  assert argv[1] == str(launcher.REPO_ROOT / 'scripts' / 'run_evaluator.py')
  assert '--self_play=false' in rest
  assert r'--player.ai.path=C:\M1' in rest
  assert '--player.character=FOX' in rest
  assert r'--opponent.ai.path=C:\M2' in rest
  assert '--opponent.character=FALCO' in rest
  assert '--num_envs=4' in rest
  assert '--rollout_length=3600' in rest
  assert not any(a.startswith('--num_games') for a in rest)
  assert '--use_gpu=true' in rest


def test_run_evaluator_build_argv_self_play_omits_opponent():
  tv = _eval_defaults()
  tv['self_play'] = True
  tv['num_games'] = '10'
  argv = launcher.RunEvaluatorTab.build_argv(
      global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
      tab_values=tv,
  )
  rest = argv[2:]
  assert '--self_play=true' in rest
  assert not any(a.startswith('--opponent.') for a in rest)
  assert '--num_games=10' in rest


def test_run_evaluator_validate_requires_player_model():
  tv = _eval_defaults()
  tv['player']['model_path'] = ''
  errors = launcher.RunEvaluatorTab.validate(
      global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
      tab_values=tv,
  )
  assert any('Player' in e and 'model' in e.lower() for e in errors)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_launcher_command.py -v`
Expected: 3 new failures.

- [ ] **Step 3: Add `RunEvaluatorTab` to `scripts/launcher.py`**

```python
class RunEvaluatorTab(ScriptTab):

  TAB_KEY = 'run_evaluator'
  SCRIPT = 'scripts/run_evaluator.py'
  LABEL = 'run_evaluator'

  # (widget code condensed; follows the same pattern as EvalTwoTab)

  def _build_widgets(self):
    initial = self.app.config.tabs.get(self.TAB_KEY, {})
    # Player + opponent — reuse a small helper: model+character only, no type toggle.
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
```

- [ ] **Step 4: Register tab**

Add `RunEvaluatorTab` to the tab tuple in `LauncherApp.__init__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_launcher_command.py -v`
Expected: 12 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/launcher.py tests/test_launcher_command.py
git commit -m "[launcher] Add RunEvaluatorTab with self-play toggle and advanced flags."
```

---

### Task 8: NetplayTab

**Files:**
- Modify: `scripts/launcher.py`
- Modify: `tests/test_launcher_command.py`

**Interfaces:**
- Produces: `NetplayTab` — fields: `agent.path`, `char`, `costume` (optional int), `dolphin.connect_code` (required), `runtime` (optional seconds).

- [ ] **Step 1: Write failing test**

Add to `tests/test_launcher_command.py`:

```python
def test_netplay_build_argv_minimum():
  argv = launcher.NetplayTab.build_argv(
      global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
      tab_values={
          'model_path': r'C:\M', 'char': 'FOX', 'costume': '',
          'connect_code': 'ABCD#123', 'runtime': '',
      },
  )
  rest = argv[2:]
  assert argv[1] == str(launcher.REPO_ROOT / 'scripts' / 'netplay.py')
  assert r'--agent.path=C:\M' in rest
  assert '--char=FOX' in rest
  assert '--dolphin.connect_code=ABCD#123' in rest
  assert '--dolphin.path=D' in rest
  assert not any(a.startswith('--costume') for a in rest)
  assert not any(a.startswith('--runtime') for a in rest)


def test_netplay_build_argv_with_optionals():
  argv = launcher.NetplayTab.build_argv(
      global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
      tab_values={
          'model_path': 'M', 'char': 'FALCO', 'costume': '2',
          'connect_code': 'X#1', 'runtime': '300',
      },
  )
  rest = argv[2:]
  assert '--costume=2' in rest
  assert '--runtime=300' in rest


def test_netplay_validate_requires_connect_code():
  errors = launcher.NetplayTab.validate(
      global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
      tab_values={'model_path': 'M', 'char': 'FOX', 'costume': '', 'connect_code': '', 'runtime': ''},
  )
  assert any('connect' in e.lower() for e in errors)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_launcher_command.py -v`
Expected: 3 new failures.

- [ ] **Step 3: Add `NetplayTab` to `scripts/launcher.py`**

```python
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
    grid.pack(fill='x')

  def _pick_model(self):
    p = filedialog.askdirectory(title='Select model directory', initialdir=str(MODELS_DIR) if MODELS_DIR.is_dir() else None)
    if p:
      self.model_var.set(p)

  def _values(self) -> dict:
    return {
        'model_path': self.model_var.get(),
        'char': self.char_var.get(),
        'costume': self.costume_var.get().strip(),
        'connect_code': self.connect_var.get().strip(),
        'runtime': self.runtime_var.get().strip(),
    }

  @staticmethod
  def build_argv(global_paths: dict, tab_values: dict) -> list[str]:
    argv = [sys.executable, str(REPO_ROOT / 'scripts' / 'netplay.py')]
    argv.append(f'--agent.path={tab_values.get("model_path", "")}')
    argv.append(f'--char={tab_values.get("char", "FOX")}')
    if tab_values.get('costume'):
      argv.append(f'--costume={tab_values["costume"]}')
    argv.append(f'--dolphin.path={global_paths.get("dolphin_path", "")}')
    argv.append(f'--dolphin.iso={global_paths.get("iso_path", "")}')
    argv.append(f'--dolphin.connect_code={tab_values.get("connect_code", "")}')
    if tab_values.get('runtime'):
      argv.append(f'--runtime={tab_values["runtime"]}')
    return argv

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
    return errors
```

- [ ] **Step 4: Register tab**

Add `NetplayTab` to the tab tuple in `LauncherApp.__init__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_launcher_command.py -v`
Expected: 15 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/launcher.py tests/test_launcher_command.py
git commit -m "[launcher] Add NetplayTab."
```

---

### Task 9: Restore last-used tab + final manual smoke test

**Files:**
- Modify: `scripts/launcher.py`

**Interfaces:**
- Consumes: `Config.last_tab`, `LauncherApp.notebook`.
- Produces: on startup, the notebook selects the previously-active tab.

- [ ] **Step 1: Add tab-selection restore**

In `LauncherApp.__init__`, after the tab-registration loop:

```python
    if self.config.last_tab:
      for i, tab in enumerate(self.notebook.tabs()):
        if self.notebook.tab(tab, 'text') == self.config.last_tab:
          self.notebook.select(i)
          break
```

- [ ] **Step 2: Run full test suite**

Run: `venv\Scripts\python.exe -m pytest tests/test_launcher_command.py -v`
Expected: 15 passed.

- [ ] **Step 3: End-to-end manual smoke test**

Close any lingering launcher windows, then double-click `launcher.bat`.

Verify each of these:
1. Window opens with four tabs (`eval_two`, `run_dolphin`, `run_evaluator`, `netplay`).
2. On the `eval_two` tab, fill in Dolphin path and ISO path, set P1=human and P2=ai with `models/medium-v2`, character FALCO. Click Run.
3. Log panel shows the exact command being run, then Dolphin launches and the game starts.
4. Click Stop — the process ends cleanly, log shows an exit code, Run button re-enables.
5. Close the window (X). Reopen. Confirm Dolphin/ISO paths and last selection persisted, and `eval_two` is the active tab.
6. Verify the argv line in the log matches the user's original PowerShell command modulo `--key=value` vs `--key value` formatting.

- [ ] **Step 4: Commit**

```bash
git add scripts/launcher.py
git commit -m "[launcher] Restore last-selected tab on startup."
```

---

## Self-review notes

- All spec sections mapped: UI layout (Tasks 4, 5), components (Tasks 2–8), data flow (Task 5's `_on_run`), error handling (per-tab `validate`, ProcessRunner OSError handling in Task 3, `_on_close` prompt in Task 3), testing (per-tab unit tests + final smoke test), collapsible Advanced sections (`AdvancedSection` in Task 3, wired into EvalTwoTab in Task 5 and RunEvaluatorTab in Task 7).
- Character list matches spec (26 names).
- Config file location matches spec (`%USERPROFILE%\.slippi_ai_launcher.json`).
- Command formatting rule (`--key=value`) applied consistently in every `build_argv`.
- All types/methods referenced by later tasks are defined in earlier tasks (`ScriptTab`, `PlayerFrame`, `Config`, `ProcessRunner`, `LogPanel`, `AdvancedSection`, `_add_player_flags`).
- No placeholders: every code block is complete and runnable in context.
