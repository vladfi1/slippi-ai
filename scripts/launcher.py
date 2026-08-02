"""Tkinter launcher for slippi-ai scripts. See docs/superpowers/specs/2026-08-01-slippi-ai-launcher-gui-design.md."""

import json
import os
import pathlib
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

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


class LauncherApp:

  def __init__(self):
    self.root = tk.Tk()
    self.root.title('Slippi-AI Launcher')
    self.root.geometry('900x700')
    self.config = Config.load()
    self.global_paths = GlobalPathsFrame(self.root, initial=self.config.global_)
    self.global_paths.pack(fill='x', padx=8, pady=(8, 4))
    self.notebook = ttk.Notebook(self.root)
    self.notebook.pack(fill='both', expand=True, padx=8, pady=8)
    for tab_cls in (EvalTwoTab,):
      tab = tab_cls(self.notebook, self)
      self.notebook.add(tab, text=tab_cls.LABEL)
    self.log = LogPanel(self.root)
    self.log.pack(fill='both', expand=False, padx=8, pady=(0, 8))
    self.runner = ProcessRunner(self.root, self.log)
    self.root.protocol('WM_DELETE_WINDOW', self._on_close)

  def _on_close(self):
    if self.runner.is_running:
      from tkinter import messagebox
      if not messagebox.askyesno('Script running', 'A script is still running. Stop and quit?'):
        return
      self.runner.stop()
    self.config.global_ = self.global_paths.values()
    self.config.save()
    self.root.destroy()

  def run(self):
    self.root.mainloop()


if __name__ == '__main__':
  LauncherApp().run()
