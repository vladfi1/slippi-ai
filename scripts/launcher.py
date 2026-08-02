"""Tkinter launcher for slippi-ai scripts. See docs/superpowers/specs/2026-08-01-slippi-ai-launcher-gui-design.md."""

import json
import os
import pathlib
import sys
import tkinter as tk
from tkinter import ttk

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


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


class LauncherApp:

  def __init__(self):
    self.root = tk.Tk()
    self.root.title('Slippi-AI Launcher')
    self.root.geometry('900x700')
    self.notebook = ttk.Notebook(self.root)
    self.notebook.pack(fill='both', expand=True, padx=8, pady=8)
    self.config = Config.load()
    self.root.protocol('WM_DELETE_WINDOW', self._on_close)

  def _on_close(self):
    self.config.save()
    self.root.destroy()

  def run(self):
    self.root.mainloop()


if __name__ == '__main__':
  LauncherApp().run()
