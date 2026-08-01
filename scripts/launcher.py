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
