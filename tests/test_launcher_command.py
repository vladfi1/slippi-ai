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
