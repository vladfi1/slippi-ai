"""Unit tests for scripts/launcher.py."""

import json
import pathlib
import sys
import tempfile
import unittest

# Allow importing `scripts/launcher.py` as a module.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

import launcher  # noqa: E402


class ConfigTest(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.tmp_path = pathlib.Path(self._tmp.name)
    self._orig_path = launcher.Config.path
    launcher.Config.path = staticmethod(lambda: self.tmp_path / 'cfg.json')

  def tearDown(self):
    launcher.Config.path = self._orig_path
    self._tmp.cleanup()

  def test_load_missing_file_returns_defaults(self):
    cfg = launcher.Config.load()
    self.assertEqual(cfg.global_, {})
    self.assertEqual(cfg.tabs, {})
    self.assertEqual(cfg.last_tab, '')

  def test_load_malformed_returns_defaults(self):
    (self.tmp_path / 'cfg.json').write_text('not json {{{')
    cfg = launcher.Config.load()
    self.assertEqual(cfg.global_, {})

  def test_save_then_load_roundtrips(self):
    cfg = launcher.Config()
    cfg.global_ = {'dolphin_path': 'X', 'iso_path': 'Y'}
    cfg.tabs = {'eval_two': {'p1_type': 'human'}}
    cfg.last_tab = 'eval_two'
    cfg.save()
    loaded = launcher.Config.load()
    self.assertEqual(loaded.global_, cfg.global_)
    self.assertEqual(loaded.tabs, cfg.tabs)
    self.assertEqual(loaded.last_tab, 'eval_two')
    data = json.loads((self.tmp_path / 'cfg.json').read_text())
    self.assertEqual(data['global'], cfg.global_)


if __name__ == '__main__':
  unittest.main()
