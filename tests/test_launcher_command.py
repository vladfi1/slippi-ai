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


_DEFAULT_ADV = {'sample_temperature': '1.0', 'async_inference': True, 'name': '', 'mirror': False}


class EvalTwoBuildArgvTest(unittest.TestCase):

  def test_human_vs_ai(self):
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
    self.assertEqual(argv[0], sys.executable)
    self.assertEqual(argv[1], str(launcher.REPO_ROOT / 'scripts' / 'eval_two.py'))
    rest = argv[2:]
    self.assertIn('--p1.type=human', rest)
    self.assertIn('--p2.type=ai', rest)
    self.assertIn('--p2.character=FALCO', rest)
    self.assertIn(r'--p2.ai.path=C:\models\medium-v2', rest)
    self.assertIn(r'--dolphin.path=C:\Dolphin\Slippi Dolphin.exe', rest)
    self.assertIn(r'--dolphin.iso=C:\ISO\SSBM.iso', rest)
    self.assertIn('--p2.ai.sample_temperature=1.0', rest)
    self.assertIn('--p2.ai.async_inference=true', rest)
    self.assertIn('--p2.ai.mirror=false', rest)
    self.assertFalse(any(a.startswith('--p1.ai.') for a in rest))
    self.assertFalse(any(a.startswith('--p2.ai.name') for a in rest))
    self.assertFalse(any(a.startswith('--num_games') for a in rest))

  def test_includes_num_games_when_set(self):
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
    self.assertIn('--num_games=3', rest)
    self.assertIn('--p2.level=7', rest)

  def test_advanced_overrides(self):
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
    self.assertIn('--p2.ai.sample_temperature=0.8', rest)
    self.assertIn('--p2.ai.async_inference=false', rest)
    self.assertIn('--p2.ai.name=FalcoBot', rest)
    self.assertIn('--p2.ai.mirror=true', rest)


class EvalTwoValidateTest(unittest.TestCase):

  def test_flags_missing_dolphin(self):
    errors = launcher.EvalTwoTab.validate(
        global_paths={'dolphin_path': '', 'iso_path': ''},
        tab_values={
            'p1': {'type': 'human', 'character': 'FOX', 'model_path': '', 'cpu_level': '9'},
            'p2': {'type': 'ai', 'character': 'FALCO', 'model_path': '', 'cpu_level': '9'},
            'num_games': '',
            'advanced': _DEFAULT_ADV,
        },
    )
    self.assertTrue(any('Dolphin' in e for e in errors))
    self.assertTrue(any('ISO' in e for e in errors))
    self.assertTrue(any('model' in e.lower() for e in errors))


class RunDolphinTest(unittest.TestCase):

  def test_build_argv_defaults(self):
    argv = launcher.RunDolphinTab.build_argv(
        global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
        tab_values={'N': '1', 'frames': '3600', 'render': False},
    )
    rest = argv[2:]
    self.assertEqual(argv[1], str(launcher.REPO_ROOT / 'scripts' / 'run_dolphin.py'))
    self.assertIn('--N=1', rest)
    self.assertIn('--frames=3600', rest)
    self.assertIn('--render=false', rest)
    self.assertIn('--dolphin.path=D', rest)
    self.assertIn('--dolphin.iso=I', rest)

  def test_validate_requires_paths(self):
    errors = launcher.RunDolphinTab.validate(
        global_paths={'dolphin_path': '', 'iso_path': ''},
        tab_values={'N': '1', 'frames': '3600', 'render': False},
    )
    self.assertTrue(any('Dolphin' in e for e in errors))
    self.assertTrue(any('ISO' in e for e in errors))


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


class RunEvaluatorTest(unittest.TestCase):

  def test_build_argv_no_self_play(self):
    argv = launcher.RunEvaluatorTab.build_argv(
        global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
        tab_values=_eval_defaults(),
    )
    rest = argv[2:]
    self.assertEqual(argv[1], str(launcher.REPO_ROOT / 'scripts' / 'run_evaluator.py'))
    self.assertIn('--self_play=false', rest)
    self.assertIn(r'--player.ai.path=C:\M1', rest)
    self.assertIn('--player.character=FOX', rest)
    self.assertIn(r'--opponent.ai.path=C:\M2', rest)
    self.assertIn('--opponent.character=FALCO', rest)
    self.assertIn('--num_envs=4', rest)
    self.assertIn('--rollout_length=3600', rest)
    self.assertFalse(any(a.startswith('--num_games') for a in rest))
    self.assertIn('--use_gpu=true', rest)

  def test_build_argv_self_play_omits_opponent(self):
    tv = _eval_defaults()
    tv['self_play'] = True
    tv['num_games'] = '10'
    argv = launcher.RunEvaluatorTab.build_argv(
        global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
        tab_values=tv,
    )
    rest = argv[2:]
    self.assertIn('--self_play=true', rest)
    self.assertFalse(any(a.startswith('--opponent.') for a in rest))
    self.assertIn('--num_games=10', rest)

  def test_validate_requires_player_model(self):
    tv = _eval_defaults()
    tv['player']['model_path'] = ''
    errors = launcher.RunEvaluatorTab.validate(
        global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
        tab_values=tv,
    )
    self.assertTrue(any('Player' in e and 'model' in e.lower() for e in errors))


if __name__ == '__main__':
  unittest.main()
