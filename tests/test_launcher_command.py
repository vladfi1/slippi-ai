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

  def test_flags_both_players_human(self):
    errors = launcher.EvalTwoTab.validate(
        global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
        tab_values={
            'p1': {'type': 'human', 'character': 'FOX', 'model_path': '', 'cpu_level': '9'},
            'p2': {'type': 'human', 'character': 'FALCO', 'model_path': '', 'cpu_level': '9'},
            'num_games': '',
            'advanced': _DEFAULT_ADV,
        },
    )
    self.assertTrue(any('both cannot be human' in e.lower() or 'both' in e.lower() for e in errors))


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


class NetplayTest(unittest.TestCase):

  def test_build_argv_minimum(self):
    argv = launcher.NetplayTab.build_argv(
        global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
        tab_values={
            'model_path': r'C:\M', 'char': 'FOX', 'costume': '',
            'connect_code': 'ABCD#123', 'runtime': '', 'user_json_path': '',
        },
    )
    rest = argv[2:]
    self.assertEqual(argv[1], str(launcher.REPO_ROOT / 'scripts' / 'netplay.py'))
    self.assertIn(r'--agent.path=C:\M', rest)
    self.assertIn('--char=fox', rest)
    self.assertIn('--dolphin.connect_code=ABCD#123', rest)
    self.assertIn('--dolphin.path=D', rest)
    self.assertFalse(any(a.startswith('--costume') for a in rest))
    self.assertFalse(any(a.startswith('--runtime') for a in rest))
    self.assertFalse(any(a.startswith('--dolphin.user_json_path') for a in rest))

  def test_build_argv_with_optionals(self):
    argv = launcher.NetplayTab.build_argv(
        global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
        tab_values={
            'model_path': 'M', 'char': 'FALCO', 'costume': '2',
            'connect_code': 'X#1', 'runtime': '300',
            'user_json_path': r'C:\Slippi\user.json',
        },
    )
    rest = argv[2:]
    self.assertIn('--costume=2', rest)
    self.assertIn('--runtime=300', rest)
    self.assertIn(r'--dolphin.user_json_path=C:\Slippi\user.json', rest)

  def test_validate_requires_connect_code(self):
    errors = launcher.NetplayTab.validate(
        global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
        tab_values={'model_path': 'M', 'char': 'FOX', 'costume': '', 'connect_code': '', 'runtime': '', 'user_json_path': ''},
    )
    self.assertTrue(any('connect' in e.lower() for e in errors))

  def test_validate_requires_user_json(self):
    errors = launcher.NetplayTab.validate(
        global_paths={'dolphin_path': 'D', 'iso_path': 'I'},
        tab_values={'model_path': 'M', 'char': 'FOX', 'costume': '', 'connect_code': 'X#1', 'runtime': '', 'user_json_path': ''},
    )
    self.assertTrue(any('user.json' in e.lower() for e in errors))

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


class MatchRequestTest(unittest.TestCase):

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


class ScrubForPublicTest(unittest.TestCase):

  def test_drops_python_traceback_frames(self):
    raw = (
        'Traceback (most recent call last):\n'
        '  File "C:\\Users\\alice\\proj\\scripts\\netplay.py", line 106, in <module>\n'
        '    app.run(main)\n'
        '  File "C:\\Users\\alice\\proj\\venv\\Lib\\site-packages\\absl\\app.py", line 410, in run\n'
        '    _run_main(main, args)\n'
        'melee.slippstream.EnetDisconnected\n'
    )
    out = launcher.scrub_for_public(raw)
    self.assertNotIn('File "', out)
    self.assertNotIn('alice', out)
    self.assertIn('melee.slippstream.EnetDisconnected', out)

  def test_redacts_home_paths_left_in_body(self):
    for raw in [
        r'error opening C:\Users\bob\slippi-ai\models\medium-v2',
        r'error opening C:/Users/bob/slippi-ai/models/medium-v2',
        r'error opening /Users/bob/slippi-ai/models/medium-v2',
        r'error opening /home/bob/slippi-ai/models/medium-v2',
    ]:
      with self.subTest(raw=raw):
        out = launcher.scrub_for_public(raw)
        self.assertNotIn('bob', out)
        self.assertIn('~', out)

  def test_leaves_non_path_text_alone(self):
    raw = 'Match ended with score 3-1 in stage FINAL_DESTINATION\n'
    self.assertEqual(launcher.scrub_for_public(raw), raw)

  def test_drops_caret_indicator_lines(self):
    raw = (
        '    gamestate = self.console.step()\n'
        '                ^^^^^^^^^^^^^^^^^^^\n'
        'Error: something\n'
    )
    out = launcher.scrub_for_public(raw)
    self.assertNotIn('^^^', out)
    self.assertIn('Error: something', out)


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


class DiscoverModelsTest(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.tmp_dir = pathlib.Path(self._tmp.name)
    self.models_dir = self.tmp_dir / 'models'
    self.models_dir.mkdir()
    self._orig_cache_path = launcher._model_cache_path
    launcher._model_cache_path = lambda: self.tmp_dir / 'cache.json'

  def tearDown(self):
    launcher._model_cache_path = self._orig_cache_path
    self._tmp.cleanup()

  def test_discover_models_scans_dir_and_caches(self):
    m1 = self.models_dir / 'model1'
    m1.write_text('dummy')
    m2 = self.models_dir / 'model2'
    m2.write_text('dummy')

    load_calls = []

    def mock_load(path):
      load_calls.append(path)
      if 'model1' in path:
        return ['fox', 'falco']
      return ['marth']

    orig_load = launcher._load_model_chars
    launcher._load_model_chars = mock_load
    try:
      res1 = launcher.discover_models(self.models_dir)
      self.assertEqual(res1, {'model1': ['fox', 'falco'], 'model2': ['marth']})
      self.assertEqual(len(load_calls), 2)

      # Second scan should hit cache
      load_calls.clear()
      res2 = launcher.discover_models(self.models_dir)
      self.assertEqual(res2, {'model1': ['fox', 'falco'], 'model2': ['marth']})
      self.assertEqual(len(load_calls), 0)

      # Removing a file drops it from cache and results
      m1.unlink()
      res3 = launcher.discover_models(self.models_dir)
      self.assertEqual(res3, {'model2': ['marth']})
    finally:
      launcher._load_model_chars = orig_load



class DiscordBotModelTest(unittest.TestCase):

  def test_format_models_reply_empty(self):
    bot = launcher.DiscordBotThread(None, lambda status: None)
    self.assertEqual(bot._format_models_reply(), 'No models loaded.')

  def test_format_models_reply_with_models(self):
    bot = launcher.DiscordBotThread(None, lambda status: None)
    bot._models = {
        'm1': ['fox', 'falco'],
        'm2': ['marth'],
    }
    bot._default_model = 'm1'
    reply = bot._format_models_reply()
    self.assertIn('Available models:', reply)
    self.assertIn('`m1` (default): fox, falco', reply)
    self.assertIn('`m2`: marth', reply)


if __name__ == '__main__':
  unittest.main()

