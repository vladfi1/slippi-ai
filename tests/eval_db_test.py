"""Tests for slippi_ai.eval_db. Run with `python -m tests.eval_db_test`."""

import os
import pickle
import random
import tempfile
import time
import unittest

import melee

from slippi_ai import dolphin, evaluation
from slippi_ai.eval_db import agents, db, matchups, ratings, runner


def imitation_state(characters: str, opponents: str = 'all', delay: int = 18):
  return dict(
      config=dict(
          dataset=dict(allowed_characters=characters, allowed_opponents=opponents),
          policy=dict(delay=delay),
      ),
      name_map={},
      state={},
  )


def rl_state(character: str, opponent: str, delay: int = 21):
  """A train_two-style state: one character trained against one opponent."""
  state = imitation_state(character, 'all', delay)
  state['agent_config'] = dict(name=['Cody', 'Hax'], rating=3000.0)
  state['opponent'] = opponent
  return state


def multichar_rl_state(characters: list[str], delay: int = 21):
  state = imitation_state(','.join(characters), 'all', delay)
  state['rl_config'] = dict(agent=dict(name='Master Player', rating=2500.0))
  return state


class Repo:
  """A temporary stripped_models / deployed_models pair."""

  def __init__(self, root: str):
    self.stripped = os.path.join(root, 'stripped_models')
    self.deployed = os.path.join(root, 'deployed_models')
    os.makedirs(os.path.join(self.stripped, 'jax'))
    os.makedirs(self.deployed)

  def write(self, rel_path: str, state: dict, mtime: float | None = None):
    path = os.path.join(self.stripped, rel_path)
    with open(path, 'wb') as f:
      pickle.dump(state, f)
    if mtime is not None:
      os.utime(path, (mtime, mtime))
    return path

  def link(self, name: str, rel_path: str):
    link = os.path.join(self.deployed, name)
    if os.path.lexists(link):
      os.remove(link)
    os.symlink(os.path.join('..', 'stripped_models', rel_path), link)

  def unlink(self, name: str):
    os.remove(os.path.join(self.deployed, name))

  def sync(self, conn, now='2026-01-01T00:00:00+00:00'):
    return agents.sync(conn, self.deployed, self.stripped, now=now)


class SyncTest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.repo = Repo(self.tmp.name)
    self.conn = db.connect(':memory:')

  def tearDown(self):
    self.conn.close()
    self.tmp.cleanup()

  def test_add_rename_update_alias_remove(self):
    repo, conn = self.repo, self.conn
    repo.write('fox_ditto', rl_state('fox', 'fox'), mtime=1000)
    repo.write('jax/marth_vs_fox', rl_state('marth', 'fox'), mtime=1000)
    repo.link('fox_ditto_v1', 'fox_ditto')
    repo.link('marth_vs_fox_v1', 'jax/marth_vs_fox')

    report = repo.sync(conn)
    self.assertEqual(len(report.added), 2)
    active = db.active_agents(conn)
    self.assertEqual(len(active), 2)
    by_path = {row['stripped_path']: row for row in active.values()}
    marth = by_path['jax/marth_vs_fox']
    self.assertEqual(db.loads(marth['characters']), ['MARTH'])
    self.assertEqual(db.loads(marth['opponents']), ['FOX'])
    self.assertEqual(marth['agent_type'], 'RL')
    self.assertEqual(marth['delay'], 21)
    self.assertEqual(db.loads(marth['rl_names']), ['Cody', 'Hax'])
    self.assertEqual(marth['rl_rating'], 3000.0)
    fox_hash = by_path['fox_ditto']['hash']

    # Unchanged files are not rehashed or re-added.
    report = repo.sync(conn)
    self.assertEqual(report.summary(),
                     'added=0 updated=0 renamed=0 removed=0 unchanged=2 skipped=0')

    # Renaming a symlink keeps the agent and its hash; the old name retires.
    repo.unlink('fox_ditto_v1')
    repo.link('fox_ditto_v2', 'fox_ditto')
    report = repo.sync(conn)
    self.assertEqual(len(report.renamed), 1)
    self.assertEqual(report.removed, ['fox_ditto_v1'])
    self.assertEqual(db.names_by_hash(conn)[fox_hash], ['fox_ditto_v2'])
    self.assertEqual(
        db.names_by_hash(conn, active_only=False)[fox_hash],
        ['fox_ditto_v1', 'fox_ditto_v2'])
    self.assertEqual(db.display_name(conn, fox_hash), 'fox_ditto_v2')

    # Two names for one file are aliases of one agent.
    repo.link('gold', 'fox_ditto')
    repo.sync(conn)
    self.assertEqual(db.names_by_hash(conn)[fox_hash], ['fox_ditto_v2', 'gold'])
    self.assertEqual(len(db.active_agents(conn)), 2)

    # Rewriting the file with a new mtime is a new agent under the same names;
    # the old hash stays in the table but is no longer active.
    repo.write('fox_ditto', rl_state('fox', 'fox', delay=24), mtime=2000)
    report = repo.sync(conn)
    self.assertEqual(len(report.added), 1)
    self.assertEqual(len(report.updated), 2)
    active = db.active_agents(conn)
    self.assertEqual(len(active), 2)
    self.assertNotIn(fox_hash, active)
    self.assertIn(fox_hash, db.all_agents(conn))
    new_fox = [h for h, row in active.items()
               if row['stripped_path'] == 'fox_ditto'][0]
    self.assertEqual(active[new_fox]['delay'], 24)
    self.assertEqual(sorted(db.names_by_hash(conn)[new_fox]),
                     ['fox_ditto_v2', 'gold'])
    self.assertIn('(retired)', db.display_name(conn, fox_hash))

  def test_touched_mtime_same_bytes_keeps_hash(self):
    repo, conn = self.repo, self.conn
    repo.write('fox', rl_state('fox', 'fox'), mtime=1000)
    repo.link('fox', 'fox')
    repo.sync(conn)
    (old_hash,) = db.active_agents(conn)
    os.utime(os.path.join(repo.stripped, 'fox'), (3000, 3000))
    report = repo.sync(conn)
    self.assertEqual(report.added, [])
    self.assertEqual(report.unchanged, 1)
    (new_hash,) = db.active_agents(conn)
    self.assertEqual(old_hash, new_hash)
    self.assertEqual(db.all_agents(conn)[new_hash]['mtime'], 3000)

  def test_unreadable_file_is_skipped(self):
    repo, conn = self.repo, self.conn
    with open(os.path.join(repo.stripped, 'junk'), 'wb') as f:
      f.write(b'not a pickle')
    repo.link('junk', 'junk')
    repo.write('fox', rl_state('fox', 'fox'))
    repo.link('fox', 'fox')
    report = repo.sync(conn)
    self.assertEqual(len(report.skipped), 1)
    self.assertEqual(len(report.added), 1)


class MatchupTest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.repo = Repo(self.tmp.name)
    self.conn = db.connect(':memory:')
    repo = self.repo
    repo.write('fox_ditto', rl_state('fox', 'fox'))
    repo.write('fox_vs_marth', rl_state('fox', 'marth'))
    repo.write('marth_vs_fox', rl_state('marth', 'fox'))
    repo.write('marth_imitation', imitation_state('marth'))
    repo.write('multi', multichar_rl_state(['fox', 'marth', 'peach']))
    for name in ['fox_ditto', 'fox_vs_marth', 'marth_vs_fox',
                 'marth_imitation', 'multi']:
      repo.link(name, name)
    repo.sync(self.conn)
    self.agents = db.active_agents(self.conn)
    self.by_name = {
        db.display_name(self.conn, h): row for h, row in self.agents.items()}

  def tearDown(self):
    self.conn.close()
    self.tmp.cleanup()

  def player(self, name, character):
    return matchups.Player(self.by_name[name]['hash'], character)

  def pairs(self, names, filters=matchups.MatchupFilters(max_tier_gap=None),
            tiers=None):
    agents = {self.by_name[n]['hash']: self.by_name[n] for n in names}
    return set(matchups.relevant_pairs(agents, filters, tiers))

  def test_relevance(self):
    cp = matchups.canonical_pair
    # Both trained against each other's character.
    self.assertEqual(
        self.pairs(['fox_vs_marth', 'marth_vs_fox']),
        {cp(self.player('fox_vs_marth', 'FOX'), self.player('marth_vs_fox', 'MARTH'))})
    # A Fox ditto agent never faced Marth.
    self.assertEqual(self.pairs(['fox_ditto', 'marth_vs_fox']), set())
    # Imitation agents accept any opponent, but the other side must accept them.
    self.assertEqual(self.pairs(['marth_imitation', 'fox_ditto']), set())
    self.assertEqual(len(self.pairs(['marth_imitation', 'fox_vs_marth'])), 1)
    # No self play, and each unordered pair appears once.
    self.assertEqual(self.pairs(['fox_ditto']), set())
    all_pairs = matchups.relevant_pairs(
        self.agents, matchups.MatchupFilters(max_tier_gap=None))
    self.assertEqual(len(all_pairs), len(set(all_pairs)))
    for a, b in all_pairs:
      self.assertLessEqual(a, b)
      self.assertNotEqual(a.hash, b.hash)

  def test_multichar_agent_is_one_player_per_character(self):
    # multi plays fox/marth/peach and (self-play RL) faces the same set.
    self.assertEqual(
        self.pairs(['multi', 'fox_vs_marth']),
        {matchups.canonical_pair(
            self.player('multi', 'MARTH'), self.player('fox_vs_marth', 'FOX'))})
    got = self.pairs(['multi', 'marth_imitation'])
    self.assertEqual(
        {p.character for pair in got for p in pair if p.hash == self.by_name['multi']['hash']},
        {'FOX', 'MARTH', 'PEACH'})
    self.assertEqual(len(got), 3)

  def test_weak_filter(self):
    strong = matchups.strong_agents(self.conn, self.agents)
    strong_names = {db.display_name(self.conn, h) for h in strong}
    self.assertEqual(
        strong_names, {'fox_ditto', 'fox_vs_marth', 'marth_vs_fox', 'multi'})
    row = self.by_name['fox_ditto']
    self.assertTrue(matchups.is_weak(row, ['gold-v2']))
    self.assertTrue(matchups.is_weak(row, ['falco_d21_vs_ics_plat']))
    self.assertTrue(matchups.is_weak(row, ['fox_ditto', 'diamond']))
    self.assertFalse(matchups.is_weak(row, ['super-gm', 'master-v2']))
    self.assertFalse(matchups.is_weak(row, ['fox_d21_ditto_v6.1']))
    pending_all = matchups.pending_matchups(self.conn, 'k', pool='all')
    pending_strong = matchups.pending_matchups(self.conn, 'k')
    self.assertLess(len(pending_all_pairs(pending_strong)),
                    len(pending_all_pairs(pending_all)))
    for m in pending_strong:
      self.assertNotIn(self.by_name['marth_imitation']['hash'], m.pair)

  def test_tiers(self):
    t = matchups.tier_from_names
    self.assertEqual(t(['super-gm']), 'super-gm')
    self.assertEqual(t(['gm-v2']), 'gm')
    self.assertEqual(t(['medium-v1']), 'plat')
    self.assertEqual(t(['falco_d21_vs_ics_plat']), 'plat')
    self.assertEqual(t(['gold', 'silver-v2']), 'gold')  # strongest alias wins
    self.assertIsNone(t(['fox_d21_ditto_v6.1', 'top12']))
    # Single-character RL agents are super-gm whatever their name; imitation
    # agents are bronze; multi-character agents go by name.
    self.assertEqual(
        matchups.tier_of(self.by_name['fox_ditto'], ['fox_ditto']), 'super-gm')
    self.assertEqual(
        matchups.tier_of(self.by_name['fox_ditto'], ['fox_vs_ics_master']),
        'super-gm')
    self.assertEqual(
        matchups.tier_of(self.by_name['marth_imitation'], ['marth_imitation']),
        'bronze')
    self.assertIsNone(matchups.tier_of(self.by_name['multi'], ['multi']))
    self.assertEqual(matchups.tier_of(self.by_name['multi'], ['gold-v3']), 'gold')
    self.assertEqual(matchups.tier_gap('bronze', 'silver'), 1)
    self.assertEqual(matchups.tier_gap('plat', 'super-gm'), 4)
    self.assertIsNone(matchups.tier_gap(None, 'gm'))

  def test_tier_gap_filter(self):
    fox, marth = self.by_name['fox_vs_marth'], self.by_name['marth_vs_fox']
    agents = {fox['hash']: fox, marth['hash']: marth}
    close = {fox['hash']: 'gm', marth['hash']: 'super-gm'}
    far = {fox['hash']: 'master', marth['hash']: 'super-gm'}
    f1 = matchups.MatchupFilters(max_tier_gap=1)
    self.assertEqual(len(matchups.relevant_pairs(agents, f1, close)), 1)
    self.assertEqual(len(matchups.relevant_pairs(agents, f1, far)), 0)
    self.assertEqual(len(matchups.relevant_pairs(
        agents, matchups.MatchupFilters(max_tier_gap=None), far)), 1)
    # Untiered agents are never restricted.
    self.assertEqual(len(matchups.relevant_pairs(
        agents, f1, {fox['hash']: None, marth['hash']: 'super-gm'})), 1)
    # With real tiers, bronze imitation vs super-gm 1v1 agents is out by
    # default while the untiered multi agent plays everyone.
    default = pending_all_pairs(
        matchups.pending_matchups(self.conn, 'k', pool='all'))
    imit = self.by_name['marth_imitation']['hash']
    multi = self.by_name['multi']['hash']
    self.assertTrue(all(
        multi in (a.hash, b.hash)
        for a, b in default if imit in (a.hash, b.hash)))

  def test_character_filters(self):
    multi, imit = self.by_name['multi'], self.by_name['marth_imitation']
    fox, marth = self.by_name['fox_vs_marth'], self.by_name['marth_vs_fox']
    only_fox = matchups.MatchupFilters(None, allowed_chars=frozenset({'FOX'}))
    # Both sides restricted: multi keeps only Fox, imitation Marth is out.
    self.assertEqual(self.pairs(['multi', 'marth_imitation'], only_fox), set())
    fox_marth = matchups.MatchupFilters(
        None, allowed_chars=frozenset({'FOX', 'MARTH'}))
    got = self.pairs(['multi', 'marth_imitation'], fox_marth)
    self.assertEqual(
        sorted(p.character for pair in got for p in pair
               if p.hash == multi['hash']), ['FOX', 'MARTH'])
    # Targeted matchup: one side Fox, the other Marth, either way round.
    target = matchups.MatchupFilters(
        None, allowed_chars=frozenset({'FOX'}),
        allowed_opponents=frozenset({'MARTH'}))
    self.assertEqual(len(self.pairs(['fox_vs_marth', 'marth_vs_fox'], target)), 1)
    self.assertEqual(len(self.pairs(['marth_vs_fox', 'fox_vs_marth'], target)), 1)
    self.assertEqual(len(self.pairs(['fox_ditto', 'fox_vs_marth'], target)), 0)
    self.assertTrue(target.characters_ok('MARTH', 'FOX'))
    self.assertFalse(target.characters_ok('FOX', 'FOX'))

  def test_batching_and_coverage(self):
    params = runner.EvalParams()
    kw = dict(pool='all', filters=matchups.MatchupFilters(None))
    pending = matchups.pending_matchups(self.conn, params.key(), **kw)
    # One eval per agent pair when everything fits.
    self.assertEqual(len({m.pair for m in pending}), len(pending))
    for m in pending:
      self.assertLess(m.p1_hash, m.p2_hash)
    multi_vs_imit = [m for m in pending if set(m.pair) == {
        self.by_name['multi']['hash'], self.by_name['marth_imitation']['hash']}]
    self.assertEqual(len(multi_vs_imit), 1)
    self.assertEqual(len(multi_vs_imit[0].character_pairs), 3)
    # Capping pairs per eval splits an agent pair into several evals.
    split = matchups.pending_matchups(
        self.conn, params.key(), max_pairs_per_eval=2, **kw)
    self.assertEqual(len(split), len(pending) + 1)

    # Finishing an eval covers exactly its character pairs, in either
    # orientation; the remaining pairs of that agent pair stay pending.
    (m,) = multi_vs_imit
    partial = matchups.Matchup(m.p1_hash, m.p2_hash, m.character_pairs[:1])
    eval_id = runner.start_eval(self.conn, partial, params)
    still = matchups.pending_matchups(self.conn, params.key(), **kw)
    rest = [x for x in still if x.pair == m.pair]
    self.assertEqual(len(rest), 1)
    self.assertEqual(len(rest[0].character_pairs), 2)
    runner.finish_eval(self.conn, eval_id, fake_games(4, p1_wins=2))
    self.assertEqual(
        len([x for x in matchups.pending_matchups(self.conn, params.key(), **kw)
             if x.pair == m.pair][0].character_pairs), 2)
    # Different params start over.
    other = runner.EvalParams(
        config=params.config, stage=runner.melee.Stage.FINAL_DESTINATION)
    self.assertEqual(
        pending_all_pairs(matchups.pending_matchups(self.conn, other.key(), **kw)),
        pending_all_pairs(pending))
    # ... except the rollout length and env count, which only decide how many
    # games get played.
    longer = runner.EvalParams(config=runner.dataclasses.replace(
        params.config, rollout_length=params.config.rollout_length + 1,
        num_envs=params.config.num_envs + 1))
    self.assertEqual(longer.key(), params.key())
    self.assertNotEqual(longer.to_dict(), params.to_dict())
    # Agents that already have evals sort after those with none.
    counts = matchups.eval_counts(self.conn)
    self.assertEqual(counts[m.p1_hash], 1)
    first = matchups.pending_matchups(self.conn, params.key(), **kw)[0]
    self.assertNotIn(m.p1_hash, first.pair)
    self.assertNotIn(m.p2_hash, first.pair)

  def test_ladder_pool(self):
    row = self.by_name['multi']
    self.assertTrue(matchups.is_ladder(row, ['gm-v2']))
    self.assertTrue(matchups.is_ladder(row, ['bronze']))
    self.assertTrue(matchups.is_ladder(row, ['medium-v1']))
    self.assertFalse(matchups.is_ladder(row, ['top12_rl']))
    # Single-character agents are never ladder agents, whatever the name.
    self.assertFalse(matchups.is_ladder(self.by_name['fox_ditto'], ['gm']))
    self.assertEqual(matchups.select_pool(self.conn, self.agents, 'ladder'), {})
    self.assertEqual(
        matchups.select_pool(self.conn, self.agents, 'strong-ladder'), {})
    self.assertEqual(
        matchups.pending_matchups(self.conn, 'k', pool='ladder'), [])
    with self.assertRaises(ValueError):
      matchups.select_pool(self.conn, self.agents, 'bogus')


def pending_all_pairs(ms):
  return {pair for m in ms for pair in m.player_pairs}


def fake_games(n: int, p1_wins: int, ties: int = 0,
               characters=('FOX', 'MARTH')) -> list[dict]:
  games = []
  for i in range(n):
    if i < p1_wins:
      winner, stocks = 1, (2, 0)
    elif i < p1_wins + ties:
      winner, stocks = None, (1, 1)
    else:
      winner, stocks = 2, (0, 3)
    games.append(dict(
        env_id=i, episode_id=0, stage='BATTLEFIELD', stage_id=31,
        frame_id=6000, frames=6123, characters=tuple(characters),
        winner_port=winner, stocks=stocks, percents=(10.0, 20.5),
        match_ended=True, stockout=winner is not None,
        max_frame_reached=winner is None))
  return games


def fake_result(games: list[dict]) -> evaluation.EvaluationResult:
  return evaluation.EvaluationResult(
      config=runner.DEFAULT_CONFIG, total_steps=1, rewards={1: 0.0, 2: 0.0},
      kdpm=0.5, completed_games=games, timings={}, env_fps=100.0,
      player_fps=200.0, sps=1.0, character_pairs=[('FOX', 'MARTH')])


# Stand-ins for runner.evaluate_matchup; module-level so a spawned child can
# import them by name.
def evaluate_ok(matchup, params, p1_path, p2_path):
  del params
  assert matchup.character_pairs == (('FOX', 'MARTH'),)
  paths = (os.path.basename(p1_path), os.path.basename(p2_path))
  assert paths == ('aaa', 'bbb'), paths
  return fake_result(fake_games(5, p1_wins=3))


def evaluate_raises(matchup, params, p1_path, p2_path):
  del matchup, params, p1_path, p2_path
  raise RuntimeError('agent exploded')


def evaluate_crashes(matchup, params, p1_path, p2_path):
  del matchup, params, p1_path, p2_path
  os._exit(3)  # like a segfault: no exception, no report


def insert_agent(conn, h, characters, opponents, active=True,
                 now='2026-01-01T00:00:00+00:00'):
  conn.execute(
      """INSERT INTO agents VALUES (?, ?, 0, 0, 'jax', 'RL', 21, ?, ?,
         NULL, NULL, ?, ?)""",
      (h, h[:3], db.dumps(characters), db.dumps(opponents), now, now))
  conn.execute(
      'INSERT INTO deployed_names VALUES (?, ?, ?, ?, ?)',
      (h[:3], h, now, now, int(active)))


class IngestTest(unittest.TestCase):

  def test_finish_eval_stores_summary_and_games(self):
    conn = db.connect(':memory:')
    insert_agent(conn, 'a' * 32, ['FOX'], ['FOX', 'MARTH'])
    insert_agent(conn, 'b' * 32, ['MARTH'], ['FOX', 'MARTH'])
    matchup = matchups.Matchup('a' * 32, 'b' * 32, (('FOX', 'MARTH'),))
    params = runner.EvalParams()
    eval_id = runner.start_eval(conn, matchup, params)
    row = conn.execute('SELECT * FROM evals WHERE id = ?', (eval_id,)).fetchone()
    self.assertEqual(row['status'], 'running')
    self.assertEqual(db.loads(row['params'])['num_envs'], 1024)
    self.assertEqual(db.loads(row['character_pairs']), [['FOX', 'MARTH']])

    runner.finish_eval(
        conn, eval_id, fake_games(10, p1_wins=6, ties=1), kdpm=1.5, env_fps=3e4)
    row = conn.execute('SELECT * FROM evals WHERE id = ?', (eval_id,)).fetchone()
    self.assertEqual(row['status'], 'done')
    self.assertEqual(
        (row['num_games'], row['p1_wins'], row['p2_wins'], row['ties'],
         row['timeouts']), (10, 6, 3, 1, 1))
    games = conn.execute(
        'SELECT * FROM games WHERE eval_id = ? ORDER BY env_id', (eval_id,)
    ).fetchall()
    self.assertEqual(len(games), 10)
    self.assertEqual(games[0]['p1_character'], 'FOX')
    self.assertEqual(games[0]['p2_character'], 'MARTH')
    self.assertEqual(games[6]['winner_port'], None)
    self.assertEqual(games[9]['p2_stocks'], 3)

    empty = runner.start_eval(conn, matchup, params)
    runner.finish_eval(conn, empty, [])
    row = conn.execute('SELECT * FROM evals WHERE id = ?', (empty,)).fetchone()
    self.assertEqual((row['status'], row['error']), ('failed', 'no completed games'))

    # Games are attributed to (agent, character) players, so records that
    # lost their characters must not be ingested silently.
    unattributed = runner.start_eval(conn, matchup, params)
    games = fake_games(4, p1_wins=2)
    for game in games:
      del game['characters']
    with self.assertRaises(ValueError):
      runner.finish_eval(conn, unattributed, games)
    # run_eval turns the exception into a failed eval.
    runner.fail_eval(conn, unattributed, 'no characters')

    other = runner.start_eval(conn, matchup, params)
    runner.fail_eval(conn, other, 'boom')
    self.assertEqual(runner.reset_stale_running(conn), 0)
    third = runner.start_eval(conn, matchup, params)
    self.assertEqual(runner.reset_stale_running(conn), 1)
    row = conn.execute('SELECT status FROM evals WHERE id = ?', (third,)).fetchone()
    self.assertEqual(row['status'], 'failed')

  def test_schema_migration(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'old.sqlite')
      conn = db.connect(path)
      insert_agent(conn, 'a' * 32, ['FOX'], ['FOX'])
      conn.commit()
      # Pretend to be an older, empty layout: version 1 with a stale table.
      conn.execute('DROP TABLE ratings')
      conn.execute('CREATE TABLE ratings (run_id INTEGER, hash TEXT)')
      conn.execute('PRAGMA user_version = 1')
      conn.commit()
      conn.close()
      conn = db.connect(path)
      self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0], 2)
      self.assertIn('character',
                    [r[1] for r in conn.execute('PRAGMA table_info(ratings)')])
      self.assertEqual(len(db.all_agents(conn)), 1)
      # Populated eval tables are never dropped silently.
      m = matchups.Matchup('a' * 32, 'a' * 32, (('FOX', 'FOX'),))
      runner.start_eval(conn, m, runner.EvalParams())
      conn.execute('PRAGMA user_version = 1')
      conn.commit()
      conn.close()
      with self.assertRaises(RuntimeError):
        db.connect(path)


class RunEvalTest(unittest.TestCase):

  def setUp(self):
    self.conn = db.connect(':memory:')
    insert_agent(self.conn, 'a' * 32, ['FOX'], ['FOX', 'MARTH'])
    insert_agent(self.conn, 'b' * 32, ['MARTH'], ['FOX', 'MARTH'])
    self.matchup = matchups.Matchup('a' * 32, 'b' * 32, (('FOX', 'MARTH'),))

  def run_eval(self, evaluate_fn, isolate=True):
    eval_id = runner.run_eval(
        self.conn, self.matchup, runner.EvalParams(), 'stripped',
        isolate=isolate, evaluate_fn=evaluate_fn)
    return self.conn.execute(
        'SELECT * FROM evals WHERE id = ?', (eval_id,)).fetchone()

  def assert_done(self, row):
    self.assertEqual(row['status'], 'done')
    self.assertEqual(
        (row['num_games'], row['p1_wins'], row['p2_wins']), (5, 3, 2))
    self.assertEqual((row['p1_kdpm'], row['env_fps']), (0.5, 100.0))
    games = self.conn.execute(
        'SELECT COUNT(*) FROM games WHERE eval_id = ?', (row['id'],)).fetchone()
    self.assertEqual(games[0], 5)

  def test_in_process(self):
    self.assert_done(self.run_eval(evaluate_ok, isolate=False))
    row = self.run_eval(evaluate_raises, isolate=False)
    self.assertEqual(row['status'], 'failed')
    self.assertIn('agent exploded', row['error'])

  def test_child_process(self):
    self.assert_done(self.run_eval(evaluate_ok))

  def test_child_exception_is_recorded(self):
    row = self.run_eval(evaluate_raises)
    self.assertEqual(row['status'], 'failed')
    # The child's traceback, not just the parent's.
    self.assertIn('agent exploded', row['error'])
    self.assertIn('evaluate_raises', row['error'])

  def test_child_crash_is_recorded(self):
    row = self.run_eval(evaluate_crashes)
    self.assertEqual(row['status'], 'failed')
    self.assertIn('exited with code 3', row['error'])
    # The session goes on: the next eval still works.
    self.assert_done(self.run_eval(evaluate_ok))


class RatingsTest(unittest.TestCase):

  def test_recovers_ordering_and_scale(self):
    rng = random.Random(0)
    true = {'a': 400.0, 'b': 200.0, 'c': 0.0, 'd': -300.0}
    players = {n: matchups.Player(n * 32, 'FOX') for n in true}
    names = list(true)
    results = []
    for i, p1 in enumerate(names):
      for p2 in names[i + 1:]:
        p_win = 1 / (1 + 10 ** ((true[p2] - true[p1]) / 400))
        wins = sum(rng.random() < p_win for _ in range(2000))
        results.append(
            ratings.PairResult(players[p1], players[p2], wins, 2000 - wins, 0))
    result = ratings.fit(results, alpha=1e-4)
    est = {n: result.ratings[players[n]] for n in names}
    self.assertEqual(sorted(est, key=est.get, reverse=True), names)
    # Ratings are anchored to mean zero, so compare centered values.
    mean = sum(true.values()) / len(true)
    for name in names:
      self.assertAlmostEqual(est[name], true[name] - mean, delta=25)
    self.assertAlmostEqual(sum(est.values()), 0.0, places=6)
    self.assertEqual(result.num_games[players['a']], 6000)

  def test_ties_count_half(self):
    a, b = matchups.Player('a' * 32, 'FOX'), matchups.Player('b' * 32, 'FOX')
    result = ratings.fit([ratings.PairResult(a, b, 0, 0, 100)], alpha=1e-3)
    self.assertAlmostEqual(result.ratings[a], result.ratings[b], places=6)
    self.assertEqual(result.score[a], 50)

  def test_undefeated_stays_finite_and_disconnected_ok(self):
    p = lambda n: matchups.Player(n * 32, 'FOX')
    results = [
        ratings.PairResult(p('a'), p('b'), 50, 0, 0),
        ratings.PairResult(p('c'), p('d'), 30, 20, 0),  # separate component
    ]
    result = ratings.fit(results, alpha=0.01)
    for value in result.ratings.values():
      self.assertTrue(abs(value) < 5000)
    self.assertGreater(result.ratings[p('a')], result.ratings[p('b')])
    self.assertGreater(result.ratings[p('c')], result.ratings[p('d')])

  def test_players_are_agent_character_pairs(self):
    conn = db.connect(':memory:')
    insert_agent(conn, 'a' * 32, ['FOX', 'MARTH'], ['FOX', 'MARTH'])
    insert_agent(conn, 'b' * 32, ['FOX'], ['FOX', 'MARTH'])
    insert_agent(conn, 'c' * 32, ['FOX'], ['FOX', 'MARTH'], active=False)
    params = runner.EvalParams()
    # a-fox and a-marth both play b-fox in one eval; games carry characters.
    m = matchups.Matchup('a' * 32, 'b' * 32, (('FOX', 'FOX'), ('MARTH', 'FOX')))
    eval_id = runner.start_eval(conn, m, params)
    games = (fake_games(10, p1_wins=8, characters=('FOX', 'FOX'))
             + fake_games(10, p1_wins=2, characters=('MARTH', 'FOX')))
    runner.finish_eval(conn, eval_id, games)
    # A retired agent's eval, recorded the other way round.
    m2 = matchups.Matchup('b' * 32, 'c' * 32, (('FOX', 'FOX'),))
    runner.finish_eval(conn, runner.start_eval(conn, m2, params),
                       fake_games(10, p1_wins=5, characters=('FOX', 'FOX')))

    results, num_evals = ratings.load_pair_results(conn, {'a' * 32, 'b' * 32})
    self.assertEqual(num_evals, 1)
    self.assertEqual(len(results), 2)
    a_fox, a_marth = matchups.Player('a' * 32, 'FOX'), matchups.Player('a' * 32, 'MARTH')
    b_fox = matchups.Player('b' * 32, 'FOX')
    by_pair = {(r.p1, r.p2): r for r in results}
    self.assertEqual(by_pair[(a_fox, b_fox)].p1_wins, 8)
    self.assertEqual(by_pair[(a_marth, b_fox)].p1_wins, 2)

    run_id, result = ratings.compute_and_store(conn)
    self.assertEqual(set(result.ratings), {a_fox, a_marth, b_fox})
    self.assertGreater(result.ratings[a_fox], result.ratings[b_fox])
    self.assertGreater(result.ratings[b_fox], result.ratings[a_marth])
    self.assertEqual(result.num_games[b_fox], 20)
    board = ratings.leaderboard(conn, run_id)
    self.assertEqual([(r['name'], r['character']) for r in board],
                     [('aaa', 'FOX'), ('bbb', 'FOX'), ('aaa', 'MARTH')])
    self.assertAlmostEqual(board[1]['win_rate'], 10 / 20)
    _, result = ratings.compute_and_store(conn, active_only=False)
    self.assertEqual(len(result.ratings), 4)
    self.assertEqual(result.num_evals, 2)
    self.assertEqual(ratings.latest_run_id(conn), run_id + 1)


class EvalParamsTest(unittest.TestCase):

  def test_defaults_match_shell_script(self):
    script = runner.parse_shell_script_flags(runner.EVALUATOR_SHELL_SCRIPT)
    config = runner.DEFAULT_CONFIG
    for name, value in script.items():
      if name in ('use_gpu',):
        continue  # boolean flags parse as 'true' below
      self.assertTrue(hasattr(config, name), name)
    self.assertEqual(config.num_envs, int(script['num_envs']))
    self.assertEqual(config.rollout_length, int(script['rollout_length']))

    self.assertEqual(config.chunk_length, int(script['chunk_length']))
    self.assertEqual(config.num_env_steps, int(script['num_env_steps']))
    self.assertEqual(config.num_agent_steps, int(script['num_agent_steps']))
    for flag in ('sim_envs', 'async_envs', 'use_gpu'):
      self.assertEqual(script[flag], 'true')
      self.assertTrue(getattr(config, flag))

  def test_rekey_evals_updates_stale_keys(self):
    conn = db.connect(':memory:')
    insert_agent(conn, 'a' * 32, ['FOX'], ['FOX'])
    insert_agent(conn, 'b' * 32, ['FOX'], ['FOX'])
    matchup = matchups.Matchup('a' * 32, 'b' * 32, (('FOX', 'FOX'),))
    params = runner.EvalParams()
    eval_id = runner.start_eval(conn, matchup, params)
    # Simulate a row written when rollout_length was part of the key.
    conn.execute('UPDATE evals SET params_key = params WHERE id = ?', (eval_id,))
    self.assertEqual(len(matchups.covered_pairs(conn, params.key())), 0)
    self.assertEqual(runner.rekey_evals(conn), 1)
    self.assertEqual(runner.rekey_evals(conn), 0)
    self.assertEqual(len(matchups.covered_pairs(conn, params.key())), 1)

  def test_key_ignores_presentation_fields(self):
    a = runner.EvalParams()
    b = runner.EvalParams(config=runner.dataclasses.replace(
        a.config, quiet=False, burnin=True))
    self.assertEqual(a.key(), b.key())
    c = runner.EvalParams(stage=melee.Stage.BATTLEFIELD)
    self.assertNotEqual(a.key(), c.key())

  def test_to_flags_roundtrip_shape(self):
    flags = runner.EvalParams().to_flags()
    self.assertIn('--num_envs=1024', flags)
    self.assertIn('--sim_envs', flags)
    self.assertIn('--nofake_envs', flags)
    self.assertIn('--dolphin.stage=RANDOM_STAGE', flags)


class EvaluationLibTest(unittest.TestCase):

  def test_per_env_character_pairs_cycle(self):
    players = {1: dolphin.AI(melee.Character.FOX), 2: dolphin.AI(melee.Character.MARTH)}
    pairs = evaluation.character_pairs_from_lists(
        {1: [melee.Character.FOX, melee.Character.FALCO], 2: []})
    self.assertEqual(pairs, [(melee.Character.FOX, None), (melee.Character.FALCO, None)])
    per_env = evaluation.per_env_dolphin_kwargs(
        dict(stage=melee.Stage.RANDOM_STAGE), players, pairs, num_envs=4)
    chars = [(kw['players'][1].character, kw['players'][2].character)
             for kw in per_env]
    self.assertEqual(chars, [
        (melee.Character.FOX, melee.Character.MARTH),
        (melee.Character.FALCO, melee.Character.MARTH),
        (melee.Character.FOX, melee.Character.MARTH),
        (melee.Character.FALCO, melee.Character.MARTH),
    ])
    self.assertEqual(per_env[0]['stage'], melee.Stage.RANDOM_STAGE)
    # Explicit pairs need not be a product.
    per_env = evaluation.per_env_dolphin_kwargs(
        dict(), players,
        [(melee.Character.FOX, melee.Character.FALCO),
         (melee.Character.MARTH, melee.Character.FOX)], num_envs=3)
    self.assertEqual(
        [(kw['players'][1].character, kw['players'][2].character) for kw in per_env],
        [(melee.Character.FOX, melee.Character.FALCO),
         (melee.Character.MARTH, melee.Character.FOX),
         (melee.Character.FOX, melee.Character.FALCO)])
    # Untouched when nothing is overridden.
    kwargs = dict(stage=melee.Stage.RANDOM_STAGE, players=players)
    self.assertEqual(evaluation.character_pairs_from_lists({1: [], 2: []}), [])
    self.assertIs(evaluation.per_env_dolphin_kwargs(kwargs, players, [], 4), kwargs)

  def test_default_player_kwargs_match_flags(self):
    kwargs = evaluation.default_player_kwargs('some/path', melee.Character.MARTH)
    self.assertEqual(kwargs['type'], 'ai')
    self.assertEqual(kwargs['character'], melee.Character.MARTH)
    self.assertEqual(kwargs['ai']['path'], 'some/path')
    self.assertTrue(kwargs['ai']['tf']['jit_compile'])
    self.assertTrue(kwargs['ai']['jax']['pack_args'])

  def test_summary(self):
    games = fake_games(10, p1_wins=6, ties=1)
    s = evaluation.summarize_games(games)
    self.assertEqual((s['wins'], s['losses'], s['ties'], s['timeouts']), (6, 3, 1, 1))
    text = evaluation.format_game_summary(games)
    self.assertIn('win_rate=0.600', text)
    self.assertIn('stage BATTLEFIELD', text)
    self.assertEqual(evaluation.format_game_summary([]), 'completed games: 0')


if __name__ == '__main__':
  unittest.main()
