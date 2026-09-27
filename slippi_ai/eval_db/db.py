"""Sqlite schema and helpers for the agent evaluation database."""

import datetime
import json
import sqlite3
import typing as tp

# Bump when tables change shape. `connect` recreates the eval/rating tables
# when they are empty and refuses to open a populated database that is behind.
SCHEMA_VERSION = 2
VERSIONED_TABLES = ('games', 'ratings', 'rating_runs', 'evals')

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
  -- md5 of the stripped model file; the stable identity of an agent.
  hash TEXT PRIMARY KEY,
  -- Path of the model file relative to the stripped_models root.
  stripped_path TEXT NOT NULL,
  mtime REAL NOT NULL,
  size INTEGER NOT NULL,
  platform TEXT,             -- 'tf' or 'jax'
  agent_type TEXT NOT NULL,  -- 'IMITATION' or 'RL'
  delay INTEGER NOT NULL,
  characters TEXT NOT NULL,  -- JSON list of libmelee character names
  opponents TEXT NOT NULL,   -- JSON list of libmelee character names
  rl_names TEXT,             -- JSON list of names the agent was trained with
  rl_rating REAL,            -- rating the agent was conditioned on, if any
  first_seen TEXT NOT NULL,
  last_seen TEXT NOT NULL
);

-- deployed_models entries are symlinks that get renamed and repointed, so a
-- name is an alias of a hash for some period rather than an identity.
CREATE TABLE IF NOT EXISTS deployed_names (
  name TEXT NOT NULL,
  hash TEXT NOT NULL REFERENCES agents(hash),
  first_seen TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (name, hash)
);
CREATE INDEX IF NOT EXISTS idx_deployed_names_hash ON deployed_names(hash);

CREATE TABLE IF NOT EXISTS evals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  -- Ports don't matter, so pairs are stored with p1_hash <= p2_hash.
  p1_hash TEXT NOT NULL REFERENCES agents(hash),
  p2_hash TEXT NOT NULL REFERENCES agents(hash),
  -- JSON list of [p1_character, p2_character] cycled across envs.
  character_pairs TEXT NOT NULL,
  params TEXT NOT NULL,      -- JSON of the evaluator parameters
  params_key TEXT NOT NULL,  -- canonical form of params, for "already done"
  status TEXT NOT NULL,      -- 'running', 'done' or 'failed'
  started_at TEXT NOT NULL,
  finished_at TEXT,
  git_commit TEXT,
  log_path TEXT,
  error TEXT,
  num_games INTEGER,
  p1_wins INTEGER,
  p2_wins INTEGER,
  ties INTEGER,
  timeouts INTEGER,
  p1_kdpm REAL,
  env_fps REAL
);
CREATE INDEX IF NOT EXISTS idx_evals_pair ON evals(p1_hash, p2_hash);
CREATE INDEX IF NOT EXISTS idx_evals_status ON evals(status);

CREATE TABLE IF NOT EXISTS games (
  eval_id INTEGER NOT NULL REFERENCES evals(id),
  env_id INTEGER NOT NULL,
  episode_id INTEGER NOT NULL,
  stage TEXT NOT NULL,
  frames INTEGER NOT NULL,
  winner_port INTEGER,       -- 1, 2 or NULL for a tie
  p1_character TEXT,
  p2_character TEXT,
  p1_stocks INTEGER,
  p2_stocks INTEGER,
  p1_percent REAL,
  p2_percent REAL,
  stockout INTEGER NOT NULL,
  max_frame_reached INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_games_eval ON games(eval_id);

CREATE TABLE IF NOT EXISTS rating_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  computed_at TEXT NOT NULL,
  method TEXT NOT NULL,
  params TEXT NOT NULL,      -- JSON
  num_agents INTEGER NOT NULL,
  num_evals INTEGER NOT NULL,
  num_games INTEGER NOT NULL
);

-- A rated player is an (agent, character) pair: gm-fox and gm-falco are
-- separate players.
CREATE TABLE IF NOT EXISTS ratings (
  run_id INTEGER NOT NULL REFERENCES rating_runs(id),
  hash TEXT NOT NULL REFERENCES agents(hash),
  character TEXT NOT NULL,
  rating REAL NOT NULL,      -- Elo-like scale, see ratings.py
  num_games INTEGER NOT NULL,
  score REAL NOT NULL,       -- wins + ties / 2
  PRIMARY KEY (run_id, hash, character)
);
"""


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
  return conn.execute(
      "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
      (table,)).fetchone() is not None


def _migrate(conn: sqlite3.Connection):
  version = conn.execute('PRAGMA user_version').fetchone()[0]
  if version == SCHEMA_VERSION:
    return
  if version > SCHEMA_VERSION:
    raise RuntimeError(
        f'Database schema version {version} is newer than this code '
        f'({SCHEMA_VERSION}).')
  populated = [
      table for table in VERSIONED_TABLES
      if _table_exists(conn, table)
      and conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]]
  if populated:
    raise RuntimeError(
        f'Database schema version {version} is behind {SCHEMA_VERSION} and '
        f'{populated} hold data; migrate or move the database aside.')
  # Agents and deployed names keep their shape; only eval data is dropped.
  for table in VERSIONED_TABLES:
    conn.execute(f'DROP TABLE IF EXISTS {table}')
  conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')


def connect(path: str) -> sqlite3.Connection:
  """Opens (and creates if needed) the database at `path`."""
  conn = sqlite3.connect(path)
  conn.row_factory = sqlite3.Row
  conn.execute('PRAGMA foreign_keys = ON')
  _migrate(conn)
  conn.executescript(SCHEMA)
  return conn


def now() -> str:
  return datetime.datetime.now(datetime.timezone.utc).isoformat(
      timespec='seconds')


def dumps(obj: tp.Any) -> str:
  """Canonical JSON, so equal params compare equal as strings."""
  return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def loads(s: str) -> tp.Any:
  return json.loads(s)


def active_agents(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
  """Agents currently pointed to by some deployed name, keyed by hash."""
  rows = conn.execute("""
      SELECT DISTINCT agents.* FROM agents
      JOIN deployed_names ON deployed_names.hash = agents.hash
      WHERE deployed_names.active = 1
  """).fetchall()
  return {row['hash']: row for row in rows}


def all_agents(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
  rows = conn.execute('SELECT * FROM agents').fetchall()
  return {row['hash']: row for row in rows}


def names_by_hash(
    conn: sqlite3.Connection, active_only: bool = True,
) -> dict[str, list[str]]:
  query = 'SELECT name, hash FROM deployed_names'
  if active_only:
    query += ' WHERE active = 1'
  query += ' ORDER BY name'
  result: dict[str, list[str]] = {}
  for row in conn.execute(query):
    result.setdefault(row['hash'], []).append(row['name'])
  return result


def display_name(conn: sqlite3.Connection, agent_hash: str) -> str:
  """A human-readable label: the active deployed name(s), else the file."""
  names = names_by_hash(conn).get(agent_hash)
  if not names:
    names = names_by_hash(conn, active_only=False).get(agent_hash)
    if names:
      return '/'.join(names) + ' (retired)'
    row = conn.execute(
        'SELECT stripped_path FROM agents WHERE hash = ?',
        (agent_hash,)).fetchone()
    return row['stripped_path'] if row else agent_hash[:8]
  return '/'.join(names)
