"""Scanning deployed_models and registering agents in the database."""

import dataclasses
import hashlib
import logging
import os
import sqlite3
import typing as tp

from slippi_ai import eval_lib, saving
from slippi_ai.eval_db import db


@dataclasses.dataclass(frozen=True)
class DeployedFile:
  name: str           # entry name in deployed_models
  path: str           # resolved path of the model file
  stripped_path: str  # path relative to stripped_models, or absolute if outside
  mtime: float
  size: int


@dataclasses.dataclass
class SyncReport:
  added: list[str] = dataclasses.field(default_factory=list)     # new hashes
  updated: list[str] = dataclasses.field(default_factory=list)   # name -> new hash
  renamed: list[str] = dataclasses.field(default_factory=list)   # known hash, new name
  removed: list[str] = dataclasses.field(default_factory=list)   # names gone
  unchanged: int = 0
  skipped: list[str] = dataclasses.field(default_factory=list)   # unreadable

  def summary(self) -> str:
    return (
        f'added={len(self.added)} updated={len(self.updated)} '
        f'renamed={len(self.renamed)} removed={len(self.removed)} '
        f'unchanged={self.unchanged} skipped={len(self.skipped)}')


def md5_file(path: str, chunk_size: int = 1 << 24) -> str:
  digest = hashlib.md5()
  with open(path, 'rb') as f:
    while chunk := f.read(chunk_size):
      digest.update(chunk)
  return digest.hexdigest()


def scan_deployed(deployed_dir: str, stripped_dir: str) -> list[DeployedFile]:
  """Lists the model files behind every entry of deployed_dir."""
  stripped_dir = os.path.realpath(stripped_dir)
  files = []
  for name in sorted(os.listdir(deployed_dir)):
    link = os.path.join(deployed_dir, name)
    path = os.path.realpath(link)
    if not os.path.isfile(path):
      logging.warning('Skipping %s: not a file (%s)', name, path)
      continue
    if path.startswith(stripped_dir + os.sep):
      stripped_path = os.path.relpath(path, stripped_dir)
    else:
      logging.warning('%s does not point into %s', name, stripped_dir)
      stripped_path = path
    stat = os.stat(path)
    files.append(DeployedFile(
        name=name,
        path=path,
        stripped_path=stripped_path,
        mtime=stat.st_mtime,
        size=stat.st_size,
    ))
  return files


def agent_metadata(state: dict) -> dict[str, tp.Any]:
  """Extracts the columns of `agents` that come from the pickled state."""
  summary = eval_lib.AgentSummary.from_state(state)
  agent_config = eval_lib.get_agent_config(state)
  rl_names = None
  rl_rating = None
  if agent_config is not None:
    name = agent_config.get('name')
    if isinstance(name, str):
      rl_names = [name]
    elif name is not None:
      # Names may repeat per character; keep them unique but ordered.
      rl_names = list(dict.fromkeys(name))
    rl_rating = agent_config.get('rating')
  return dict(
      platform=state['config'].get(saving.PLATFORM_KEY),
      agent_type=summary.type.name,
      delay=summary.delay,
      characters=db.dumps([c.name for c in summary.characters]),
      opponents=db.dumps([c.name for c in summary.opponents]),
      rl_names=db.dumps(rl_names) if rl_names is not None else None,
      rl_rating=rl_rating,
  )


def _find_by_file(
    conn: sqlite3.Connection, file: DeployedFile) -> tp.Optional[str]:
  """Returns the hash of an agent row matching the file's path, mtime and size.

  This is how we avoid rehashing hundreds of large files on every sync: the
  spec is that a changed mtime means a changed agent.
  """
  row = conn.execute(
      'SELECT hash FROM agents WHERE stripped_path = ? AND mtime = ? AND size = ?',
      (file.stripped_path, file.mtime, file.size)).fetchone()
  return row['hash'] if row else None


def register_file(
    conn: sqlite3.Connection,
    file: DeployedFile,
    now: str,
    report: SyncReport,
) -> tp.Optional[str]:
  """Makes sure the agent behind `file` is in `agents`; returns its hash."""
  agent_hash = _find_by_file(conn, file)
  if agent_hash is None:
    agent_hash = md5_file(file.path)
    row = conn.execute(
        'SELECT hash FROM agents WHERE hash = ?', (agent_hash,)).fetchone()
    if row is None:
      try:
        state = saving.load_state_from_disk(file.path)
        metadata = agent_metadata(state)
      except Exception as e:  # pylint: disable=broad-except
        logging.exception('Could not read agent %s (%s)', file.name, file.path)
        report.skipped.append(f'{file.name}: {e}')
        return None
      conn.execute(
          """INSERT INTO agents (hash, stripped_path, mtime, size, platform,
               agent_type, delay, characters, opponents, rl_names, rl_rating,
               first_seen, last_seen)
             VALUES (:hash, :stripped_path, :mtime, :size, :platform,
               :agent_type, :delay, :characters, :opponents, :rl_names,
               :rl_rating, :now, :now)""",
          dict(
              hash=agent_hash,
              stripped_path=file.stripped_path,
              mtime=file.mtime,
              size=file.size,
              now=now,
              **metadata,
          ))
      report.added.append(agent_hash)
    else:
      # Same bytes at a new location or with a touched mtime.
      conn.execute(
          'UPDATE agents SET stripped_path = ?, mtime = ?, size = ? WHERE hash = ?',
          (file.stripped_path, file.mtime, file.size, agent_hash))
  conn.execute(
      'UPDATE agents SET last_seen = ? WHERE hash = ?', (now, agent_hash))
  return agent_hash


def sync(
    conn: sqlite3.Connection,
    deployed_dir: str,
    stripped_dir: str,
    now: tp.Optional[str] = None,
) -> SyncReport:
  """Brings `agents` and `deployed_names` in line with deployed_dir."""
  now = now or db.now()
  report = SyncReport()
  files = scan_deployed(deployed_dir, stripped_dir)

  previous = {
      row['name']: row['hash']
      for row in conn.execute(
          'SELECT name, hash FROM deployed_names WHERE active = 1')
  }

  seen_names = set()
  with conn:
    for file in files:
      agent_hash = register_file(conn, file, now, report)
      if agent_hash is None:
        continue
      seen_names.add(file.name)

      previous_hash = previous.get(file.name)
      if previous_hash == agent_hash:
        report.unchanged += 1
      elif previous_hash is None:
        if agent_hash not in report.added:
          report.renamed.append(f'{file.name} -> {agent_hash[:8]}')
      else:
        report.updated.append(
            f'{file.name}: {previous_hash[:8]} -> {agent_hash[:8]}')
        conn.execute(
            'UPDATE deployed_names SET active = 0 WHERE name = ? AND hash = ?',
            (file.name, previous_hash))

      conn.execute(
          """INSERT INTO deployed_names (name, hash, first_seen, last_seen, active)
             VALUES (?, ?, ?, ?, 1)
             ON CONFLICT (name, hash) DO UPDATE SET last_seen = excluded.last_seen,
               active = 1""",
          (file.name, agent_hash, now, now))

    for name in previous:
      if name not in seen_names:
        report.removed.append(name)
        conn.execute(
            'UPDATE deployed_names SET active = 0 WHERE name = ?', (name,))

  return report
