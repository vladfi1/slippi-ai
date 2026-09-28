"""Running one matchup through slippi_ai.evaluation and storing the outcome."""

import dataclasses
import json
import logging
import multiprocessing
import multiprocessing.connection
import os
import sqlite3
import subprocess
import sys
import traceback
import typing as tp

import melee

from slippi_ai import data, dolphin, evaluation, paths
from slippi_ai.eval_db import db
from slippi_ai.eval_db.matchups import Matchup

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(paths.__file__)))
EVALUATOR_SCRIPT = os.path.join(REPO_ROOT, 'scripts', 'run_evaluator.py')
EVALUATOR_SHELL_SCRIPT = os.path.join(
    REPO_ROOT, 'run_scripts', 'run_evaluator.sh')

# Mirrors run_scripts/run_evaluator.sh; tests check they stay in sync.
DEFAULT_CONFIG = evaluation.EvaluationConfig(
    num_envs=1024,
    rollout_length=18000,  # 5 minutes at 60 fps; floaty matchups run long
    chunk_length=120,
    num_env_steps=4,
    num_agent_steps=4,
    sim_envs=True,
    async_envs=True,
    use_gpu=True,
    quiet=True,
)
DEFAULT_STAGE = melee.Stage.RANDOM_STAGE


# Params that change what an eval measures only mildly, so an eval run with a
# different value still counts as covering its player pairs. A longer rollout
# just lets more (slower) games finish.
NON_DISQUALIFYING_PARAMS = ('rollout_length',)


def params_key(params: dict[str, tp.Any]) -> str:
  """The coverage key of an eval's params dict (see EvalParams.key)."""
  return db.dumps({
      k: v for k, v in params.items() if k not in NON_DISQUALIFYING_PARAMS})


def rekey_evals(conn: sqlite3.Connection) -> int:
  """Recomputes stored params_keys after the key rule changes.

  Returns the number of evals whose key changed.
  """
  updates = []
  for row in conn.execute('SELECT id, params, params_key FROM evals'):
    key = params_key(db.loads(row['params']))
    if key != row['params_key']:
      updates.append((key, row['id']))
  with conn:
    conn.executemany('UPDATE evals SET params_key = ? WHERE id = ?', updates)
  return len(updates)


@dataclasses.dataclass
class EvalParams:
  """Everything that determines what games an eval plays."""
  config: evaluation.EvaluationConfig = dataclasses.field(
      default_factory=lambda: dataclasses.replace(DEFAULT_CONFIG))
  stage: melee.Stage = DEFAULT_STAGE

  def to_dict(self) -> dict[str, tp.Any]:
    return dict(self.config.result_affecting(), stage=self.stage.name)

  def key(self) -> str:
    """Canonical string identifying these settings in the evals table."""
    return params_key(self.to_dict())

  def dolphin_kwargs(self) -> dict:
    return dolphin.DolphinConfig(
        infinite_time=False, headless=True, stage=self.stage).to_kwargs()

  def to_flags(self) -> list[str]:
    """Flags for scripts/run_evaluator.py reproducing this config."""
    c = self.config
    flags = []
    for field in dataclasses.fields(c):
      value = getattr(c, field.name)
      if value is None:
        continue
      if isinstance(value, bool):
        flags.append(f'--{"" if value else "no"}{field.name}')
      else:
        flags.append(f'--{field.name}={value}')
    flags.append(f'--dolphin.stage={self.stage.name}')
    return flags


def parse_shell_script_flags(path: str) -> dict[str, str]:
  """Reads `--flag value`, `--flag=value` and bare `--flag` from a .sh file.

  Used by tests to check that DEFAULT_CONFIG tracks the shell script.
  """
  with open(path) as f:
    text = f.read().replace('\\\n', ' ')
  tokens = [t for line in text.splitlines() for t in line.split()]
  result: dict[str, str] = {}
  i = 0
  while i < len(tokens):
    token = tokens[i]
    if token.startswith('--'):
      body = token[2:]
      if '=' in body:
        name, value = body.split('=', 1)
        result[name] = value
      elif (i + 1 < len(tokens) and not tokens[i + 1].startswith('--')
            and not tokens[i + 1].startswith('"')):
        result[body] = tokens[i + 1]
        i += 1
      else:
        result[body] = 'true'
    i += 1
  return result


def git_commit(cwd: str = REPO_ROOT) -> tp.Optional[str]:
  try:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=cwd, text=True,
        stderr=subprocess.DEVNULL).strip()
  except (subprocess.CalledProcessError, OSError):
    return None


def characters_of(names: tp.Iterable[str]) -> list[melee.Character]:
  return [data.name_to_character[name.lower()] for name in names]


def summarize_games(games: list[dict]) -> dict[str, tp.Any]:
  return dict(
      num_games=len(games),
      p1_wins=sum(g['winner_port'] == 1 for g in games),
      p2_wins=sum(g['winner_port'] == 2 for g in games),
      ties=sum(g['winner_port'] is None for g in games),
      timeouts=sum(bool(g['max_frame_reached']) for g in games),
  )


def insert_games(conn: sqlite3.Connection, eval_id: int, games: list[dict]):
  rows = []
  for g in games:
    # Ratings are per (agent, character), so a game without its lane's
    # characters is useless; evaluation.evaluate annotates every record.
    characters = g.get('characters')
    if not characters or len(characters) != 2 or None in characters:
      raise ValueError(
          f'Game {g.get("env_id")}/{g.get("episode_id")} has no characters.')
    rows.append((
        eval_id, g['env_id'], g['episode_id'], g['stage'], g['frames'],
        g['winner_port'], characters[0], characters[1],
        g['stocks'][0], g['stocks'][1], g['percents'][0], g['percents'][1],
        int(g['stockout']), int(g['max_frame_reached']),
    ))
  conn.executemany(
      """INSERT INTO games (eval_id, env_id, episode_id, stage, frames,
           winner_port, p1_character, p2_character, p1_stocks, p2_stocks,
           p1_percent, p2_percent, stockout, max_frame_reached)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
      rows)


def start_eval(
    conn: sqlite3.Connection,
    matchup: Matchup,
    params: EvalParams,
    log_path: tp.Optional[str] = None,
) -> int:
  with conn:
    cursor = conn.execute(
        """INSERT INTO evals (p1_hash, p2_hash, character_pairs,
             params, params_key, status, started_at, git_commit, log_path)
           VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)""",
        (matchup.p1_hash, matchup.p2_hash,
         db.dumps([list(pair) for pair in matchup.character_pairs]),
         db.dumps(params.to_dict()), params.key(),
         db.now(), git_commit(), log_path))
  return tp.cast(int, cursor.lastrowid)


def finish_eval(
    conn: sqlite3.Connection,
    eval_id: int,
    games: list[dict],
    kdpm: tp.Optional[float] = None,
    env_fps: tp.Optional[float] = None,
):
  """Records a successful run: summary columns plus one row per game.

  A run that produced no games (e.g. a rollout shorter than a game) is
  recorded as failed so its player pairs stay pending.
  """
  if not games:
    fail_eval(conn, eval_id, 'no completed games')
    return
  summary = summarize_games(games)
  with conn:
    conn.execute(
        """UPDATE evals SET status = 'done', finished_at = ?, num_games = ?,
             p1_wins = ?, p2_wins = ?, ties = ?, timeouts = ?, p1_kdpm = ?,
             env_fps = ?
           WHERE id = ?""",
        (db.now(), summary['num_games'], summary['p1_wins'],
         summary['p2_wins'], summary['ties'], summary['timeouts'],
         kdpm, env_fps, eval_id))
    insert_games(conn, eval_id, games)


def fail_eval(conn: sqlite3.Connection, eval_id: int, error: str):
  with conn:
    conn.execute(
        """UPDATE evals SET status = 'failed', finished_at = ?, error = ?
           WHERE id = ?""",
        (db.now(), error[-4000:], eval_id))


def agent_paths(
    conn: sqlite3.Connection, matchup: Matchup, stripped_dir: str,
) -> tuple[str, str]:
  agents = db.all_agents(conn)
  return tuple(  # type: ignore[return-value]
      os.path.join(stripped_dir, agents[h]['stripped_path'])
      for h in (matchup.p1_hash, matchup.p2_hash))


def evaluate_matchup(
    matchup: Matchup,
    params: EvalParams,
    p1_path: str,
    p2_path: str,
) -> evaluation.EvaluationResult:
  """Runs the evaluation for a matchup in this process."""
  pairs = [
      tuple(characters_of(pair)) for pair in matchup.character_pairs]
  return evaluation.evaluate(
      config=params.config,
      player_kwargs={
          1: evaluation.default_player_kwargs(p1_path, pairs[0][0]),
          2: evaluation.default_player_kwargs(p2_path, pairs[0][1]),
      },
      dolphin_kwargs=params.dolphin_kwargs(),
      character_pairs=pairs,
  )


@dataclasses.dataclass
class EvalOutcome:
  """What `finish_eval` needs from an evaluation; small enough to pickle."""
  games: list[dict]
  kdpm: tp.Optional[float] = None
  env_fps: tp.Optional[float] = None

  @classmethod
  def from_result(cls, result: evaluation.EvaluationResult) -> 'EvalOutcome':
    return cls(
        games=result.completed_games, kdpm=result.kdpm, env_fps=result.env_fps)


EvaluateFn = tp.Callable[
    [Matchup, EvalParams, str, str], evaluation.EvaluationResult]


class ChildEvalError(Exception):
  """The eval child process failed; str(e) is its traceback or exit status."""


def _child_main(
    pipe: multiprocessing.connection.Connection,
    evaluate_fn: EvaluateFn,
    matchup: Matchup,
    params: EvalParams,
    p1_path: str,
    p2_path: str,
    log_level: int,
):
  """Entry point of the per-eval child: evaluates and reports through `pipe`."""
  root = logging.getLogger()
  if not root.handlers:
    logging.basicConfig()
  root.setLevel(log_level)
  try:
    outcome = EvalOutcome.from_result(
        evaluate_fn(matchup, params, p1_path, p2_path))
    message = ('ok', outcome)
  except BaseException:  # pylint: disable=broad-except
    message = ('error', traceback.format_exc())
  try:
    pipe.send(message)
  finally:
    pipe.close()


def evaluate_matchup_in_child(
    matchup: Matchup,
    params: EvalParams,
    p1_path: str,
    p2_path: str,
    evaluate_fn: EvaluateFn = evaluate_matchup,
) -> EvalOutcome:
  """Runs `evaluate_fn` in a fresh (spawned) process and returns its outcome.

  Building agents and compiling their policies leaks a few hundred MB per
  eval into the process that does it (TF and JAX keep compiled functions in
  process-wide caches), so a long session runs each eval in a child that
  exits when it's done. Spawning also keeps the parent free of GPU state.

  Raises ChildEvalError if the child raised or died without reporting.
  """
  ctx = multiprocessing.get_context('spawn')
  parent_end, child_end = ctx.Pipe(duplex=False)
  # Not a daemon: the child spawns the sim env workers itself.
  process = ctx.Process(
      target=_child_main,
      args=(child_end, evaluate_fn, matchup, params, p1_path, p2_path,
            logging.getLogger().level),
      name='eval',
  )
  process.start()
  child_end.close()  # so recv() sees EOF if the child dies
  try:
    # Receive before joining: a large game list would otherwise block the
    # child's send while we block on join.
    try:
      message = parent_end.recv()
    except EOFError:
      message = None
    process.join()
    exitcode = process.exitcode
  except BaseException:
    # Interrupted (e.g. Ctrl-C, which the child also got): don't leave it.
    process.join(timeout=10)
    if process.is_alive():
      process.terminate()
      process.join()
    raise
  finally:
    parent_end.close()
    process.close()

  if message is None:
    raise ChildEvalError(
        f'eval child exited with code {exitcode} without reporting')
  status, payload = message
  if status != 'ok':
    raise ChildEvalError(payload)
  return payload


def run_eval(
    conn: sqlite3.Connection,
    matchup: Matchup,
    params: EvalParams,
    stripped_dir: str,
    isolate: bool = True,
    evaluate_fn: EvaluateFn = evaluate_matchup,
) -> int:
  """Runs one matchup and stores the outcome. Returns the eval id.

  With `isolate`, the evaluation runs in a spawned child process (see
  `evaluate_matchup_in_child`); otherwise in this process, which leaks memory
  across evals.

  The eval is marked 'running' before it starts, so a crash that takes the
  process down leaves a visible record (see `reset_stale_running`).
  """
  p1_path, p2_path = agent_paths(conn, matchup, stripped_dir)
  eval_id = start_eval(conn, matchup, params)
  logging.info(
      'eval %d: %s vs %s over %d character pairs', eval_id,
      p1_path, p2_path, len(matchup.character_pairs))
  try:
    if isolate:
      outcome = evaluate_matchup_in_child(
          matchup, params, p1_path, p2_path, evaluate_fn=evaluate_fn)
    else:
      outcome = EvalOutcome.from_result(
          evaluate_fn(matchup, params, p1_path, p2_path))
    finish_eval(
        conn, eval_id, outcome.games,
        kdpm=outcome.kdpm, env_fps=outcome.env_fps)
  except ChildEvalError as e:
    error = str(e)
    logging.error('eval %d failed in its child process:\n%s', eval_id, error)
    fail_eval(conn, eval_id, error)
  except Exception:  # pylint: disable=broad-except
    error = traceback.format_exc()
    logging.error('eval %d failed:\n%s', eval_id, error)
    fail_eval(conn, eval_id, error)
  return eval_id


def evaluator_command(
    matchup: Matchup,
    params: EvalParams,
    p1_path: str,
    p2_path: str,
    results_path: str,
    python: str = sys.executable,
) -> list[str]:
  # The script only takes per-port lists (a full product), so a subprocess
  # eval can carry exactly one character pair.
  if len(matchup.character_pairs) != 1:
    raise ValueError(
        'Subprocess evals support one character pair per run; got '
        f'{len(matchup.character_pairs)}. Use --min_lanes_per_pair >= num_envs.')
  (p1_char, p2_char), = matchup.character_pairs
  return [
      python, EVALUATOR_SCRIPT,
      *params.to_flags(),
      f'--player.ai.path={p1_path}',
      f'--opponent.ai.path={p2_path}',
      f'--player.character={p1_char}',
      f'--opponent.character={p2_char}',
      f'--results_path={results_path}',
  ]


def _tail(path: str, lines: int = 30) -> str:
  if not os.path.exists(path):
    return ''
  with open(path, errors='replace') as f:
    return ''.join(f.readlines()[-lines:])


def run_eval_subprocess(
    conn: sqlite3.Connection,
    matchup: Matchup,
    params: EvalParams,
    stripped_dir: str,
    log_dir: str,
    python: str = sys.executable,
) -> int:
  """Like `run_eval` but through scripts/run_evaluator.py, with its own log.

  Only supports one character pair per eval; `run_eval` (which isolates in a
  spawned child by default) is the usual choice.
  """
  p1_path, p2_path = agent_paths(conn, matchup, stripped_dir)
  os.makedirs(log_dir, exist_ok=True)
  eval_id = start_eval(conn, matchup, params)
  log_path = os.path.join(log_dir, f'eval_{eval_id}.log')
  results_path = os.path.join(log_dir, f'eval_{eval_id}.json')
  with conn:
    conn.execute(
        'UPDATE evals SET log_path = ? WHERE id = ?', (log_path, eval_id))

  command = evaluator_command(
      matchup, params, p1_path, p2_path, results_path, python=python)
  logging.info('eval %d: %s', eval_id, ' '.join(command))
  try:
    with open(log_path, 'w') as log:
      log.write(' '.join(command) + '\n')
      log.flush()
      subprocess.run(
          command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
          check=True)
    with open(results_path) as f:
      results = json.load(f)
  except (subprocess.CalledProcessError, OSError, json.JSONDecodeError) as e:
    fail_eval(conn, eval_id, f'{e}\n{_tail(log_path)}')
    logging.error('eval %d failed: %s', eval_id, e)
    return eval_id

  finish_eval(
      conn, eval_id, results['completed_games'],
      kdpm=results.get('kdpm'), env_fps=results.get('env_fps'))
  return eval_id


def reset_stale_running(conn: sqlite3.Connection) -> int:
  """Marks evals left in 'running' by a dead session as failed.

  Only call this when no other eval_db process is running.
  """
  with conn:
    cursor = conn.execute(
        """UPDATE evals SET status = 'failed', finished_at = ?,
             error = 'left running by a previous session'
           WHERE status = 'running'""",
        (db.now(),))
  return cursor.rowcount
