"""Bradley-Terry ratings of players from recorded games.

A player is an (agent, character) pair; see matchups.Player.
"""

import dataclasses
import math
import sqlite3
import typing as tp
import warnings

import numpy as np
import scipy.optimize
import scipy.special

from slippi_ai.eval_db import db
from slippi_ai.eval_db.matchups import Player, tier_of

# Elo convention: a rating gap of 400 is 10:1 odds. Bradley-Terry log-strengths
# are natural-log odds, so multiply by 400 / ln(10).
ELO_PER_NAT = 400 / math.log(10)


@dataclasses.dataclass(frozen=True)
class PairResult:
  p1: Player
  p2: Player
  p1_wins: int
  p2_wins: int
  ties: int


@dataclasses.dataclass
class RatingResult:
  ratings: dict[Player, float]  # Elo-like, mean zero
  num_games: dict[Player, int]
  score: dict[Player, float]    # wins + ties / 2
  num_evals: int
  num_games_total: int


def load_pair_results(
    conn: sqlite3.Connection,
    hashes: tp.Optional[tp.Collection[str]] = None,
) -> tuple[list[PairResult], int]:
  """Per player pair outcomes over every finished eval's games.

  Returns the results and the number of evals they came from. Ports are
  assumed symmetric, so a pair's games are pooled whichever way it was run.
  """
  rows = conn.execute(
      """SELECT e.id AS eval_id, e.p1_hash, e.p2_hash,
                g.p1_character, g.p2_character, g.winner_port, COUNT(*) AS n
         FROM games g JOIN evals e ON g.eval_id = e.id
         WHERE e.status = 'done'
           AND g.p1_character IS NOT NULL AND g.p2_character IS NOT NULL
         GROUP BY 1, 2, 3, 4, 5, 6""").fetchall()
  counts: dict[tuple[Player, Player], list[int]] = {}
  eval_ids = set()
  for row in rows:
    if hashes is not None and (
        row['p1_hash'] not in hashes or row['p2_hash'] not in hashes):
      continue
    eval_ids.add(row['eval_id'])
    a = Player(row['p1_hash'], row['p1_character'])
    b = Player(row['p2_hash'], row['p2_character'])
    a_wins, b_wins, ties = 0, 0, 0
    if row['winner_port'] == 1:
      a_wins = row['n']
    elif row['winner_port'] == 2:
      b_wins = row['n']
    else:
      ties = row['n']
    if b < a:
      a, b = b, a
      a_wins, b_wins = b_wins, a_wins
    tally = counts.setdefault((a, b), [0, 0, 0])
    tally[0] += a_wins
    tally[1] += b_wins
    tally[2] += ties
  results = [
      PairResult(a, b, *tally) for (a, b), tally in sorted(counts.items())]
  return results, len(eval_ids)


def bradley_terry(
    n: int,
    i: np.ndarray,
    j: np.ndarray,
    w_ij: np.ndarray,
    w_ji: np.ndarray,
    alpha: float,
) -> np.ndarray:
  """MAP Bradley-Terry log-strengths from aggregated pairwise outcomes.

  Pair k is players i[k] and j[k]; i[k] beat j[k] w_ij[k] times and lost
  w_ji[k] times (weights, so a tie can count half each way). The prior is an
  isotropic Gaussian with precision `alpha`. Minimizes

    alpha/2 |theta|^2 + sum_k w_ij log(1 + e^-d_k) + w_ji log(1 + e^d_k),

  with d_k = theta[i] - theta[j], by Newton-CG with an exact Hessian-vector
  product. Each step costs O(n + pairs), independent of the number of games.
  """
  def objective(theta):
    d = theta[i] - theta[j]
    value = 0.5 * alpha * (theta @ theta)
    value += w_ij @ np.logaddexp(0, -d) + w_ji @ np.logaddexp(0, d)
    p = scipy.special.expit(d)  # P(i beats j)
    g = w_ji * p - w_ij * (1 - p)  # d value / d d_k
    grad = alpha * theta
    grad += np.bincount(i, weights=g, minlength=n)
    grad -= np.bincount(j, weights=g, minlength=n)
    return value, grad

  def hessp(theta, v):
    p = scipy.special.expit(theta[i] - theta[j])
    h = (w_ij + w_ji) * p * (1 - p) * (v[i] - v[j])
    out = alpha * v
    out += np.bincount(i, weights=h, minlength=n)
    out -= np.bincount(j, weights=h, minlength=n)
    return out

  with warnings.catch_warnings():
    warnings.simplefilter('ignore', scipy.optimize.OptimizeWarning)
    result = scipy.optimize.minimize(
        objective, np.zeros(n), jac=True, hessp=hessp, method='Newton-CG',
        options=dict(xtol=1e-10))
  # With many games the objective is large (~1e5-1e6), so scipy stops on
  # "precision loss" once improvements fall below float64 resolution, i.e. at
  # the optimum. Judge convergence by the per-player Newton step estimate
  # |grad_k| / H_kk instead of result.success.
  theta = result.x
  p = scipy.special.expit(theta[i] - theta[j])
  h = (w_ij + w_ji) * p * (1 - p)
  hess_diag = (
      alpha + np.bincount(i, weights=h, minlength=n)
      + np.bincount(j, weights=h, minlength=n))
  _, grad = objective(theta)
  step = np.abs(grad / hess_diag).max()
  if step > 1e-5:
    raise RuntimeError(
        f'Bradley-Terry fit did not converge (max step {step:.1e} nats): '
        f'{result.message}')
  return theta


def fit(
    results: list[PairResult],
    alpha: float = 0.01,
    num_evals: int = 0,
) -> RatingResult:
  """Fits Bradley-Terry log-strengths and converts to an Elo-like scale.

  Ties count as half a win for each side. `alpha` is the Gaussian prior's
  precision per game: a small positive value keeps undefeated players and
  disconnected clusters finite, and barely moves players with hundreds of
  games.
  """
  players = sorted({p for r in results for p in (r.p1, r.p2)})
  index = {p: k for k, p in enumerate(players)}
  num_games = {p: 0 for p in players}
  score = {p: 0.0 for p in players}
  total = 0
  for r in results:
    n = r.p1_wins + r.p2_wins + r.ties
    total += n
    num_games[r.p1] += n
    num_games[r.p2] += n
    score[r.p1] += r.p1_wins + r.ties / 2
    score[r.p2] += r.p2_wins + r.ties / 2

  if not players:
    return RatingResult({}, {}, {}, num_evals, 0)

  params = bradley_terry(
      len(players),
      i=np.array([index[r.p1] for r in results], dtype=np.intp),
      j=np.array([index[r.p2] for r in results], dtype=np.intp),
      w_ij=np.array([r.p1_wins + r.ties / 2 for r in results]),
      w_ji=np.array([r.p2_wins + r.ties / 2 for r in results]),
      alpha=alpha)
  params -= params.mean()
  ratings = {p: float(params[k] * ELO_PER_NAT) for p, k in index.items()}
  return RatingResult(
      ratings=ratings,
      num_games=num_games,
      score=score,
      num_evals=num_evals,
      num_games_total=total,
  )


def compute_and_store(
    conn: sqlite3.Connection,
    active_only: bool = False,
    alpha: float = 0.01,
) -> tuple[int, RatingResult]:
  """Fits ratings over the database and records them as a rating run."""
  hashes = set(db.active_agents(conn)) if active_only else None
  results, num_evals = load_pair_results(conn, hashes)
  result = fit(results, alpha=alpha, num_evals=num_evals)
  with conn:
    cursor = conn.execute(
        """INSERT INTO rating_runs (computed_at, method, params, num_agents,
             num_evals, num_games)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (db.now(), 'bradley_terry/newton_cg',
         db.dumps(dict(alpha=alpha, active_only=active_only)),
         len(result.ratings), result.num_evals, result.num_games_total))
    run_id = tp.cast(int, cursor.lastrowid)
    conn.executemany(
        """INSERT INTO ratings (run_id, hash, character, rating, num_games, score)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [(run_id, p.hash, p.character, result.ratings[p],
          result.num_games[p], result.score[p])
         for p in result.ratings])
  return run_id, result


def latest_run_id(conn: sqlite3.Connection) -> tp.Optional[int]:
  row = conn.execute('SELECT MAX(id) AS id FROM rating_runs').fetchone()
  return row['id'] if row and row['id'] is not None else None


def leaderboard(
    conn: sqlite3.Connection, run_id: tp.Optional[int] = None,
) -> list[dict[str, tp.Any]]:
  """Ratings of a run (latest by default) with display names, best first."""
  if run_id is None:
    run_id = latest_run_id(conn)
  if run_id is None:
    return []
  agents = db.all_agents(conn)
  names = db.names_by_hash(conn, active_only=False)
  active = db.names_by_hash(conn, active_only=True)
  rows = conn.execute(
      """SELECT hash, character, rating, num_games, score FROM ratings
         WHERE run_id = ? ORDER BY rating DESC""", (run_id,)).fetchall()
  result = []
  for row in rows:
    agent = agents[row['hash']]
    agent_names = names.get(row['hash'], [])
    result.append(dict(
        hash=row['hash'],
        name='/'.join(active.get(row['hash']) or agent_names),
        character=row['character'],
        active=row['hash'] in active,
        stripped_path=agent['stripped_path'],
        agent_type=agent['agent_type'],
        tier=tier_of(agent, agent_names),
        delay=agent['delay'],
        rating=row['rating'],
        num_games=row['num_games'],
        win_rate=row['score'] / row['num_games'] if row['num_games'] else None,
    ))
  return result
