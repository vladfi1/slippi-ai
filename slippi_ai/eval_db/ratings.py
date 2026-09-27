"""Bradley-Terry ratings of players from recorded games, via choix.

A player is an (agent, character) pair; see matchups.Player.
"""

import dataclasses
import math
import sqlite3
import typing as tp

import numpy as np

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


def comparisons(
    results: tp.Iterable[PairResult],
    index: tp.Mapping[Player, int],
) -> list[tuple[int, int]]:
  """Expands aggregated results into choix's (winner, loser) list.

  choix has no tie support and no per-comparison weights, so a decisive game
  becomes two comparisons and a tie becomes one comparison each way: the usual
  half-win-each convention, at twice the count.
  """
  data: list[tuple[int, int]] = []
  for r in results:
    a, b = index[r.p1], index[r.p2]
    data.extend([(a, b)] * (2 * r.p1_wins))
    data.extend([(b, a)] * (2 * r.p2_wins))
    data.extend([(a, b)] * r.ties)
    data.extend([(b, a)] * r.ties)
  return data


def fit(
    results: list[PairResult],
    alpha: float = 0.01,
    method: str = 'opt',
    num_evals: int = 0,
) -> RatingResult:
  """Fits Bradley-Terry log-strengths and converts to an Elo-like scale.

  `alpha` is choix's Gaussian prior strength. A small positive value keeps
  undefeated players and disconnected clusters finite; it barely moves players
  with hundreds of games.
  """
  import choix  # pylint: disable=import-outside-toplevel

  players = sorted({p for r in results for p in (r.p1, r.p2)})
  index = {p: i for i, p in enumerate(players)}
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

  data = comparisons(results, index)
  if method == 'opt':
    params = choix.opt_pairwise(len(players), data, alpha=alpha)
  elif method == 'mm':
    params = choix.mm_pairwise(len(players), data, alpha=alpha)
  elif method == 'ilsr':
    params = choix.ilsr_pairwise(len(players), data, alpha=alpha)
  else:
    raise ValueError(f'Unknown method {method!r}')
  params = np.asarray(params, dtype=np.float64)
  params -= params.mean()
  ratings = {p: float(params[i] * ELO_PER_NAT) for p, i in index.items()}
  return RatingResult(
      ratings=ratings,
      num_games=num_games,
      score=score,
      num_evals=num_evals,
      num_games_total=total,
  )


def compute_and_store(
    conn: sqlite3.Connection,
    active_only: bool = True,
    alpha: float = 0.01,
    method: str = 'opt',
) -> tuple[int, RatingResult]:
  """Fits ratings over the database and records them as a rating run."""
  hashes = set(db.active_agents(conn)) if active_only else None
  results, num_evals = load_pair_results(conn, hashes)
  result = fit(results, alpha=alpha, method=method, num_evals=num_evals)
  with conn:
    cursor = conn.execute(
        """INSERT INTO rating_runs (computed_at, method, params, num_agents,
             num_evals, num_games)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (db.now(), f'bradley_terry/{method}',
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
