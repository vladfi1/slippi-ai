"""Deciding which pairs of players still need an evaluation.

A player is an (agent, character) pair: a multi-character agent like `gm` is
one player per character it plays. Ports are assumed symmetric, so player
pairs are unordered and evaluated once. Evals are batched per agent pair: one
run between two agents cycles its lanes over all their pending character
pairs, and the recorded per-game characters attribute results to players.
"""

import collections
import dataclasses
import sqlite3
import typing as tp

import melee
import melee_sim

from slippi_ai.eval_db import db

# Same derivation as slippi_ai.sim_env.SUPPORTED_CHARACTERS, done here so the
# planner doesn't import the env stack (and JAX) just to plan.
SIM_CHARACTERS = frozenset(
    melee.Character(int(c.value)).name for c in melee_sim.Character)

# Ranked ladder tiers, weakest first, as they appear in deployed names
# (e.g. 'gold-v2', 'falco_d21_vs_ics_plat', 'super-gm'). Medium is folded into
# plat, single-character RL agents count as super-gm, imitation as bronze.
TIERS = ('bronze', 'silver', 'gold', 'plat', 'diamond', 'master', 'gm', 'super-gm')
TIER_ALIASES = {'medium': 'plat'}
WEAK_TIERS = frozenset(TIERS[:TIERS.index('master')])
MASTER_RATING = 2200.0


def _name_parts(name: str) -> list[str]:
  return name.lower().replace('-', '_').split('_')


def tier_from_names(names: tp.Iterable[str]) -> tp.Optional[str]:
  """The ladder tier a set of deployed names implies, or None.

  If names disagree (an alias in another tier), the strongest wins.
  """
  found = set()
  for name in names:
    parts = _name_parts(name)
    if 'super' in parts and 'gm' in parts:
      found.add('super-gm')
      continue
    for part in parts:
      part = TIER_ALIASES.get(part, part)
      if part in TIERS:
        found.add(part)
  if not found:
    return None
  return max(found, key=TIERS.index)


def is_multichar(agent: sqlite3.Row) -> bool:
  return len(db.loads(agent['characters'])) > 1


def tier_of(agent: sqlite3.Row, names: tp.Sequence[str]) -> tp.Optional[str]:
  """Ladder tier for matchmaking.

  Imitation agents are bronze. Single-character RL agents all count as
  super-gm, even the few named after a rating tier (e.g.
  'falcon_d21_vs_ics_master'). Multi-character RL agents are tiered by name.
  """
  if agent['agent_type'] != 'RL':
    return 'bronze'
  if not is_multichar(agent):
    return 'super-gm'
  return tier_from_names(names)


def tier_gap(a: tp.Optional[str], b: tp.Optional[str]) -> tp.Optional[int]:
  """Distance between tiers; None when either side is untiered."""
  if a is None or b is None:
    return None
  return abs(TIERS.index(a) - TIERS.index(b))


def is_weak(agent: sqlite3.Row, names: tp.Sequence[str]) -> bool:
  """Imitation agents, or RL agents below master by name or training rating."""
  if agent['agent_type'] != 'RL':
    return True
  rating = agent['rl_rating']
  if rating is not None and rating < MASTER_RATING:
    return True
  return tier_from_names(names) in WEAK_TIERS


def is_ladder(agent: sqlite3.Row, names: tp.Sequence[str]) -> bool:
  """A ranked multi-character agent."""
  return is_multichar(agent) and tier_from_names(names) is not None


POOLS = ('all', 'strong', 'ladder', 'strong-ladder')


def select_pool(
    conn: sqlite3.Connection,
    agents: tp.Mapping[str, sqlite3.Row],
    pool: str = 'strong',
) -> dict[str, sqlite3.Row]:
  """Restricts agents to a pool, judged by every name they were deployed as.

  'all' keeps everything; 'strong' drops weak agents (see `is_weak`);
  'ladder' keeps only the ranked multi-character agents, weak tiers included;
  'strong-ladder' keeps the ranked multi-character agents from master up.
  """
  if pool not in POOLS:
    raise ValueError(f'Unknown pool {pool!r}; expected one of {POOLS}')
  if pool == 'all':
    return dict(agents)
  names = db.names_by_hash(conn, active_only=False)
  result = {}
  for h, agent in agents.items():
    agent_names = names.get(h, [])
    weak = is_weak(agent, agent_names)
    ladder = is_ladder(agent, agent_names)
    keep = {
        'strong': not weak,
        'ladder': ladder,
        'strong-ladder': ladder and not weak,
    }[pool]
    if keep:
      result[h] = agent
  return result


def strong_agents(
    conn: sqlite3.Connection,
    agents: tp.Mapping[str, sqlite3.Row],
) -> dict[str, sqlite3.Row]:
  return select_pool(conn, agents, 'strong')


def agent_tiers(
    conn: sqlite3.Connection,
    agents: tp.Mapping[str, sqlite3.Row],
) -> dict[str, tp.Optional[str]]:
  names = db.names_by_hash(conn, active_only=False)
  return {h: tier_of(agent, names.get(h, [])) for h, agent in agents.items()}


@dataclasses.dataclass(frozen=True, order=True)
class Player:
  hash: str
  character: str  # libmelee name


def agent_players(agent: sqlite3.Row) -> list[Player]:
  return [
      Player(agent['hash'], c)
      for c in db.loads(agent['characters']) if c in SIM_CHARACTERS]


# An unordered player pair, stored with the smaller (hash, character) first.
PlayerPair = tuple[Player, Player]


def canonical_pair(a: Player, b: Player) -> PlayerPair:
  return (a, b) if a <= b else (b, a)


@dataclasses.dataclass(frozen=True)
class MatchupFilters:
  """Optional restrictions on which player pairs get planned.

  max_tier_gap: largest allowed ladder-tier distance between the two agents;
    None disables the check. Untiered agents are never restricted.
  allowed_chars: characters players may use; None means any.
  allowed_opponents: if set, one player must use a character from
    allowed_chars and the other one from allowed_opponents, so e.g.
    allowed_chars={FOX}, allowed_opponents={FALCO} targets Fox vs Falco only.
  """
  max_tier_gap: tp.Optional[int] = 1
  allowed_chars: tp.Optional[frozenset[str]] = None
  allowed_opponents: tp.Optional[frozenset[str]] = None

  def characters_ok(self, a: str, b: str) -> bool:
    chars = self.allowed_chars
    opponents = self.allowed_opponents
    if opponents is None:
      opponents = chars

    def in_set(c: str, allowed: tp.Optional[frozenset[str]]) -> bool:
      return allowed is None or c in allowed

    return ((in_set(a, chars) and in_set(b, opponents))
            or (in_set(a, opponents) and in_set(b, chars)))


def relevant_pairs(
    agents: tp.Mapping[str, sqlite3.Row],
    filters: MatchupFilters = MatchupFilters(),
    tiers: tp.Optional[tp.Mapping[str, tp.Optional[str]]] = None,
) -> list[PlayerPair]:
  """Every unordered player pair worth evaluating among `agents`.

  A pair is relevant when the agents differ, each was trained to face the
  other's character, and the pair passes `filters`.
  """
  opponents = {h: set(db.loads(a['opponents'])) for h, a in agents.items()}
  players = {h: agent_players(a) for h, a in agents.items()}
  hashes = sorted(agents)
  result = []
  for i, h1 in enumerate(hashes):
    for h2 in hashes[i + 1:]:
      if filters.max_tier_gap is not None and tiers is not None:
        gap = tier_gap(tiers.get(h1), tiers.get(h2))
        if gap is not None and gap > filters.max_tier_gap:
          continue
      for p in players[h1]:
        if p.character not in opponents[h2]:
          continue
        for q in players[h2]:
          if q.character not in opponents[h1]:
            continue
          if not filters.characters_ok(p.character, q.character):
            continue
          result.append(canonical_pair(p, q))
  return result


@dataclasses.dataclass(frozen=True)
class Matchup:
  """One evaluation run: two agents and the character pairs to cycle over.

  p1_hash <= p2_hash so the same agent pair always gets the same ports.
  """
  p1_hash: str
  p2_hash: str
  character_pairs: tuple[tuple[str, str], ...]  # (p1 character, p2 character)

  @property
  def pair(self) -> tuple[str, str]:
    return (self.p1_hash, self.p2_hash)

  @property
  def player_pairs(self) -> list[PlayerPair]:
    return [
        canonical_pair(Player(self.p1_hash, c1), Player(self.p2_hash, c2))
        for c1, c2 in self.character_pairs]

  @classmethod
  def from_player_pairs(cls, pairs: tp.Sequence[PlayerPair]) -> 'Matchup':
    hashes = {p.hash for pair in pairs for p in pair}
    if len(hashes) != 2:
      raise ValueError(f'A matchup needs exactly two agents, got {hashes}')
    p1_hash, p2_hash = sorted(hashes)
    character_pairs = []
    for a, b in pairs:
      if a.hash != p1_hash:
        a, b = b, a
      character_pairs.append((a.character, b.character))
    return cls(p1_hash, p2_hash, tuple(character_pairs))


def covered_pairs(
    conn: sqlite3.Connection,
    params_key: str,
    include_running: bool = True,
) -> set[PlayerPair]:
  """Player pairs already covered by a finished (or in-flight) eval."""
  statuses = ('done', 'running') if include_running else ('done',)
  rows = conn.execute(
      f"""SELECT p1_hash, p2_hash, character_pairs FROM evals
          WHERE params_key = ? AND status IN ({','.join('?' * len(statuses))})""",
      (params_key, *statuses)).fetchall()
  covered = set()
  for row in rows:
    for c1, c2 in db.loads(row['character_pairs']):
      covered.add(canonical_pair(
          Player(row['p1_hash'], c1), Player(row['p2_hash'], c2)))
  return covered


def eval_counts(conn: sqlite3.Connection) -> dict[str, int]:
  """Number of finished evals each agent has taken part in."""
  counts: dict[str, int] = {}
  for row in conn.execute(
      "SELECT p1_hash, p2_hash FROM evals WHERE status = 'done'"):
    for h in (row['p1_hash'], row['p2_hash']):
      counts[h] = counts.get(h, 0) + 1
  return counts


def batch_pairs(
    pairs: tp.Iterable[PlayerPair],
    max_pairs_per_eval: int,
) -> list[Matchup]:
  """Groups player pairs by agent pair, at most `max_pairs_per_eval` each."""
  by_agents: dict[tuple[str, str], list[PlayerPair]] = collections.defaultdict(list)
  for pair in pairs:
    key = tuple(sorted((pair[0].hash, pair[1].hash)))
    by_agents[key].append(pair)  # type: ignore[index]
  matchups = []
  for key in sorted(by_agents):
    group = by_agents[key]
    for start in range(0, len(group), max_pairs_per_eval):
      matchups.append(
          Matchup.from_player_pairs(group[start:start + max_pairs_per_eval]))
  return matchups


def pending_matchups(
    conn: sqlite3.Connection,
    params_key: str,
    agents: tp.Optional[tp.Mapping[str, sqlite3.Row]] = None,
    pool: str = 'strong',
    filters: MatchupFilters = MatchupFilters(),
    max_pairs_per_eval: int = 128,
) -> list[Matchup]:
  """Evals that would cover every relevant, not-yet-covered player pair.

  Sorted so that agents with the fewest finished evals come first: a newly
  added or updated agent gets its whole slate before we top up old ones.
  """
  if agents is None:
    agents = db.active_agents(conn)
  agents = select_pool(conn, agents, pool)
  tiers = agent_tiers(conn, agents)
  covered = covered_pairs(conn, params_key)
  pending = [
      pair for pair in relevant_pairs(agents, filters, tiers)
      if pair not in covered]
  matchups = batch_pairs(pending, max_pairs_per_eval)
  counts = eval_counts(conn)
  matchups.sort(key=lambda m: (
      min(counts.get(m.p1_hash, 0), counts.get(m.p2_hash, 0)),
      counts.get(m.p1_hash, 0) + counts.get(m.p2_hash, 0),
      m.pair,
  ))
  return matchups
