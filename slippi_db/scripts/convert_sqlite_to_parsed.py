"""Convert parsed.sqlite back to parsed.pkl.

Reverses the flattening done by convert_parsed_to_sqlite.py.

Usage: python slippi_db/scripts/convert_sqlite_to_parsed.py --input=parsed.sqlite --output=parsed.pkl
"""

import pickle
import sqlite3
from typing import Any

from absl import app, flags

INPUT = flags.DEFINE_string('input', None, 'Input SQLite database path', required=True)
OUTPUT = flags.DEFINE_string('output', None, 'Output parsed.pkl file', required=True)


def _parse_version(version_str: str | None) -> list[int] | None:
  if version_str is None:
    return None
  return [int(v) for v in version_str.split('.')]


def _build_player(row: dict, prefix: str) -> dict[str, Any] | None:
  """Reconstruct a player dict from flattened SQLite columns."""
  port = row.get(f'{prefix}_port')
  if port is None:
    return None

  netplay_name = row.get(f'{prefix}_netplay_name')
  netplay_code = row.get(f'{prefix}_netplay_code')
  netplay_suid = row.get(f'{prefix}_netplay_suid')

  if netplay_name is not None or netplay_code is not None or netplay_suid is not None:
    netplay = dict(name=netplay_name, code=netplay_code, suid=netplay_suid)
  else:
    netplay = None

  return dict(
    port=port,
    character=row.get(f'{prefix}_character'),
    type=row.get(f'{prefix}_type'),
    name_tag=row.get(f'{prefix}_name_tag'),
    netplay=netplay,
    damage_taken=row.get(f'{prefix}_damage_taken'),
  )


def sqlite_row_to_dict(row: dict) -> dict[str, Any]:
  """Convert a flattened SQLite row back to the nested parsed.pkl format."""
  result: dict[str, Any] = dict(
    name=row['name'],
    raw=row['raw'],
    valid=bool(row['valid']),
  )

  if not result['valid']:
    if row.get('parse_error'):
      result['reason'] = row['parse_error']
    return result

  # Optional fields present on valid rows.
  if row.get('slp_md5') is not None:
    result['slp_md5'] = row['slp_md5']
  if row.get('slp_size') is not None:
    result['slp_size'] = row['slp_size']
  if row.get('start_at') is not None:
    result['startAt'] = row['start_at']
  if row.get('played_on') is not None:
    result['playedOn'] = row['played_on']
  if row.get('last_frame') is not None:
    result['lastFrame'] = row['last_frame']
  if row.get('slippi_version') is not None:
    result['slippi_version'] = _parse_version(row['slippi_version'])
  if row.get('stage') is not None:
    result['stage'] = row['stage']
  if row.get('timer') is not None:
    result['timer'] = row['timer']
  if row.get('is_teams') is not None:
    result['is_teams'] = bool(row['is_teams'])
  if row.get('num_players') is not None:
    result['num_players'] = row['num_players']
  if row.get('winner') is not None:
    result['winner'] = row['winner']

  if row.get('is_training') is not None:
    result['is_training'] = bool(row['is_training'])
  if row.get('not_training_reason') is not None:
    result['not_training_reason'] = row['not_training_reason']

  if row.get('pq_size') is not None:
    result['pq_size'] = row['pq_size']
  if row.get('compression') is not None:
    result['compression'] = row['compression']

  # Reconstruct match dict.
  match_id = row.get('match_id')
  match_game = row.get('match_game')
  match_tiebreaker = row.get('match_tiebreaker')
  if match_id is not None or match_game is not None or match_tiebreaker is not None:
    result['match'] = dict(id=match_id, game=match_game, tiebreaker=match_tiebreaker)

  # Reconstruct players list.
  players = []
  for prefix in ('p0', 'p1'):
    player = _build_player(row, prefix)
    if player is not None:
      players.append(player)
  if players:
    result['players'] = players

  return result


def main(_):
  print(f"Loading {INPUT.value}...")
  conn = sqlite3.connect(f'file:{INPUT.value}?mode=ro', uri=True)
  conn.row_factory = sqlite3.Row

  cursor = conn.execute("SELECT COUNT(*) FROM replays")
  count = cursor.fetchone()[0]
  print(f"Found {count} rows.")

  cursor = conn.execute("SELECT * FROM replays")
  rows = [sqlite_row_to_dict(dict(r)) for r in cursor]
  conn.close()

  print(f"Converted {len(rows)} rows.")

  print(f"Writing {OUTPUT.value}...")
  with open(OUTPUT.value, 'wb') as f:
    pickle.dump(rows, f)

  print("Done.")


if __name__ == '__main__':
  app.run(main)
