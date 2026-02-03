"""Convert parsed.pkl to SQLite database.

Usage: python slippi_db/scripts/convert_parsed_to_sqlite.py --input=parsed.pkl --output=parsed.db
"""

import pickle
import sqlite3
from typing import Any

from absl import app, flags
import tqdm

INPUT = flags.DEFINE_string('input', None, 'Input parsed.pkl file', required=True)
OUTPUT = flags.DEFINE_string('output', None, 'Output SQLite database path', required=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS replays (
    -- Core fields (name + raw identify the source file)
    name TEXT NOT NULL,
    raw TEXT NOT NULL,
    slp_md5 TEXT,
    slp_size INTEGER,
    start_at TEXT,
    played_on TEXT,
    last_frame INTEGER,
    slippi_version TEXT,
    stage INTEGER,
    timer INTEGER,
    is_teams INTEGER,
    num_players INTEGER,
    winner INTEGER,

    -- Filtering fields
    valid INTEGER NOT NULL,
    is_training INTEGER,
    not_training_reason TEXT,
    parse_error TEXT,  -- error message if valid=False

    -- Parsed output fields
    pq_size INTEGER,
    compression TEXT,

    -- Match fields (flattened)
    match_id TEXT,
    match_game INTEGER,
    match_tiebreaker INTEGER,

    -- Player 0 fields
    p0_port INTEGER,
    p0_character INTEGER,
    p0_type INTEGER,
    p0_name_tag TEXT,
    p0_netplay_name TEXT,
    p0_netplay_code TEXT,
    p0_netplay_suid TEXT,
    p0_damage_taken REAL,

    -- Player 1 fields
    p1_port INTEGER,
    p1_character INTEGER,
    p1_type INTEGER,
    p1_name_tag TEXT,
    p1_netplay_name TEXT,
    p1_netplay_code TEXT,
    p1_netplay_suid TEXT,
    p1_damage_taken REAL,

    PRIMARY KEY (name, raw)
);

CREATE INDEX IF NOT EXISTS idx_replays_slp_md5 ON replays(slp_md5);
CREATE INDEX IF NOT EXISTS idx_replays_raw ON replays(raw);
CREATE INDEX IF NOT EXISTS idx_replays_is_training ON replays(is_training);
CREATE INDEX IF NOT EXISTS idx_replays_valid ON replays(valid);
"""

COLUMNS = [
    'name', 'raw', 'slp_md5', 'slp_size', 'start_at', 'played_on',
    'last_frame', 'slippi_version', 'stage', 'timer', 'is_teams',
    'num_players', 'winner', 'valid', 'is_training', 'not_training_reason',
    'parse_error', 'pq_size', 'compression',
    'match_id', 'match_game', 'match_tiebreaker',
    'p0_port', 'p0_character', 'p0_type', 'p0_name_tag',
    'p0_netplay_name', 'p0_netplay_code', 'p0_netplay_suid', 'p0_damage_taken',
    'p1_port', 'p1_character', 'p1_type', 'p1_name_tag',
    'p1_netplay_name', 'p1_netplay_code', 'p1_netplay_suid', 'p1_damage_taken',
]

INSERT_SQL = f"""
INSERT OR REPLACE INTO replays ({', '.join(COLUMNS)})
VALUES ({', '.join('?' * len(COLUMNS))})
"""


def format_version(version: tuple | list | None) -> str | None:
    if version is None:
        return None
    return '.'.join(str(v) for v in version)


def extract_player(player: dict, prefix: str) -> dict[str, Any]:
    """Extract flattened player fields."""
    netplay = player.get('netplay') or {}
    return {
        f'{prefix}_port': player.get('port'),
        f'{prefix}_character': player.get('character'),
        f'{prefix}_type': player.get('type'),
        f'{prefix}_name_tag': player.get('name_tag'),
        f'{prefix}_netplay_name': netplay.get('name'),
        f'{prefix}_netplay_code': netplay.get('code'),
        f'{prefix}_netplay_suid': netplay.get('suid'),
        f'{prefix}_damage_taken': player.get('damage_taken'),
    }


def row_to_tuple(row: dict) -> tuple:
    """Convert a parsed.pkl row dict to a tuple for SQLite insertion."""
    match = row.get('match') or {}
    players = row.get('players') or []

    flat = {
        'name': row.get('name'),
        'raw': row.get('raw'),
        'slp_md5': row.get('slp_md5'),
        'slp_size': row.get('slp_size'),
        'start_at': row.get('startAt'),
        'played_on': row.get('playedOn'),
        'last_frame': row.get('lastFrame'),
        'slippi_version': format_version(row.get('slippi_version')),
        'stage': row.get('stage'),
        'timer': row.get('timer'),
        'is_teams': row.get('is_teams'),
        'num_players': row.get('num_players'),
        'winner': row.get('winner'),
        'valid': row.get('valid'),
        'is_training': row.get('is_training'),
        'not_training_reason': row.get('not_training_reason'),
        'parse_error': row.get('reason'),  # error message for invalid replays
        'pq_size': row.get('pq_size'),
        'compression': row.get('compression'),
        'match_id': match.get('id'),
        'match_game': match.get('game'),
        'match_tiebreaker': match.get('tiebreaker'),
    }

    # Add player fields
    if len(players) >= 1:
        flat.update(extract_player(players[0], 'p0'))
    else:
        flat.update(extract_player({}, 'p0'))

    if len(players) >= 2:
        flat.update(extract_player(players[1], 'p1'))
    else:
        flat.update(extract_player({}, 'p1'))

    return tuple(flat[col] for col in COLUMNS)


def main(_):
    print(f"Loading {INPUT.value}...")
    with open(INPUT.value, 'rb') as f:
        rows: list[dict[str, Any]] = pickle.load(f)
    print(f"Loaded {len(rows)} rows.")

    print(f"Creating {OUTPUT.value}...")
    conn = sqlite3.connect(OUTPUT.value)
    conn.executescript(SCHEMA)

    print("Converting rows...")
    tuples = [row_to_tuple(row) for row in tqdm.tqdm(rows)]

    print(f"Inserting {len(tuples)} rows into database...")
    conn.executemany(INSERT_SQL, tuples)
    conn.commit()

    # Verify
    cursor = conn.execute("SELECT COUNT(*) FROM replays")
    count = cursor.fetchone()[0]
    print(f"Inserted {count} rows.")

    cursor = conn.execute("SELECT COUNT(*) FROM replays WHERE is_training = 1")
    training_count = cursor.fetchone()[0]
    print(f"Training replays: {training_count}")

    conn.close()
    print("Done.")


if __name__ == '__main__':
    app.run(main)
