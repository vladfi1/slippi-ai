"""Grab N replays per character on each legal stage, from a character pool.

Given a list of characters, for each character finds up to N replays on each
legal stage where that character is played and the opponent's character is
also drawn from the same list (this includes dittos, where the opponent
plays the same character). Candidates are chosen uniformly at random
(subject to a minimum Slippi version and a seed, for reproducibility)
rather than e.g. picking the longest game, to avoid biasing towards
outliers. Matched .slp files are copied directly out of their source Raw
zip archives into a single output zip -- without decompressing/
recompressing them (via `zip -U` / `zipmerge`).

Usage:
  python -m slippi_db.scripts.get_character_replays \
      --root=/path/to/dataset \
      --characters=fox,falco,marth \
      --num_replays=2 \
      --output=fox_falco_marth.zip
"""

import collections
from pathlib import Path
import random
import sqlite3
import typing as tp

from absl import app
from absl import flags
from melee import enums

from slippi_db import characters
from slippi_db import utils

ROOT = flags.DEFINE_string(
    'root', None,
    'Path to the dataset root directory (containing Raw/ and parsed.sqlite).',
    required=True)
CHARACTERS = flags.DEFINE_string(
    'characters', None,
    'Comma-separated character names (e.g. "fox,falco,marth").',
    required=True)
NUM_REPLAYS = flags.DEFINE_integer(
    'num_replays', 1,
    'Number of replays to pick for each (character, stage) pair.')
MIN_VERSION = flags.DEFINE_string(
    'min_version', '3.18.0',
    'Minimum Slippi version required (inclusive).')
SEED = flags.DEFINE_integer(
    'seed', 0, 'Random seed used to sample replays.')
OUTPUT = flags.DEFINE_string(
    'output', None,
    'Output zip path. Defaults to "<characters>_replays.zip" in the root dir.')


def _legal_stages() -> list[enums.Stage]:
  """Stages recognized as legal by this codebase (see preprocessing.py)."""
  stages = set()
  for stage_id in range(256):
    stage = enums.to_internal_stage(stage_id)
    if stage is not enums.Stage.NO_STAGE:
      stages.add(stage)
  return sorted(stages, key=lambda s: s.value)


LEGAL_STAGES = _legal_stages()


def parse_version(version_str: str) -> tuple[int, ...]:
  return tuple(int(v) for v in version_str.split('.'))


def character_name(char_id: int) -> str:
  for name, char in characters.NAME_TO_CHARACTER.items():
    if char.value == char_id:
      return name
  return str(char_id)


def find_candidates(
    parsed_db_path: str,
    character_ids: tp.Set[int],
    min_version: tuple[int, ...],
) -> list[sqlite3.Row]:
  """Finds valid replays where both players' characters are in character_ids."""
  char_list = list(character_ids)
  placeholders = ', '.join('?' * len(char_list))

  conn = sqlite3.connect(f'file:{parsed_db_path}?mode=ro', uri=True)
  conn.row_factory = sqlite3.Row
  try:
    cursor = conn.execute(
        f"""
        SELECT name, raw, stage, slippi_version, last_frame,
               p0_character, p1_character
        FROM replays
        WHERE valid = 1
          AND p0_character IN ({placeholders})
          AND p1_character IN ({placeholders})
        ORDER BY raw, name
        """,
        char_list + char_list)
    rows = cursor.fetchall()
  finally:
    conn.close()

  candidates = []
  for row in rows:
    if row['slippi_version'] is None:
      continue
    if parse_version(row['slippi_version']) < min_version:
      continue
    if enums.to_internal_stage(row['stage']) not in LEGAL_STAGES:
      continue
    candidates.append(row)
  return candidates


def select_replays(
    candidates: list[sqlite3.Row],
    character_ids: tp.Set[int],
    num_replays: int,
    seed: int,
) -> dict[tuple[int, enums.Stage], list[sqlite3.Row]]:
  """For each (character, stage), randomly samples up to num_replays rows.

  A single replay can satisfy the quota of both characters involved (e.g. a
  Fox vs. Falco game counts as a candidate for both Fox's and Falco's bucket
  on that stage).
  """
  by_bucket: dict[tuple[int, enums.Stage], list[sqlite3.Row]] = (
      collections.defaultdict(list))
  for row in candidates:
    stage = enums.to_internal_stage(row['stage'])
    row_characters = {row['p0_character'], row['p1_character']} & character_ids
    for char_id in row_characters:
      by_bucket[(char_id, stage)].append(row)

  rng = random.Random(seed)
  selected: dict[tuple[int, enums.Stage], list[sqlite3.Row]] = {}
  for key, rows in by_bucket.items():
    k = min(num_replays, len(rows))
    selected[key] = rng.sample(rows, k)
  return selected


def find_archive_member(raw_path: str, slp_name: str) -> str:
  """Finds the actual archive member name for a normalized .slp name."""
  for slp_file in utils.traverse_slp_files_zip(raw_path):
    if slp_file.name == slp_name:
      return slp_file.path
  raise FileNotFoundError(f'{slp_name} not found in {raw_path}')


def main(_):
  root = Path(ROOT.value)
  parsed_db_path = root / 'parsed.sqlite'
  if not parsed_db_path.exists():
    raise FileNotFoundError(f'parsed.sqlite not found at {parsed_db_path}')

  character_ids = characters.parse_characters(CHARACTERS.value)
  if not character_ids:
    raise ValueError(f'No characters specified: {CHARACTERS.value!r}')

  min_version = parse_version(MIN_VERSION.value)
  num_replays = NUM_REPLAYS.value

  char_names = sorted(character_name(c) for c in character_ids)
  print(f'Searching for replays among {", ".join(char_names)} '
        f'(slippi_version >= {MIN_VERSION.value})')

  candidates = find_candidates(str(parsed_db_path), character_ids, min_version)
  print(f'Found {len(candidates)} candidate replay(s).')

  selected_by_bucket = select_replays(
      candidates, character_ids, num_replays, SEED.value)

  # Dedup replays selected for multiple (character, stage) buckets.
  chosen_rows: dict[tuple[str, str], sqlite3.Row] = {}
  for char_id in sorted(character_ids):
    for stage in LEGAL_STAGES:
      rows = selected_by_bucket.get((char_id, stage), [])
      if len(rows) < num_replays:
        print(f'WARNING: only found {len(rows)}/{num_replays} replay(s) '
              f'for {character_name(char_id)} on {stage.name}')
      for row in rows:
        chosen_rows[(row['raw'], row['name'])] = row

  if not chosen_rows:
    print('No replays found. Exiting.')
    return

  raw_dir = root / 'Raw'
  sources_and_files: dict[str, list[str]] = collections.defaultdict(list)
  for (raw, name), row in chosen_rows.items():
    raw_path = str(raw_dir / raw)
    member = find_archive_member(raw_path, name)
    sources_and_files[raw].append(member)
    stage_name = enums.to_internal_stage(row['stage']).name
    p0_name = character_name(row['p0_character'])
    p1_name = character_name(row['p1_character'])
    print(f'{raw}::{member} (stage={stage_name}, {p0_name} vs {p1_name}, '
          f'v{row["slippi_version"]}, {row["last_frame"]} frames)')

  if OUTPUT.value:
    output = Path(OUTPUT.value)
  else:
    output = root / f'{"_".join(char_names)}_replays.zip'

  if output.exists():
    raise FileExistsError(f'Output file already exists: {output}')

  utils.copy_multi_zip_files(
      sources_and_files=[
          (str(raw_dir / raw), files)
          for raw, files in sources_and_files.items()
      ],
      dest_zip=str(output),
  )

  print(f'Wrote {len(chosen_rows)} replay(s) to {output}')


if __name__ == '__main__':
  app.run(main)
