"""Filter a dataset to only include games with a specific character.

Creates a new dataset folder with symlinks to the original parsed games.

Usage:
  python -m slippi_db.scripts.filter_by_character \
    --root=/path/to/dataset \
    --characters=fox

  python -m slippi_db.scripts.filter_by_character \
    --root=/path/to/dataset \
    --characters=sheik,zelda \
    --both

  python -m slippi_db.scripts.filter_by_character \
    --root=/path/to/dataset \
    --characters=marth \
    --output=/custom/output/path
"""

import json
import os
from pathlib import Path
from typing import Set

from absl import app
from absl import flags
import melee

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "root", None,
    "Path to the dataset root directory (containing meta.json and Parsed/).",
    required=True)
flags.DEFINE_string(
    "characters", None,
    "Comma-separated list of character names (e.g., 'fox', 'sheik,zelda').",
    required=True)
flags.DEFINE_string(
    "output", None,
    "Output directory path. Defaults to 'Dataset-CHARACTER' in current directory.")
flags.DEFINE_bool(
    "both", False,
    "Require both players to play one of the specified characters.")
flags.DEFINE_integer(
    "limit", None,
    "Maximum number of replays to include. If None, include all matching replays.")

# Build character name -> ID mapping
NAME_TO_CHARACTER = {c.name.lower(): c for c in melee.Character}

# Also allow common aliases
CHARACTER_ALIASES = {
  "falcon": "cptfalcon",
  "puff": "jigglypuff",
  "jiggs": "jigglypuff",
  "icies": "popo",
  "ics": "popo",
  "ic": "popo",
  "iceclimbers": "popo",
  "ice_climbers": "popo",
  "gnw": "gameandwatch",
  "g&w": "gameandwatch",
  "game_and_watch": "gameandwatch",
  "gw": "gameandwatch",
  "yl": "ylink",
  "younglink": "ylink",
  "young_link": "ylink",
  "drmario": "doc",
  "dr_mario": "doc",
  "doctor_mario": "doc",
  "mew2": "mewtwo",
  "m2": "mewtwo",
  "ganon": "ganondorf",
}
for v in CHARACTER_ALIASES.values():
  assert v in NAME_TO_CHARACTER, f"Alias target '{v}' not in character list"


def parse_characters(chars_str: str) -> Set[int]:
  """Parse a comma-separated string of character names into a set of character IDs."""
  char_ids = set()
  for name in chars_str.lower().split(","):
    name = name.strip()
    if not name:
      continue
    # Check aliases first
    canonical = CHARACTER_ALIASES.get(name, name)
    if canonical not in NAME_TO_CHARACTER:
      available = sorted(NAME_TO_CHARACTER.keys())
      raise ValueError(
        f"Unknown character: '{name}'. Available characters: {available}")
    char_ids.add(NAME_TO_CHARACTER[canonical].value)
  return char_ids


def get_game_characters(row: dict) -> Set[int]:
  """Get the set of character IDs present in a game."""
  return {player["character"] for player in row["players"]}


def filter_games(
  meta_rows: list, char_ids: Set[int], require_both: bool = False
) -> list:
  """Filter games to those containing the specified characters.

  Args:
    meta_rows: List of metadata rows from meta.json.
    char_ids: Set of character IDs to filter for.
    require_both: If True, both players must play one of the specified characters.
                  If False (default), at least one player must play a specified character.

  Returns:
    Filtered list of metadata rows.
  """
  filtered = []
  for row in meta_rows:
    game_chars = get_game_characters(row)
    matching_chars = game_chars & char_ids

    if require_both:
      # Both players must be playing specified characters
      if game_chars <= char_ids:
        filtered.append(row)
    else:
      # At least one player must be playing a specified character
      if matching_chars:
        filtered.append(row)

  return filtered


def create_filtered_dataset(
  root: Path,
  char_ids: Set[int],
  output_dir: Path,
  require_both: bool = False,
) -> None:
  """Create a filtered dataset with symlinks to original parsed games.

  Args:
    root: Path to the original dataset root.
    char_ids: Set of character IDs to filter for.
    output_dir: Path to the output dataset folder.
    require_both: If True, both players must play specified characters.
  """
  # Read original meta.json
  meta_path = root / "meta.json"
  if not meta_path.exists():
    raise FileNotFoundError(f"meta.json not found at {meta_path}")

  with open(meta_path) as f:
    meta_rows = json.load(f)

  print(f"Loaded {len(meta_rows)} games from {meta_path}")

  # Filter games
  filtered_rows = filter_games(meta_rows, char_ids, require_both)
  print(f"Filtered to {len(filtered_rows)} games ({len(filtered_rows)/len(meta_rows)*100:.1f}%)")

  if FLAGS.limit is not None and len(filtered_rows) > FLAGS.limit:
    filtered_rows = filtered_rows[:FLAGS.limit]
    print(f"Limited to {FLAGS.limit} games")

  if not filtered_rows:
    print("No games match the filter criteria. Exiting.")
    return

  # Create output directory structure
  output_dir.mkdir(parents=True, exist_ok=True)
  games_dir = output_dir / "games"
  games_dir.mkdir(exist_ok=True)

  # Find the Parsed directory
  parsed_dir = root / "Parsed"
  if not parsed_dir.exists():
    raise FileNotFoundError(f"Parsed directory not found at {parsed_dir}")

  # Create symlinks to parsed games
  created = 0
  skipped = 0
  missing = 0

  for row in filtered_rows:
    slp_md5 = row["slp_md5"]
    source = parsed_dir / slp_md5
    target = games_dir / slp_md5

    if not source.exists():
      missing += 1
      continue

    if target.exists() or target.is_symlink():
      skipped += 1
      continue

    # Create relative symlink for portability
    rel_source = os.path.relpath(source, games_dir)
    target.symlink_to(rel_source)
    created += 1

  print(f"Created {created} symlinks, skipped {skipped} existing, {missing} missing source files")

  # Write filtered meta.json
  output_meta_path = output_dir / "meta.json"
  with open(output_meta_path, "w") as f:
    json.dump(filtered_rows, f, indent=2)

  print(f"Wrote filtered metadata to {output_meta_path}")


def generate_output_name(char_ids: Set[int]) -> str:
  """Generate a dataset folder name from character IDs."""
  # Get character names and sort them
  char_names = []
  for char_id in sorted(char_ids):
    for char in melee.Character:
      if char.value == char_id:
        char_names.append(char.name.capitalize())
        break

  return "Dataset-" + "-".join(char_names)


def main(_):
  root = Path(FLAGS.root)

  # Parse characters
  char_ids = parse_characters(FLAGS.characters)

  char_names = [c.name for c in melee.Character if c.value in char_ids]
  print(f"Filtering for characters: {', '.join(char_names)}")

  # Determine output directory
  if FLAGS.output:
    output_dir = Path(FLAGS.output)
  else:
    output_dir = root / generate_output_name(char_ids)
    # output_dir = Path.cwd() / generate_output_name(char_ids)

  print(f"Output directory: {output_dir}")

  # Create filtered dataset
  create_filtered_dataset(
    root=root.resolve(),
    char_ids=char_ids,
    output_dir=output_dir.resolve(),
    require_both=FLAGS.both,
  )

  print("Done!")


if __name__ == "__main__":
  app.run(main)
