"""Shared character name parsing for slippi_db scripts."""

from typing import Set

import melee

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
