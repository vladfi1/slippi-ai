"""Track known player nametags and connect codes."""

from typing import Optional

import melee

DEFAULT_NAME = 'Master Player'
NAME_UNKNOWN = ''

def get_player(raw: str) -> Optional[str]:
  """The convention for personal dumps is Players/NAME/..."""
  if raw.startswith('Players/'):
    return raw.split('/')[1]
  return None

# Some player dumps have a lot of local games with no name or code.
# For such players, we assume any game with that player's main is them.
PLAYER_MAINS = {
    ('Solobattle', melee.Character.JIGGLYPUFF),
    ('Franz', melee.Character.DOC),
}

def name_from_metadata(player_meta: dict, raw: Optional[str] = None) -> str:
  netplay = player_meta['netplay']

  if netplay is not None:
    # Player dumps will have netplay codes, while the ranked-anonymized dumps
    # have an empty code and the name set to "Platinum/Diamond/Master Player".
    if netplay['code']:
      # Internally, connect codes use the Shift-JIS hash sign.
      return netplay['code'].replace('＃', '#')

    if netplay['name']:
      return netplay['name']

  if raw:
    player_name = get_player(raw)
    if player_name:
      char = melee.Character(player_meta['character'])
      if (player_name, char) in PLAYER_MAINS:
        return player_name

  # Offline games (e.g. tournaments)
  if player_meta['name_tag']:
    return player_meta['name_tag']

  return NAME_UNKNOWN

# TODO: we could scrape code -> ELO from the slippi website?

# TODO: put this in a json?
name_groups = [
  ('Mang0', 'mang', 'mang0', 'MANG#0'),
  ('Zain', 'zain', 'DontTestMe', 'ZAIN#0', 'DTM#664'),
  ('Cody', 'iBDW', 'cody', 'IBDW#0', 'IBDW#734', 'JBDW#120'),
  ('S2J', 'Mr Plow', 'John Redcorn', 'SSJ#998'),
  ('Amsa', 'AMSA#0'),
  ('Phillip AI', 'PHAI#591'),
  ('Hax', 'XX#02', 'HAX#472'),
  ('Aklo', 'AKLO#381', 'AKLO#239'),
  ('Morsecode', 'MORS#762'),
  ('YCZ6', 'YCZ#667', 'YCZ#6', 'WH#0'),
  ('BBB', 'BBB#960'),
  ('Kodorin', 'KOD#0', '8#9'),
  ('SFAT', 'SFAT#9', 'OHMA#175', 'SFAT#99', 'SFAT#783'),
  ('Solobattle', '666#666', 'SOLO#735'),  # TODO: many Solobattle games have no name
  ('Frenzy', 'FRNZ#141'),
  ('Gosu', 'WIZZ#310'),
  # Most Franz games are local with no name; for those we assume any Doctor Mario is Franz.
  ('Franz', 'XELA#158', 'PLATO#0'),
  ('Isdsar', 'ISDS#767'),
]

name_map = {}
for first, *rest in name_groups:
  for name in rest:
    name_map[name] = first

def normalize_name(name):
  return name_map.get(name, name)

def max_name_code(name_map: dict[str, int]) -> int:
  return (max(name_map.values()) + 1) if name_map else 0

def name_encoder(name_map: dict[str, int]):
  missing_name_code = max_name_code(name_map)
  def encode_name(name: str) -> int:
    # Do we want to normalize here?
    return name_map.get(normalize_name(name), missing_name_code)
  return encode_name


BANNED_NAMES = {
    'Mang0',  # Has asked not to be included in AI training
    'Zain',  # Has asked not to be included in AI training
    'Phillip AI',  # This is us!
}
for name in BANNED_NAMES:
  assert name in name_map.values(), name

def is_banned_name(name: str) -> bool:
  return normalize_name(name) in BANNED_NAMES

for name, _ in PLAYER_MAINS:
  assert name in name_map.values(), name
