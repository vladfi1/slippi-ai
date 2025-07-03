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
NAME_GROUPS = [
  ('Zain', 'zain', 'DontTestMe', 'ZAIN#0', 'DTM#664'),  # 13K replays
  ('Cody', 'iBDW', 'cody', 'IBDW#0', 'IBDW#734', 'JBDW#120'),  # 52K replays
  ('S2J', 'Mr Plow', 'John Redcorn', 'SSJ#998'),  # 3.6K replays
  ('Amsa', 'AMSA#0'),  # 26K replays
  ('Phillip AI', 'PHAI#591'),
  ('Hax', 'XX#02', 'HAX#472'),  # 85K replays
  ('Aklo', 'AKLO#381', 'AKLO#239'),  # 18K replays
  ('Morsecode', 'MORS#762'),  # 1.3K replays
  ('YCZ6', 'YCZ#667', 'YCZ#6', 'WH#0'),  # 2.6K replays
  ('BBB', 'BBB#960'),  # 3.6K replays
  ('Kodorin', 'KOD#0', '8#9'),  # 21K replays
  ('SFAT', 'SFAT#9', 'OHMA#175', 'SFAT#99', 'SFAT#783'),  # 10K replays
  ('Solobattle', '666#666', 'SOLO#735'),  # 19K replays
  ('Frenzy', 'FRNZ#141'),  # 20K replays
  ('Gosu', 'WIZZ#310'),  # 18K replays
  # Most Franz games are local with no name; for those we assume any Dr. Mario is Franz.
  ('Franz', 'XELA#158', 'PLATO#0'),  # 4K replays
  ('Isdsar', 'ISDS#767'),  # 7.7K replays
  ('Ginger', 'GING#345'),  # 20K replays
  ('DruggedFox', 'SAMI#669'),  # 1.3K replays
  ('KJH', 'KJH#23'),  # 9K replays
  ('BillyBoPeep', 'BILLY#0'),  # 1.5K replays
  ('Spark', 'ZAID#0'),
  ('Trif', 'TRIF#0', 'TRIF#268'),  # 9K replays
  ('Inky', 'INKY#398'),  # Sheik Player from Nova Scotia, 3.5K replays
  ('JChu', 'JCHU#536'),  # 3.5K replays
  ('Axe', 'AXE#845'),  # 800 replays
  ('M2K', 'KOTU#737', 'CHU#352'),  # 9K replays, mostly Sheik
  ('Siddward', 'SIDD#539'),  # Luigi main, 14K replays
  ('Kandayo', 'KAND#898'),  # Marth main, 4K replays
  ('Krudo', 'CHUG#596', 'CODY#007'),  # 9K replays
  ('Uhhei', 'SUTT#456'),  # Samus main, 7K replays
  ('FknSilver', 'THA#837', 'FUCKIN#1'),  # Samus main, 3K replays
  ('Salt', 'SALT#747'),  # 3K replays
  ('Zamu', 'A#9'),  # 2K replays

  # Don't have permission from these players yet.
  ('Ossify', 'OSSIFY#0'),  # 1K replays
  ('Moky', 'MOKY#475'),  # 3K replays

  # These players have asked not to be included in AI training.
  ('Mang0', 'mang', 'mang0', 'MANG#0'),
  ('Wizzrobe', 'WIZY#0'),
  ('Hungrybox', 'HBOX#305', 'hbox'),
]

NAME_MAP: dict[str, str] = {}
for first, *rest in NAME_GROUPS:
  for name in rest:
    NAME_MAP[name] = first

def normalize_name(name):
  return NAME_MAP.get(name, name)

KNOWN_PLAYERS = {group[0] for group in NAME_GROUPS}

def is_known_player(name):
  return normalize_name(name) in KNOWN_PLAYERS

def max_name_code(name_map: dict[str, int]) -> int:
  return (max(name_map.values()) + 1) if name_map else 0

def name_encoder(name_map: dict[str, int]):
  missing_name_code = max_name_code(name_map)
  def encode_name(name: str) -> int:
    # Do we want to normalize here?
    return name_map.get(normalize_name(name), missing_name_code)
  return encode_name


BANNED_NAMES = {
    # Have asked not to be included in AI training
    'Mang0', 'Wizzrobe', 'Hungrybox',

    # Haven't asked yet, so don't train on for now.
    'Ossify', 'Moky',

    'Phillip AI',  # This is us!
}
for name in BANNED_NAMES:
  assert name in KNOWN_PLAYERS, name

def is_banned_name(name: str) -> bool:
  return normalize_name(name) in BANNED_NAMES

for name, _ in PLAYER_MAINS:
  assert name in KNOWN_PLAYERS, name
