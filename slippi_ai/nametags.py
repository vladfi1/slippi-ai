"""Track known player nametags and connect codes."""

DEFAULT_NAME = 'Master Player'

def name_from_metadata(player_meta: dict) -> str:
  netplay = player_meta['netplay']

  # Offline games (e.g. tournaments)
  if netplay is None:
    return player_meta['name_tag']

  # Player dumps will have netplay codes, while the ranked-anonymized dumps
  # have an empty code and the name set to "Platinum/Diamond/Master Player".
  if netplay['code']:
    # Internally, connect codes use the Shift-JIS hash sign.
    return netplay['code'].replace('＃', '#')
  return netplay['name']

# TODO: we could scrape code -> ELO from the slippi website?

# TODO: put this in a json?
name_groups = [
  ('Mang0', 'mang', 'mang0', 'MANG#0'),
  ('Zain', 'zain', 'DontTestMe', 'ZAIN#0'),
  ('Cody', 'iBDW', 'cody', 'IBDW#0', 'IBDW#734', 'JBDW#120'),
  ('S2J', 'Mr Plow', 'John Redcorn', 'SSJ#998'),
  ('Amsa', 'AMSA#0'),
  ('Phillip AI', 'PHAI#591'),
  # TODO: Aklo and Hax codes
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
    'Phillip AI',  # This is us!
}
for name in BANNED_NAMES:
  assert name in name_map.values(), name

def is_banned_name(name: str) -> bool:
  return normalize_name(name) in BANNED_NAMES
