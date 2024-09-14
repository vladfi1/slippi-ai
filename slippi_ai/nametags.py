"""Track known player nametags and connect codes."""


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
]

name_map = {}
for first, *rest in name_groups:
  for name in rest:
    name_map[name] = first

def normalize_name(name):
  return name_map.get(name, name)

def name_encoder(name_encoding: dict[str, int]):
  missing_name_code = (max(name_encoding.values()) + 1) if name_encoding else 0
  def encode_name(name: str) -> int:
    return name_encoding.get(normalize_name(name), missing_name_code)
  return encode_name
