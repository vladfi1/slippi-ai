name_groups = [
  ('Mang0', 'mang', 'mang0'),
  ('Zain', 'zain', 'DontTestMe'),
  ('iBDW', 'cody'),
  ('S2J', 'Mr Plow', 'John Redcorn'),
]

name_map = {}
for first, *rest in name_groups:
  for name in rest:
    name_map[name] = first

def normalize_name(name):
  return name_map.get(name, name)

def name_encoder(name_encoding: dict[str, int]):
  missing_name_code = max(name_encoding.values()) + 1
  def encode_name(name: str) -> int:
    return name_encoding.get(normalize_name(name), missing_name_code)
  return encode_name
