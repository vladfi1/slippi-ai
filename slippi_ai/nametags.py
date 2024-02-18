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
