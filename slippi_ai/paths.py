import pathlib

_PATHS_PATH = pathlib.Path(__file__)
PRJ_PATH = _PATHS_PATH.parent.parent
DATA_PATH = PRJ_PATH / 'data'

# A demo game.
DEMO_PATH = DATA_PATH / 'pq/demo'
