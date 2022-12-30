import pathlib

_PATHS_PATH = pathlib.Path(__file__)
PRJ_PATH = _PATHS_PATH.parent.parent
DATA_PATH = PRJ_PATH / 'data'

# A demo game.
DEMO_PQ_REPLAY = DATA_PATH / 'pq/demo'

# A tiny demo checkpoint.
DEMO_CHECKPOINT = DATA_PATH / 'checkpoints/demo'
