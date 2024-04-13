import pathlib

_FILE_PATH = pathlib.Path(__file__)
PROJECT_PATH = _FILE_PATH.parent.parent
DATA_PATH = PROJECT_PATH / 'data'

# A demo game.
DEMO_PQ_REPLAY = DATA_PATH / 'pq/demo'

# A tiny demo checkpoint.
DEMO_CHECKPOINT = DATA_PATH / 'checkpoints/demo'

TOY_DATASET = DATA_PATH / 'toy_dataset'
TOY_DATA_DIR = TOY_DATASET / 'games'
TOY_META_PATH = TOY_DATASET / 'meta.json'
