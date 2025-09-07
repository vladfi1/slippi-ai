import pathlib

_FILE_PATH = pathlib.Path(__file__)
PACKAGE_PATH = _FILE_PATH.parent
DATA_PATH = PACKAGE_PATH / 'data'

# A tiny demo checkpoint.
DEMO_CHECKPOINT = DATA_PATH / 'checkpoints/demo'

TOY_DATASET = DATA_PATH / 'toy_dataset'
TOY_DATA_DIR = TOY_DATASET / 'games'
TOY_META_PATH = TOY_DATASET / 'meta.json'
