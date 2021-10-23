import os
import pathlib

_PATHS_PATH = pathlib.Path(__file__)
PRJ_PATH = _PATHS_PATH.parent.parent
DATA_PATH = PRJ_PATH / 'data'

DB_PATH = DATA_PATH / 'melee_public_slp_dataset.sqlite3'
REPLAY_PATH = DATA_PATH / 'Game_20201113T162336.slp'
MULTISHINE_PATH = DATA_PATH / 'multishine/multishine.slp.pkl'
DATASET_PATH = DATA_PATH / 'training_data/'
COMPRESSED_PATH = DATA_PATH / 'AllCompressed/'
