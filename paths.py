import os

PRJ_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'
DATA_PATH = PRJ_PATH + 'data/'

DB_PATH = DATA_PATH + 'melee_public_slp_dataset.sqlite3'

REPLAY_PATH = DATA_PATH + 'Game_20201113T162336.slp'
DATASET_PATH = DATA_PATH + 'training_data/'
COMPRESSED_PATH = DATA_PATH + 'AllCompressed/'

'''
    Vlad's paths
'''
# DB_PATH = os.path.expanduser('~/Scratch/melee_public_slp_dataset.sqlite3')

# REPLAY_PATH = os.path.expanduser('~/Slippi/Game_20201113T162336.slp')
# DATASET_PATH = '/media/vlad/Archive-2T-Nomad/Vlad/training_data/'
# COMPRESSED_PATH = os.path.expanduser('~/Scratch/AllCompressed/')
