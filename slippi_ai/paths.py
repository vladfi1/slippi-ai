import pathlib

_FILE_PATH = pathlib.Path(__file__)
PACKAGE_PATH = _FILE_PATH.parent
DATA_PATH = PACKAGE_PATH / 'data'

# A tiny demo checkpoint.
CHECKPOINTS_DIR = DATA_PATH / 'checkpoints'
DEMO_IMITATION_CHECKPOINT = CHECKPOINTS_DIR / 'demo'

JAX_Q_FN_CKPT = CHECKPOINTS_DIR / 'jax_q_fn'
JAX_IMITATION_CKPT = CHECKPOINTS_DIR / 'jax_demo'

TOY_DATASET = DATA_PATH / 'toy_dataset'
TOY_DATA_DIR = TOY_DATASET / 'games'
TOY_META_PATH = TOY_DATASET / 'meta.json'

AGENT_OUTPUTS_DIR = DATA_PATH / 'agent_outputs'
