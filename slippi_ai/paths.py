import pathlib

_FILE_PATH = pathlib.Path(__file__)
PACKAGE_PATH = _FILE_PATH.parent
DATA_PATH = PACKAGE_PATH / 'data'

# A tiny demo checkpoint.
CHECKPOINTS_DIR = DATA_PATH / 'checkpoints'
DEMO_IMITATION_CHECKPOINT = CHECKPOINTS_DIR / 'demo'
JAX_IMITATION_CHECKPOINT = CHECKPOINTS_DIR / 'jax_demo'

JAX_VF_CHECKPOINT = CHECKPOINTS_DIR / 'jax_vf_demo'
JAX_POLICY_CHECKPOINT = CHECKPOINTS_DIR / 'jax_policy_demo'
JAX_MERGED_CHECKPOINT = CHECKPOINTS_DIR / 'jax_merged_demo'

# Checkpoints from before the frame-skip refactor (residual controller head).
JAX_IMITATION_CHECKPOINT_LEGACY = CHECKPOINTS_DIR / 'jax_demo_legacy'
JAX_VF_CHECKPOINT_LEGACY = CHECKPOINTS_DIR / 'jax_vf_demo_legacy'
JAX_POLICY_CHECKPOINT_LEGACY = CHECKPOINTS_DIR / 'jax_policy_demo_legacy'

JAX_Q_FN_CKPT = CHECKPOINTS_DIR / 'jax_q_fn'
JAX_NASH_Q_FN_CKPT = CHECKPOINTS_DIR / 'jax' / 'nash_q_fn'

TOY_DATASET = DATA_PATH / 'toy_dataset'
TOY_DATA_DIR = TOY_DATASET / 'games'
TOY_META_PATH = TOY_DATASET / 'meta.json'
TOY_BATCH_SNAPSHOT = TOY_DATASET / 'batch_snapshot.pkl'

AGENT_OUTPUTS_DIR = DATA_PATH / 'agent_outputs'
