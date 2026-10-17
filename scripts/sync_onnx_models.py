"""Exports every model in a directory to ONNX, keeping the output in sync.

Each model <input_dir>/<name> is exported to <output_dir>/<name>.onnx. A model
is re-exported when its file, following symlinks, is newer than its export, or
when it resolves to a different file than it was exported from, e.g. after
re-pointing a symlink. Recreating a symlink to the same file doesn't count.
The resolved sources are recorded in <output_dir>/.sources.json. Exports whose
model has disappeared from the input directory are deleted.

TF checkpoints are converted to JAX in memory first, which needs tensorflow.

  python scripts/sync_onnx_models.py

Requires the onnx-export extra: pip install .[jax,onnx-export]
"""

import json
import os
import sys
import typing as tp

# Exporting doesn't need a GPU; don't grab one from training jobs.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from absl import app, flags, logging
import onnx

from slippi_ai import onnx_policies, saving
from slippi_ai.agents import Platform
from slippi_ai.jax import networks, onnx_export

INPUT_DIR = flags.DEFINE_string(
    'input_dir', 'deployed_models', 'Directory of models to export.')
OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', 'onnx_models', 'Directory to write .onnx models to.')
FORCE = flags.DEFINE_boolean('force', False, 'Re-export every model.')
REEXPORT_CHAR_ACTION_JOINT = flags.DEFINE_boolean(
    'reexport_char_action_joint', False,
    'Also re-export models whose export uses the enhanced embed\'s joint '
    'character-action embedding, which exported wrongly with widened ints '
    'before 168db0a8.')
DRY_RUN = flags.DEFINE_boolean(
    'dry_run', False, 'Only print what would be exported and deleted.')
BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 1,
    'Fixed batch size, or 0 for a dynamic batch dimension. Play uses 1, and '
    'CUDA graphs need a fixed size; the GUI only lists batch size 1 models.')
WEIGHT_DTYPE = flags.DEFINE_enum(
    'weight_dtype', 'float16', ['float32', 'float16'],
    'Storage dtype of the weights; float16 halves the file size but still '
    'computes in float32.')
WIDEN_INTS = flags.DEFINE_boolean(
    'widen_ints', True,
    'Use int32 instead of 8 and 16 bit integers, which TensorRT-RTX (Windows '
    'ML) lacks. No slower on CPU or CUDA.')

SUFFIX = '.onnx'
TMP_SUFFIX = '.tmp'
SOURCES = '.sources.json'


def load_sources(output_dir: str) -> dict[str, str]:
  """Each export's resolved source path, see save_sources."""
  path = os.path.join(output_dir, SOURCES)
  if not os.path.exists(path):
    return {}
  with open(path) as f:
    return json.load(f)


def save_sources(output_dir: str, sources: dict[str, str]):
  path = os.path.join(output_dir, SOURCES)
  with open(path + TMP_SUFFIX, 'w') as f:
    json.dump(sources, f, indent=2, sort_keys=True)
  os.replace(path + TMP_SUFFIX, path)


def list_models(input_dir: str) -> dict[str, str]:
  models = {}
  for name in sorted(os.listdir(input_dir)):
    path = os.path.join(input_dir, name)
    if name.startswith('.'):
      continue
    if not os.path.exists(path):
      logging.warning(f'Skipping {name}: broken symlink.')
      continue
    if not os.path.isfile(path):
      continue
    models[name] = path
  return models


def uses_char_action_joint(onnx_path: str) -> bool:
  """Whether an export uses the enhanced embed's joint char-action embedding."""
  config = onnx_policies.read_metadata_from_file(onnx_path)['config']
  embed_config = config['network']['embed']
  if embed_config['name'] != 'enhanced':
    return False
  enhanced = networks.EnhancedEmbedModule.default_config()
  enhanced.update(embed_config.get('enhanced', {}))
  return enhanced['use_learned_action'] and enhanced['use_char_action_joint']


def needs_export(path: str, output: str, source: tp.Optional[str]) -> bool:
  if FORCE.value or not os.path.exists(output):
    return True
  if os.path.getmtime(output) < os.path.getmtime(path):
    return True
  # Exports from before the sources were recorded count as up to date.
  if source is not None and source != os.path.realpath(path):
    return True
  return REEXPORT_CHAR_ACTION_JOINT.value and uses_char_action_joint(output)


class Unsupported(Exception):
  """A model that can't be exported yet, skipped with a warning."""


def check_tf_supported(config: dict):
  # The JAX embedding has no MLP items type to convert to.
  items_type = config['embed']['items']['type']
  if getattr(items_type, 'value', items_type) == 'mlp':
    raise Unsupported('TF checkpoint with an MLP items embedding')


def load_jax_state(path: str) -> dict:
  state = saving.load_state_from_disk(path)
  config = saving.upgrade_config(state['config'])
  if saving.get_platform(config) is Platform.TF:
    check_tf_supported(config)
    # The converter lives next to this script and needs tensorflow.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import convert_tf_checkpoint_to_jax
    state = convert_tf_checkpoint_to_jax.convert_state(state)
  return state


def export(path: str, output: str):
  state = load_jax_state(path)
  model = onnx_export.export_state(
      state, batch_size=BATCH_SIZE.value or None,
      weight_dtype=WEIGHT_DTYPE.value, widen_ints=WIDEN_INTS.value)
  # Write then rename, so an interrupted export never looks up to date.
  tmp = output + TMP_SUFFIX
  onnx.save(model, tmp)
  os.replace(tmp, output)


def main(_):
  input_dir = INPUT_DIR.value
  output_dir = OUTPUT_DIR.value
  os.makedirs(output_dir, exist_ok=True)

  models = list_models(input_dir)
  sources = load_sources(output_dir)

  to_export = []
  for name, path in models.items():
    output = os.path.join(output_dir, name + SUFFIX)
    if needs_export(path, output, sources.get(name)):
      to_export.append((name, path, output))
    elif name not in sources:
      sources[name] = os.path.realpath(path)

  expected = {name + SUFFIX for name in models}
  to_delete = []
  for name in sorted(os.listdir(output_dir)):
    if name.endswith(TMP_SUFFIX) or (
        name.endswith(SUFFIX) and name not in expected):
      to_delete.append(os.path.join(output_dir, name))

  logging.info(f'{len(models)} models: {len(to_export)} to export, '
               f'{len(to_delete)} to delete.')

  for path in to_delete:
    logging.info(f'Deleting {path}')
    if not DRY_RUN.value:
      os.remove(path)
  sources = {name: src for name, src in sources.items() if name in models}

  skipped = []
  failed = []
  for i, (name, path, output) in enumerate(to_export):
    logging.info(f'[{i + 1}/{len(to_export)}] Exporting {name}')
    if DRY_RUN.value:
      continue
    try:
      # Resolve first, in case the symlink changes during the export.
      source = os.path.realpath(path)
      export(path, output)
      sources[name] = source
      save_sources(output_dir, sources)
    except Unsupported as e:
      logging.info(f'Skipping {name}: {e}')
      skipped.append(name)
    except Exception:  # Keep going; report failures at the end.
      logging.exception(f'Failed to export {name}')
      failed.append(name)

  if not DRY_RUN.value:
    save_sources(output_dir, sources)

  if skipped:
    logging.warning(
        f'Skipped {len(skipped)} unsupported models: {", ".join(skipped)}')
  if failed:
    logging.error(f'Failed to export {len(failed)} models: {", ".join(failed)}')
    sys.exit(1)

if __name__ == '__main__':
  app.run(main)
