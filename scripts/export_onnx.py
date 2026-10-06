"""Exports a JAX checkpoint to ONNX for running without jax or tensorflow.

The exported model can be used anywhere a checkpoint path is accepted, e.g.

  python scripts/eval_two.py --p2.ai.path=model.onnx

Requires the onnx-export extra: pip install .[jax,onnx-export]
"""

from absl import app, flags
import onnx

from slippi_ai import saving
from slippi_ai.jax import onnx_export

CHECKPOINT = flags.DEFINE_string('checkpoint', None, 'JAX checkpoint to export.', required=True)
OUTPUT = flags.DEFINE_string('output', None, 'Output path; defaults to <checkpoint>.onnx.')
BATCH_SIZE = flags.DEFINE_integer('batch_size', None, 'Fixed batch size; dynamic if unset.')

def main(_):
  output = OUTPUT.value or CHECKPOINT.value + '.onnx'
  state = saving.load_state_from_disk(CHECKPOINT.value)
  model = onnx_export.export_state(state, batch_size=BATCH_SIZE.value)
  onnx.save(model, output)
  print(f'Wrote {output}')

if __name__ == '__main__':
  app.run(main)
