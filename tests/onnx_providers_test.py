"""Checks that exported ONNX models give the same results on GPU as on CPU.

Runs an OnnxAgent on replay frames, and at every step runs the same packed
inputs through both a CPU session and a session with the given providers
(by default CUDA, with CUDA graphs for fixed-batch models). GPU kernels don't
match the CPU's bit for bit, so recurrent states are compared with a tolerance
and sampled actions by mismatch rate, since near-ties can sample differently.

With --reference_models, the CPU session runs those models instead, e.g. to
compare float16 exports against float32 ones (with looser tolerances).

Requires onnxruntime with the providers, e.g. pip install .[onnx-cuda]
"""

import os

from absl import app, flags
import numpy as np

from slippi_ai import data, paths, saving, utils
from slippi_ai import onnx_policies

flags.DEFINE_list('models', None, 'Exported .onnx models.', required=True)
flags.DEFINE_list(
    'reference_models', None,
    'Models to run on CPU for comparison, one per model; defaults to the models.')
flags.DEFINE_list('providers', [onnx_policies.CUDA], 'Providers to test.')
flags.DEFINE_boolean('cuda_graph', True, 'Use CUDA graphs.')
flags.DEFINE_integer('steps', 300, 'Number of frames to run.')
flags.DEFINE_float('max_state_diff', 1e-2, 'Max absolute recurrent state difference.')
flags.DEFINE_float('max_action_mismatch', 0.02, 'Max fraction of differing actions.')

FLAGS = flags.FLAGS


class ComparingRunner:
  """Runs both sessions, recording differences and returning the CPU outputs."""

  def __init__(self, reference: onnx_policies.SessionRunner, other: onnx_policies.SessionRunner):
    self.reference = reference
    self.other = other
    self.max_float_diff = 0.
    self.int_mismatches = 0
    self.int_total = 0

  def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ref = self.reference.run(inputs)
    out = self.other.run(inputs)
    for key, x in ref.items():
      y = out[key]
      if np.issubdtype(x.dtype, np.floating):
        diff = np.abs(x.astype(np.float32) - y.astype(np.float32))
        self.max_float_diff = max(self.max_float_diff, float(diff.max()))
      else:
        self.int_mismatches += int(np.sum(x != y))
        self.int_total += x.size
    return ref


def load_replay():
  path = os.path.join(paths.TOY_DATA_DIR, sorted(os.listdir(paths.TOY_DATA_DIR))[0])
  return data.read_table(path, compressed=True)


def load_policy(path: str) -> onnx_policies.OnnxPolicy:
  policy = saving.load_policy_from_state(saving.load_state_from_disk(path))
  assert isinstance(policy, onnx_policies.OnnxPolicy)
  return policy


def test_model(path: str, reference_path: str, replay):
  policy = load_policy(path)
  reference = load_policy(reference_path)
  if (policy.input_layout, policy.output_layout) != (
      reference.input_layout, reference.output_layout):
    raise ValueError(f'{path} and {reference_path} have different layouts.')
  batch_size = policy.batch_size or 1

  agent = reference.build_agent(
      batch_size, name_code=0, rating=1500, seed=0,
      providers=[onnx_policies.CPU])
  other = onnx_policies.SessionRunner(
      policy, batch_size, FLAGS.providers, FLAGS.cuda_graph)
  runner = ComparingRunner(agent.runner, other)
  agent.runner = runner

  for t in range(FLAGS.steps):
    game = utils.map_nt(lambda x: x[t:t + batch_size], replay)
    agent.step(game, np.array([t == 0] * batch_size))

  mismatch = runner.int_mismatches / max(runner.int_total, 1)
  print(f'{path} vs {reference_path} on CPU:'
        f' providers={other.providers[0]} cuda_graph={other.cuda_graph}'
        f' max float diff={runner.max_float_diff:.2e}'
        f' action mismatches={runner.int_mismatches}/{runner.int_total}')
  assert runner.max_float_diff <= FLAGS.max_state_diff, runner.max_float_diff
  assert mismatch <= FLAGS.max_action_mismatch, mismatch


def main(_):
  replay = load_replay()
  references = FLAGS.reference_models or FLAGS.models
  if len(references) != len(FLAGS.models):
    raise ValueError('Need one reference model per model.')
  for model, reference in zip(FLAGS.models, references):
    test_model(model, reference, replay)


if __name__ == '__main__':
  app.run(main)
