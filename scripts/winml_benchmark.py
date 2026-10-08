"""Times exported models on each Windows ML execution provider.

Experimental. Registers the providers in the Windows ML catalog (downloading
them if needed; vendor providers need Windows 11 24H2+), then times agent.step
at batch size 1 on every onnxruntime device, checking sampled actions against
CPU given the same inputs.

Setup (Python 3.10-3.13, plus the matching Windows App SDK Runtime):

  pip install -e . "wasdk-Microsoft.Windows.AI.MachineLearning[all]" \\
    wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap \\
    onnxruntime-windowsml
"""

import os
import time

from absl import app, flags
import numpy as np
import tree

from slippi_ai import data, onnx_policies, paths, saving, utils

MODELS = flags.DEFINE_list(
    'models', ['deployed_models/medium-v1-jax.onnx', 'deployed_models/diamond.onnx'],
    'Exported .onnx models.')
STEPS = flags.DEFINE_integer('steps', 300, 'Timed steps per model and device.')
WARMUP = flags.DEFINE_integer('warmup', 20, 'Untimed steps first.')
DOWNLOAD = flags.DEFINE_boolean(
    'download', True, 'Download catalog providers that are not installed.')


def register_catalog_providers(ort, winml):
  catalog = winml.ExecutionProviderCatalog.get_default()
  providers = list(catalog.find_all_providers())
  print(f'Catalog providers: {len(providers)}')
  for p in providers:
    print(f'  {p.name}: {p.ready_state}')
    if p.ready_state == winml.ExecutionProviderReadyState.NOT_PRESENT and not DOWNLOAD.value:
      continue
    start = time.time()
    result = p.ensure_ready_async().get()
    print(f'    ensure_ready: {result.status} ({time.time() - start:.1f} s)')
    if result.status == winml.ExecutionProviderReadyResultState.SUCCESS:
      print(f'    library: {p.library_path!r}')
      try:
        # Windows ML's own register calls don't reach Python's onnxruntime.
        ort.register_execution_provider_library(p.name, p.library_path)
      except Exception as e:  # pylint: disable=broad-except
        print(f'    could not register: {e}')


_NUMPY_TYPES = {
    'tensor(float)': np.float32, 'tensor(bool)': np.bool_,
    'tensor(int32)': np.int32, 'tensor(uint8)': np.uint8,
    'tensor(uint16)': np.uint16,
}


class CastingSession:
  """Casts packed inputs and outputs, for models whose graph I/O was retyped.

  TensorRT-RTX doesn't take uint16 inputs, so models can be rewritten to
  int32; the packed tensors keep their layout dtypes (outputs.<dtype>).
  """

  def __init__(self, session):
    self._session = session
    self._types = {i.name: _NUMPY_TYPES[i.type] for i in session.get_inputs()}

  def run(self, names, feed):
    feed = {k: v.astype(self._types[k], copy=False) for k, v in feed.items()}
    outputs = self._session.run(names, feed)
    return [o.astype(np.dtype(n.rsplit('.', 1)[1]), copy=False)
            for n, o in zip(names, outputs)]


def time_model(ort, path: str, replay):
  policy = saving.load_policy_from_state(saving.load_state_from_disk(path))
  devices = {}
  for d in ort.get_ep_devices():
    devices.setdefault(d.ep_name, []).append(d)

  reference = None
  for ep_name, ep_devices in devices.items():
    agent = policy.build_agent(
        1, name_code=0, rating=1500, seed=0, providers=[onnx_policies.CPU])
    if ep_name != onnx_policies.CPU:
      # Catalog providers are chosen by device, not by name.
      options = ort.SessionOptions()
      options.log_severity_level = 3
      options.add_provider_for_devices(ep_devices, {})
      start = time.time()
      agent.runner.session = ort.InferenceSession(policy.model, options)
      setup = time.time() - start
    else:
      setup = 0.
    agent.runner.session = CastingSession(agent.runner.session)

    times = []
    actions = []
    for t in range(WARMUP.value + STEPS.value):
      game = utils.map_nt(lambda x: x[t:t + 1], replay)
      start = time.perf_counter()
      output = agent.step(game, np.array([t == 0]))
      times.append(time.perf_counter() - start)
      actions.append(utils.map_nt(np.copy, output.controller_state))

    ms = np.array(times[WARMUP.value:]) * 1000
    mismatch = ''
    if reference is None:
      reference = actions
    else:
      differ = sum(
          not all(map(np.array_equal, tree.flatten(a), tree.flatten(b)))
          for a, b in zip(actions, reference))
      mismatch = f'  steps with different actions vs CPU: {differ}/{len(actions)}'
    vendors = ','.join(sorted({d.device.vendor for d in ep_devices}))
    print(f'{os.path.basename(path):24s} {ep_name:32s} {vendors:8s}'
          f' median {np.median(ms):5.1f} ms  p99 {np.percentile(ms, 99):5.1f} ms'
          f'  setup {setup:5.1f} s{mismatch}')


def main(_):
  from winui3.microsoft.windows.applicationmodel.dynamicdependency import bootstrap

  with bootstrap.initialize():
    import onnxruntime as ort
    import winui3.microsoft.windows.ai.machinelearning as winml

    print('onnxruntime', ort.__version__)
    register_catalog_providers(ort, winml)

    replay_path = os.path.join(
        paths.TOY_DATA_DIR, sorted(os.listdir(paths.TOY_DATA_DIR))[0])
    replay = data.read_table(replay_path, compressed=True)
    for path in MODELS.value:
      time_model(ort, path, replay)


if __name__ == '__main__':
  app.run(main)
