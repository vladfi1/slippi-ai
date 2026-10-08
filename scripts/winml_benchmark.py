"""Times exported models on each Windows ML execution provider.

Registers the providers in the Windows ML catalog (downloading them if
needed; vendor providers need Windows 11 24H2+), then times agent.step at
batch size 1 on every onnxruntime device, checking sampled actions against
CPU given the same inputs. TensorRT-RTX needs models exported with
--widen_ints (the default); otherwise it leaves them to the CPU.

Setup: pip install -e .[winml], plus the matching Windows App SDK Runtime
(see slippi_ai/winml.py).
"""

import os
import time

from absl import app, flags
import numpy as np
import tree

from slippi_ai import data, onnx_policies, paths, saving, utils, winml

MODELS = flags.DEFINE_list(
    'models', ['deployed_models/medium-v1-jax.onnx', 'deployed_models/diamond.onnx'],
    'Exported .onnx models.')
STEPS = flags.DEFINE_integer('steps', 300, 'Timed steps per model and device.')
WARMUP = flags.DEFINE_integer('warmup', 20, 'Untimed steps first.')
RUNTIME_CACHE = flags.DEFINE_string(
    'runtime_cache', None,
    "Folder for TensorRT-RTX's runtime cache (compiled kernels, ~2 MB per "
    'model), which roughly halves its session setup on later runs.')
DOWNLOAD = flags.DEFINE_boolean(
    'download', True, 'Download catalog providers that are not installed.')


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
      provider_options = {}
      if ep_name == 'NvTensorRTRTXExecutionProvider' and RUNTIME_CACHE.value:
        provider_options['nv_runtime_cache_path'] = RUNTIME_CACHE.value
      options.add_provider_for_devices(ep_devices, provider_options)
      start = time.time()
      agent.runner.session = ort.InferenceSession(policy.model, options)
      setup = time.time() - start
    else:
      setup = 0.

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
  import onnxruntime as ort

  print('onnxruntime', ort.__version__)
  print('Windows ML providers:', winml.initialize(download=DOWNLOAD.value))

  replay_path = os.path.join(
      paths.TOY_DATA_DIR, sorted(os.listdir(paths.TOY_DATA_DIR))[0])
  replay = data.read_table(replay_path, compressed=True)
  for path in MODELS.value:
    time_model(ort, path, replay)


if __name__ == '__main__':
  app.run(main)
