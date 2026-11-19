"""Helpers shared by the Pallas (Triton) kernels in lstm_pallas, ffw_pallas
and embed_lookup."""
import functools
import os

import jax

try:
  from jax.experimental.pallas import triton as plgpu
except ImportError:
  # jaxlib builds without Triton (e.g. on Windows) can't build the kernels;
  # `enabled` is then False and callers use their plain XLA paths.
  plgpu = None


def interpret() -> bool:
  """SLIPPI_PALLAS_INTERPRET=1 runs the kernels in Pallas interpret mode, on
  any backend, for testing without a GPU. Read at trace time."""
  return os.environ.get('SLIPPI_PALLAS_INTERPRET', '0') == '1'


@functools.cache
def _register_gpu_kinds() -> None:
  """Pallas Triton looks GPUs up by device_kind in a fixed list (e.g. no
  consumer Blackwell like "NVIDIA GeForce RTX 5070 Ti") and fails lowering with
  "No supported GPU devices found" otherwise. Register any unknown local CUDA
  device from its reported compute capability."""
  try:
    from jax._src.pallas.triton import gpu_info
  except ImportError:
    return
  for device in jax.local_devices():
    cc = getattr(device, 'compute_capability', None)
    if not isinstance(cc, str) or cc.startswith('gfx'):
      continue
    kind = device.device_kind
    if (gpu_info.gpu_version_from_device_kind(kind) is not None
        or kind in gpu_info.registry):
      continue
    major, minor = cc.split('.')
    gpu_info.registry[kind] = functools.partial(
        gpu_info.GpuInfo, gpu_version=None, arch_name=cc,
        compute_capability=int(major) * 10 + int(minor))


def enabled() -> bool:
  """Whether the kernels can run: on GPU or in interpret mode, with a jaxlib
  that includes Triton. Read at trace time."""
  if plgpu is None:
    return False
  if jax.default_backend() == 'gpu':
    _register_gpu_kinds()
    return True
  return interpret()


def match_vma(x: jax.Array, ref: jax.Array) -> jax.Array:
  """Under shard_map, give x (e.g. a replicated weight) the varying axes of
  ref (the activations) so every operand of a kernel has the same type."""
  varying = tuple(sorted(jax.typeof(ref).mat.varying - jax.typeof(x).mat.varying))
  return jax.lax.pcast(x, varying, to='varying') if varying else x
