"""Helpers shared by the Pallas (Triton) kernels in lstm_pallas, ffw_pallas
and embed_lookup."""
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


def enabled() -> bool:
  """Whether the kernels can run: on GPU or in interpret mode, with a jaxlib
  that includes Triton. Read at trace time."""
  if plgpu is None:
    return False
  return jax.default_backend() == 'gpu' or interpret()


def match_vma(x: jax.Array, ref: jax.Array) -> jax.Array:
  """Under shard_map, give x (e.g. a replicated weight) the varying axes of
  ref (the activations) so every operand of a kernel has the same type."""
  varying = tuple(sorted(jax.typeof(ref).mat.varying - jax.typeof(x).mat.varying))
  return jax.lax.pcast(x, varying, to='varying') if varying else x
