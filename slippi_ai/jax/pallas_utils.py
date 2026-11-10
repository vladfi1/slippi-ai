"""Helpers shared by the Pallas (Triton) kernels in lstm_pallas and ffw_pallas."""
import os

import jax


def interpret() -> bool:
  """SLIPPI_PALLAS_INTERPRET=1 runs the kernels in Pallas interpret mode, on
  any backend, for testing without a GPU. Read at trace time."""
  return os.environ.get('SLIPPI_PALLAS_INTERPRET', '0') == '1'


def match_vma(x: jax.Array, ref: jax.Array) -> jax.Array:
  """Under shard_map, give x (e.g. a replicated weight) the varying axes of
  ref (the activations) so every operand of a kernel has the same type."""
  varying = tuple(sorted(jax.typeof(ref).mat.varying - jax.typeof(x).mat.varying))
  return jax.lax.pcast(x, varying, to='varying') if varying else x
