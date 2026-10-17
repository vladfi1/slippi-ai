import unittest
from parameterized import parameterized

import jax
jax.config.update('jax_default_matmul_precision', 'highest')
import numpy as np
import jax.numpy as jnp
from flax import nnx

from slippi_ai.jax import networks, embed, jax_utils

default_network_config = networks.default_config()

def default_network(name: str, rngs: nnx.Rngs, input_size: int = 10) -> networks.Network:
  return networks.SIMPLE_CONSTRUCTORS[name](rngs=rngs, input_size=input_size, **default_network_config[name])

def assert_tensors_close(t1: jnp.ndarray, t2: jnp.ndarray):
  np.testing.assert_allclose(t1, t2, atol=1e-6, rtol=1e-5)

class NetworksTest(unittest.TestCase):

  # Mamba2 ignores resets in the middle of an unroll; see test_mamba2.
  @parameterized.expand(
      [name for name in networks.SIMPLE_CONSTRUCTORS if name != 'mamba2'])
  def test_unroll_vs_step(self, name='mlp'):
    rngs = nnx.Rngs(0)
    unroll_length = 8
    batch_size = 1
    input_size = 10
    network = default_network(name, rngs, input_size)
    initial_state = network.initial_state(batch_size, rngs)

    for _ in range(5):
      inputs = rngs.normal((unroll_length, batch_size, input_size))
      reset = rngs.bernoulli(0.1, (unroll_length, batch_size))

      unroll_outputs, unroll_final_state = network.unroll(
          inputs, reset, initial_state)
      step_outputs, step_final_state = jax_utils.dynamic_rnn(
          network._step_with_reset, (inputs, reset), initial_state)

      assert_tensors_close(unroll_outputs, step_outputs)
      jax.tree.map(assert_tensors_close, unroll_final_state, step_final_state)

  @parameterized.expand([
      ('linear', None),
      ('gated', None),
      ('gated', 2),
      ('depthwise', 2),
  ])
  def test_tcn_unroll_vs_step(self, block, dilation_base):
    rngs = nnx.Rngs(0)
    batch_size = 2
    input_size = 10
    config = dict(default_network_config['tcn'])
    config.update(
        num_layers=2, memory_size=2, block=block, dilation_base=dilation_base)
    network = networks.TemporalConvolution(
        rngs=rngs, input_size=input_size, **config)
    # Randomize the zero-initialized output projections.
    for _, param in nnx.iter_graph(network):
      if isinstance(param, nnx.Param):
        param.value = 0.1 * rngs.normal(param.value.shape)
    initial_state = network.initial_state(batch_size, rngs)

    for _ in range(3):
      inputs = rngs.normal((12, batch_size, input_size))
      reset = rngs.bernoulli(0.1, (12, batch_size))

      unroll_outputs, unroll_final_state = network.unroll(
          inputs, reset, initial_state)
      step_outputs, step_final_state = jax_utils.dynamic_rnn(
          network._step_with_reset, (inputs, reset), initial_state)

      np.testing.assert_allclose(
          unroll_outputs, step_outputs, atol=1e-4, rtol=1e-4)
      jax.tree.map(assert_tensors_close, unroll_final_state, step_final_state)
      initial_state = unroll_final_state

  @parameterized.expand([
      # (chunk_size, unroll_length, dtype)
      (32, 12, 'float32'),  # single padded chunk
      (4, 10, 'float32'),  # several chunks, the last one padded
      (4, 8, 'bfloat16'),
  ])
  def test_mamba2_unroll_vs_step(self, chunk_size, unroll_length, dtype):
    rngs = nnx.Rngs(0)
    batch_size = 2
    input_size = 10
    config = dict(default_network_config['mamba2'])
    config.update(
        hidden_size=32, num_layers=2, state_size=16, head_dim=8,
        chunk_size=chunk_size)
    network = networks.Mamba2(rngs=rngs, input_size=input_size, **config)
    network = jax_utils.cast_params_to_dtype(network, jnp.dtype(dtype))
    initial_state = network.initial_state(batch_size, rngs)

    for i in range(3):
      inputs = rngs.normal((unroll_length, batch_size, input_size)).astype(dtype)
      # Only resets at the start of the unroll are supported.
      reset = jnp.zeros((unroll_length, batch_size), dtype=bool)
      reset = reset.at[0].set(rngs.bernoulli(0.5, (batch_size,)) | (i == 0))

      unroll_outputs, unroll_final_state = network.unroll(
          inputs, reset, initial_state)
      step_outputs, step_final_state = jax_utils.dynamic_rnn(
          network._step_with_reset, (inputs, reset), initial_state)

      tol = 1e-4 if dtype == 'float32' else 3e-2
      close = lambda a, b: np.testing.assert_allclose(
          np.asarray(a, np.float32), np.asarray(b, np.float32),
          atol=tol, rtol=tol)
      close(unroll_outputs, step_outputs)
      jax.tree.map(close, unroll_final_state, step_final_state)
      self.assertEqual(
          jax_utils.struct_dtype(unroll_final_state), jnp.dtype(dtype))
      initial_state = unroll_final_state

  def test_mamba2_bf16_activations(self):
    rngs = nnx.Rngs(0)
    batch_size = 4
    unroll_length = 12
    config = dict(default_network_config['mamba2'])
    config.update(hidden_size=32, num_layers=2, state_size=16, head_dim=8)
    network = networks.Mamba2(rngs=rngs, input_size=10, **config)
    network = jax_utils.cast_params_to_dtype(network, jnp.bfloat16)
    initial_state = network.initial_state(batch_size, rngs)
    inputs = rngs.normal((unroll_length, batch_size, 10)).astype(jnp.bfloat16)
    reset = jnp.zeros((unroll_length, batch_size), dtype=bool)

    outputs, _ = network.unroll(inputs, reset, initial_state)
    for _, layer in nnx.iter_graph(network):
      if isinstance(layer, networks.Mamba2Layer):
        layer._act_dtype = jnp.float32
    fp32_outputs, _ = network.unroll(inputs, reset, initial_state)

    np.testing.assert_allclose(
        np.asarray(outputs, np.float32), np.asarray(fp32_outputs, np.float32),
        atol=3e-2, rtol=3e-2)

  @parameterized.expand([(32,), (4,)])
  def test_mamba2_matches_library(self, chunk_size):
    import mamba2_jax
    rngs = nnx.Rngs(0)
    batch_size = 3
    unroll_length = 10
    cfg = mamba2_jax.Mamba2Config(
        hidden_size=32, state_size=16, head_dim=8, chunk_size=chunk_size)
    layer = networks.Mamba2Layer(rngs, cfg)

    # Nonzero initial state to exercise the state carry.
    initial_state = jax.tree.map(
        lambda x: rngs.normal(x.shape),
        layer.initial_state(batch_size, rngs))
    inputs = rngs.normal((unroll_length, batch_size, cfg.hidden_size))
    reset = jnp.zeros((unroll_length, batch_size), dtype=bool)

    outputs, (conv_state, ssm_state) = layer.unroll(inputs, reset, initial_state)

    lib_outputs, lib_conv_state, lib_ssm_state = layer.mixer(
        layer.norm(jnp.swapaxes(inputs, 0, 1)), *initial_state)
    np.testing.assert_allclose(
        outputs, jnp.swapaxes(lib_outputs, 0, 1), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(conv_state, lib_conv_state, atol=1e-5)
    np.testing.assert_allclose(ssm_state, lib_ssm_state, atol=1e-4, rtol=1e-4)

  @parameterized.expand([
      # (dilation, memory_size, unroll_length)
      (1, 1, 8),
      (1, 3, 8),
      (2, 2, 7),
      (3, 2, 4),  # unroll shorter than the history
      (4, 1, 9),
      (2, 0, 5),
  ])
  def test_dilated_fixed_memory_unroll_vs_step(
      self, dilation, memory_size, unroll_length):
    rngs = nnx.Rngs(0)
    batch_size = 4
    input_size = 3
    network = networks.DilatedFixedMemoryBlock(
        rngs=rngs,
        input_size=input_size,
        output_size=5,
        dilation=dilation,
        memory_size=memory_size,
    )

    for i in range(5):
      # Use a nonzero initial state to exercise the state <-> history mapping.
      initial_state = jax.tree.map(
          lambda x: rngs.normal(x.shape),
          network.initial_state(batch_size, rngs))
      inputs = rngs.normal((unroll_length, batch_size, input_size))
      reset = rngs.bernoulli(0.3, (unroll_length, batch_size))
      if i == 0:
        reset = reset.at[0].set(True)

      unroll_outputs, unroll_final_state = network.unroll(
          inputs, reset, initial_state)
      step_outputs, step_final_state = jax_utils.dynamic_rnn(
          network._step_with_reset, (inputs, reset), initial_state)

      assert_tensors_close(unroll_outputs, step_outputs)
      jax.tree.map(assert_tensors_close, unroll_final_state, step_final_state)

  def _test_unroll_vs_step_sa_net(self, network: networks.StateActionNetwork):
    rngs = nnx.Rngs(0)
    unroll_length = 8
    batch_size = 1
    inputs = network.dummy(shape=(unroll_length, batch_size))

    # def step_with_reset(step_and_reset, state):
    #   step_inputs, reset = step_and_reset
    #   return network.step_with_reset(step_inputs, reset, state)

    for _ in range(5):
      initial_state = network.initial_state(batch_size, rngs)
      # TODO: randomize inputs as well
      reset = rngs.bernoulli(0.1, (unroll_length, batch_size))

      unroll_outputs, unroll_final_state = network.unroll(
          inputs, reset, initial_state)
      step_outputs, step_final_state = jax_utils.dynamic_rnn(
          network._step_with_reset, (inputs, reset), initial_state)

      jax.tree.map(assert_tensors_close, unroll_outputs, step_outputs)
      jax.tree.map(assert_tensors_close, unroll_final_state, step_final_state)

if __name__ == '__main__':
  unittest.main(failfast=True)
