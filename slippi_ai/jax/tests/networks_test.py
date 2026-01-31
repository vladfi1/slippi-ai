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
  np.testing.assert_allclose(t1, t2, atol=1e-6)

class NetworksTest(unittest.TestCase):

  @parameterized.expand(networks.SIMPLE_CONSTRUCTORS)
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

  def test_unroll_vs_step_sa_frame_tx(self):
    rngs = nnx.Rngs(0)
    embed_config = embed.EmbedConfig()
    state_action_embedding = embed_config.make_state_action_embedding(num_names=8)
    config = networks.FrameTransformer.default_config()
    config.update(
        num_layers=2,
        hidden_size=8,
        num_heads=2,
    )
    network = networks.FrameTransformer(
        rngs=rngs, embed_state_action=state_action_embedding, **config)
    self._test_unroll_vs_step_sa_net(network)

if __name__ == '__main__':
  unittest.main(failfast=True)
