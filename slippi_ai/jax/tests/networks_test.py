import unittest
from parameterized import parameterized

import jax
import numpy as np
import jax.numpy as jnp
from flax import nnx

from slippi_ai.jax import networks

default_network_config = networks.default_config()

def default_network(name: str, rngs: nnx.Rngs, input_size: int = 10) -> networks.Network:
  return networks.CONSTRUCTORS[name](rngs=rngs, input_size=input_size, **default_network_config[name])

def assert_tensors_close(t1: jnp.ndarray, t2: jnp.ndarray):
  np.testing.assert_allclose(t1, t2)

class NetworksTest(unittest.TestCase):

  @parameterized.expand(networks.CONSTRUCTORS)
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
      step_outputs, step_final_state = networks.dynamic_rnn(
          network._step_with_reset, (inputs, reset), initial_state)

      assert_tensors_close(unroll_outputs, step_outputs)
      jax.tree.map(assert_tensors_close, unroll_final_state, step_final_state)

if __name__ == '__main__':
  unittest.main(failfast=True)
