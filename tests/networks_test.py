import unittest
from parameterized import parameterized

import numpy as np
import tensorflow as tf

from slippi_ai import (
    learner, networks, paths, data, tf_utils, embed
)

def assert_tensors_close(t1, t2):
  np.testing.assert_allclose(t1.numpy(), t2.numpy())

def default_network(name):
  return networks.CONSTRUCTORS[name](**networks.DEFAULT_CONFIG[name])

embed_game = embed.make_game_embedding()
embed_controller = embed.get_controller_embedding(axis_spacing=16)

def default_data_source():
  dataset_config = data.DatasetConfig(
      data_dir=paths.TOY_DATA_DIR,
      meta_path=paths.TOY_META_PATH,
  )
  return data.DataSource(
      replays=data.replays_from_meta(dataset_config),
      batch_size=1,
      unroll_length=8,
      embed_game=embed_game,
      embed_controller=embed_controller,
      compressed=True,
  )

def get_inputs(data_source: data.DataSource):
  batch = next(data_source)[0]

  # from Learner
  bm_gamestate = batch.frames
  tm_gamestate: embed.StateAction = tf.nest.map_structure(
    learner.swap_axes, bm_gamestate.state_action)

  return data_source.embed_game(tm_gamestate.state)

class NetworksTest(unittest.TestCase):

  @parameterized.expand(networks.CONSTRUCTORS)
  def test_unroll_vs_step(self, name='mlp'):
    network = default_network(name)
    initial_state = network.initial_state(1)
    data_source = default_data_source()

    for _ in range(5):
      inputs = get_inputs(data_source)

      unroll_outputs, unroll_final_state = network.unroll(inputs, initial_state)
      step_outputs, step_final_state = tf_utils.dynamic_rnn(network.step, inputs, initial_state)

      assert_tensors_close(unroll_outputs, step_outputs)
      tf.nest.map_structure(assert_tensors_close, unroll_final_state, step_final_state)

if __name__ == '__main__':
  unittest.main(failfast=True)
