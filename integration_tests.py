import itertools
import unittest

import controller_heads
import networks
from train import ex as train_ex

class TrainTest(unittest.TestCase):
  def train(self, controller_head='independent', network='mlp'):
    run = train_ex.run(config_updates=dict(
        num_epochs=1,
        epoch_time=0,
        save_interval=0,
        controller_head=dict(name=controller_head),
        network=dict(name=network),
    ))
    self.assertEqual(run.status, 'COMPLETED')

  def test_train(self):
    for controller_head, network in itertools.product(
        controller_heads.CONSTRUCTORS, networks.CONSTRUCTORS):
      kwargs = dict(controller_head=controller_head, network=network)
      with self.subTest(**kwargs):
        self.train(**kwargs)

if __name__ == '__main__':
  unittest.main()
