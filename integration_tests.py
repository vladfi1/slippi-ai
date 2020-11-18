import itertools
import unittest

import policies
import networks
from train import ex as train_ex

class TrainTest(unittest.TestCase):
  def train(self, policy='default', network='mlp'):
    run = train_ex.run(config_updates=dict(
        num_epochs=1,
        epoch_time=0,
        save_interval=0,
        policy=dict(name=policy),
        network=dict(name=network),
    ))
    self.assertEqual(run.status, 'COMPLETED')

  def test_train(self):
    for policy, network in itertools.product(
        policies.CONSTRUCTORS, networks.CONSTRUCTORS):
      kwargs = dict(policy=policy, network=network)
      with self.subTest(**kwargs):
        self.train(**kwargs)

if __name__ == '__main__':
  unittest.main()
