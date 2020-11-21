"""Test a trained model."""

import functools
import os
import time

from sacred import Experiment
from sacred.observers import MongoObserver

import numpy as np
import sonnet as snt
import tensorflow as tf

import melee

import data
import embed
from learner import Learner
import networks
import paths
import stats
import train_lib
import utils

LOG_INTERVAL = 10
SAVE_INTERVAL = 300

ex = Experiment('imitation_test')
ex.observers.append(MongoObserver())

@ex.config
def config():
  dataset = dict(
      data_dir=paths.COMPRESSED_PATH,  # Path to pickled dataset.
      subset=None,  # Subset to train on. Defaults to all files.
      test_ratio=.1,  # Fraction of dataset for testing.
  )
  data = dict(
      batch_size=32,
      unroll_length=64,
      compressed=True,
  )
  learner = Learner.DEFAULT_CONFIG
  saved_model_path = None

@ex.automain
def main(dataset, saved_model_path, _config, _log):
  policy = tf.saved_model.load(saved_model_path)
  flat_loss = policy.loss
  policy.loss = lambda *structs: flat_loss(*tf.nest.flatten(structs))
  learner = Learner(
      policy=policy,
      **_config['learner'])

  _, test_paths = data.train_test_split(**dataset)

  embed_controller = embed.embed_controller_discrete  # TODO: configure
  data_config = dict(_config['data'], embed_controller=embed_controller)
  test_data = data.DataSource(test_paths, **data_config)
  test_manager = train_lib.TrainManager(learner, test_data, dict(train=False))

  total_steps = 0

  for _ in range(1000):
    # now test
    test_loss = test_manager.step()
    test_loss = test_loss.numpy()
    ex.log_scalar('test.loss', test_loss, total_steps)
    print(f'test_loss={test_loss:.4f}')
