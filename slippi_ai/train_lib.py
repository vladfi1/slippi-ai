import datetime
from typing import Iterator, Tuple
import os
import secrets

import numpy as np
import tensorflow as tf
import tree

import sacred

from slippi_ai import (
    data,
    utils,
)

from slippi_ai.learner import Learner

def get_experiment_tag():
  today = datetime.date.today()
  return f'{today.year}-{today.month}-{today.day}_{secrets.token_hex(8)}'

def get_experiment_directory():
  # create directory for tf checkpoints and other experiment artifacts
  expt_dir = f'experiments/{get_experiment_tag()}'
  os.makedirs(expt_dir, exist_ok=True)
  return expt_dir


class TrainManager:

  def __init__(
      self,
      learner: Learner,
      data_source: Iterator[Tuple[data.Batch, float]],
      step_kwargs={},
  ):
    self.learner = learner
    self.data_source = data_source
    self.hidden_state = learner.policy.initial_state(data_source.batch_size)
    self.step_kwargs = step_kwargs
    self.total_frames = 0
    self.data_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()

  def step(self) -> dict:
    with self.data_profiler:
      batch, epoch = next(self.data_source)
      # batch = sanitize_batch(batch)
    with self.step_profiler:
      stats, self.hidden_state = self.learner.compiled_step(
          batch, self.hidden_state, **self.step_kwargs)
    num_frames = np.sum(batch.game.action.repeat + 1)
    self.total_frames += num_frames
    stats.update(
        epoch=epoch,
        num_frames=num_frames,
        total_frames=self.total_frames,
    )
    return stats

def log_stats(ex: sacred.Experiment, stats, step=None, sep='.'):
  def log(path, value):
    if isinstance(value, tf.Tensor):
      value = value.numpy()
    if isinstance(value, np.ndarray):
      value = value.mean()
    key = sep.join(map(str, path))
    ex.log_scalar(key, value, step=step)
  tree.map_structure_with_path(log, stats)
